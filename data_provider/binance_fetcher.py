# -*- coding: utf-8 -*-
"""
BinanceFetcher — 币安现货行情数据源（加密货币，24/7）

数据源：币安公开行情 REST API
- 默认使用公开镜像 ``https://data-api.binance.vision``：
  全球可访问、无需 API Key，可规避部分地区对 ``api.binance.com`` 的封锁。
  如需改用官方端点，设置环境变量 ``BINANCE_BASE_URL=https://api.binance.com``。
- 日线：``GET /api/v3/klines``
- 实时：``GET /api/v3/ticker/24hr``

仅提供行情（K线/24小时行情），不涉及账户与下单；加密货币 24/7 交易。
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from src.services.market_symbol_utils import binance_display_name, is_crypto_symbol, is_bstock_symbol

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote

logger = logging.getLogger(__name__)

_BINANCE_DEFAULT_BASE_URL = "https://data-api.binance.vision"
# 合约公开数据（资金费率/持仓量/多空比）：与现货不同域。
# ``fapi.binance.com`` 对美国 IP 常返回 451（Unavailable For Legal Reasons），
# 因此按顺序尝试同路径镜像域名；全部失败再退到 CoinGecko / Bybit。
_BINANCE_FAPI_BASE_URL = "https://fapi.binance.com"
_BINANCE_FAPI_MIRROR_BASE_URL = "https://www.binance.com"
_COINGECKO_DERIVATIVES_URL = "https://api.coingecko.com/api/v3/derivatives"
_BYBIT_ACCOUNT_RATIO_URL = "https://api.bybit.com/v5/market/account-ratio"
_KLINE_LIMIT = 1000
_DAY_MS = 86_400_000


def _safe_float(value) -> Optional[float]:
    """容错转 float，失败返回 None。"""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class BinanceFetcher(BaseFetcher):
    """加密货币（币安现货）行情数据源。"""

    name = "BinanceFetcher"
    # 加密货币专用数据源：仅承接 crypto 市场（由 DataFetcherManager 按市场过滤），
    # 对其他市场不参与竞争，priority 取较大值以便在通用循环中排最后。
    priority = 99

    def __init__(self, base_url: Optional[str] = None):
        self._base_url = (
            (base_url or os.getenv("BINANCE_BASE_URL") or _BINANCE_DEFAULT_BASE_URL)
            .strip()
            .rstrip("/")
        )
        env_fapi = (os.getenv("BINANCE_FAPI_BASE_URL") or "").strip().rstrip("/")
        self._fapi_urls = (
            (env_fapi,)
            if env_fapi
            else (_BINANCE_FAPI_BASE_URL, _BINANCE_FAPI_MIRROR_BASE_URL)
        )
        self._coingecko_cache: Optional[list] = None
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "daily_stock_analysis/1.0"})

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _to_millis(date_str: str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _get_json(self, path: str, params: dict, timeout: int = 20):
        resp = self._session.get(f"{self._base_url}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _get_fapi_json(self, path: str, params: dict, timeout: int = 15):
        """合约公开数据（best-effort）。

        按顺序尝试若个域名：``fapi.binance.com`` 对美国 IP 常返回 451，
        而 ``www.binance.com`` 提供同路径同结构的数据。全部失败才抛出，
        异常信息带上每个域名的失败原因，便于定位地域限制。
        """
        errors = []
        for base in self._fapi_urls:
            try:
                resp = self._session.get(f"{base}{path}", params=params, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001 - try next mirror
                errors.append(f"{base}: {e}")
        raise RuntimeError("; ".join(errors) if errors else "no fapi base url configured")

    # ------------------------------------------------------------------
    # BaseFetcher 接口
    # ------------------------------------------------------------------
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = (stock_code or "").strip().upper()
        if not is_crypto_symbol(symbol):
            raise DataFetchError(f"[BinanceFetcher] {stock_code} 不是受支持的加密货币交易对")

        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": self._to_millis(start_date),
            "endTime": self._to_millis(end_date) + _DAY_MS - 1,
            "limit": _KLINE_LIMIT,
        }
        try:
            data = self._get_json("/api/v3/klines", params)
        except Exception as e:  # noqa: BLE001 - surfaced as DataFetchError for failover
            raise DataFetchError(f"[BinanceFetcher] {symbol} 请求失败: {e}") from e

        if not isinstance(data, list):
            raise DataFetchError(f"[BinanceFetcher] {symbol} 返回异常: {data}")
        if not data:
            raise DataFetchError(f"[BinanceFetcher] {symbol} 无数据（{start_date}~{end_date}）")

        rows = []
        for k in data:
            # [openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, ...]
            try:
                rows.append(
                    {
                        "date": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "amount": float(k[7]),
                    }
                )
            except (IndexError, TypeError, ValueError) as e:
                raise DataFetchError(f"[BinanceFetcher] {symbol} K线解析失败: {e}") from e

        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"])
        df.index.name = None  # 避免与 _normalize_data 重新添加的 'date' 列冲突
        return df.drop(columns=["date"])

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df.index).date
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        df["pct_chg"] = (df["close"].pct_change() * 100).fillna(0).round(2)
        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df["close"]
        df["code"] = stock_code

        keep = ["code"] + STANDARD_COLUMNS
        return df[[col for col in keep if col in df.columns]]

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        symbol = (stock_code or "").strip().upper()
        if not is_crypto_symbol(symbol):
            return None

        try:
            data = self._get_json("/api/v3/ticker/24hr", {"symbol": symbol}, timeout=15)
        except Exception as e:  # noqa: BLE001 - realtime is best-effort
            logger.warning(f"[BinanceFetcher] {symbol} 实时行情失败: {e}")
            return None

        if not isinstance(data, dict) or "lastPrice" not in data:
            return None

        def _f(key: str) -> Optional[float]:
            try:
                return float(data.get(key))
            except (TypeError, ValueError):
                return None

        change_pct = _f("priceChangePercent")
        volume = _f("volume")

        return UnifiedRealtimeQuote(
            code=symbol,
            name=binance_display_name(symbol),
            source=RealtimeSource.BINANCE,
            market="crypto",
            currency="USDT",
            price=_f("lastPrice"),
            change_pct=round(change_pct, 2) if change_pct is not None else None,
            change_amount=_f("priceChange"),
            volume=int(volume) if volume is not None else None,
            amount=_f("quoteVolume"),
            open_price=_f("openPrice"),
            high=_f("highPrice"),
            low=_f("lowPrice"),
            pre_close=_f("prevClosePrice"),
        )

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        symbol = (stock_code or "").strip().upper()
        if not is_crypto_symbol(symbol):
            return None
        return binance_display_name(symbol)

    # ------------------------------------------------------------------
    # 资金面（现货净主动买入 + 合约资金费率/持仓量/多空比）
    # ------------------------------------------------------------------
    def _coingecko_derivatives(self) -> list:
        """懒加载并缓存 CoinGecko 衍生品快照（一次请求覆盖全部 symbol）。"""
        if self._coingecko_cache is not None:
            return self._coingecko_cache
        try:
            resp = self._session.get(
                _COINGECKO_DERIVATIVES_URL,
                params={"include_tickers": "unchecked"},
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json()
            self._coingecko_cache = data if isinstance(data, list) else []
        except Exception as e:  # noqa: BLE001 - optional metric
            logger.info("[BinanceFetcher] CoinGecko derivatives 不可用: %s", e)
            self._coingecko_cache = []
        return self._coingecko_cache

    def _coingecko_futures_metrics(self, symbol: str) -> dict:
        """从 CoinGecko 取币安永续的 funding / OI（单位已归一化到币安口径）。

        - CoinGecko ``funding_rate`` 是百分比，/100 后与币安小数口径一致
        - CoinGecko ``open_interest`` 是美元名义值，按 ``index`` 价折回基础币数量，
          与币安 ``/fapi/v1/openInterest``（币本位数）可比

        注意：CoinGecko 只覆盖加密永续合约，代币化美股（bStock）无数据。
        """
        for item in self._coingecko_derivatives():
            if not isinstance(item, dict) or item.get("symbol") != symbol:
                continue
            if str(item.get("market") or "").strip().lower() != "binance (futures)":
                continue
            out: dict = {}
            funding = _safe_float(item.get("funding_rate"))
            if funding is not None:
                out["funding_rate"] = funding / 100.0
            oi_usd = _safe_float(item.get("open_interest"))
            index_price = _safe_float(item.get("index"))
            if oi_usd is not None and index_price:
                out["open_interest"] = oi_usd / index_price
            return out
        return {}

    def _bybit_long_short_ratio(self, symbol: str) -> Optional[float]:
        """Bybit 合约账户多空比（buyRatio / sellRatio）。

        币安多空比接口不可用时的市场情绪代理指标——口径为“另一家交易所”，
        因此仅在币安不可用时使用，并通过 ``futures.source`` 标注来源。
        """
        try:
            resp = self._session.get(
                _BYBIT_ACCOUNT_RATIO_URL,
                params={"category": "linear", "symbol": symbol, "period": "1d", "limit": 1},
                timeout=15,
            )
            resp.raise_for_status()
            items = ((resp.json() or {}).get("result") or {}).get("list") or []
        except Exception as e:  # noqa: BLE001 - optional metric
            logger.info("[BinanceFetcher] Bybit account-ratio 不可用: %s", e)
            return None
        if not items:
            return None
        buy = _safe_float(items[0].get("buyRatio"))
        sell = _safe_float(items[0].get("sellRatio"))
        if buy is None or not sell:
            return None
        return buy / sell

    def _collect_futures_metrics(self, symbol: str) -> tuple:
        """采集合约公开指标（best-effort，多源回退）。

        Returns:
            (metrics, errors)
            - metrics: 成功获取的字段；含 ``source`` 说明实际生效的上游
            - errors: 不可用原因，向上游汇报，避免“只有 None 没有理由”
        """
        metrics: dict = {}
        errors: list = []
        used_sources: list = []

        # bStock（代币化美股）只有现货，没有币安永续合约：
        # 实测 /fapi/v1/exchangeInfo 的 775 个 TRADING 永续中不存在 bStock 交易对，
        # premiumIndex/openInterest 一律返回 -1121 Invalid symbol。
        # 因此直接跳过，避免每个 bStock 白跑 3 次必然失败的请求。
        if is_bstock_symbol(symbol):
            return {}, []

        def _binance_metric(label: str, path: str, params: dict, extract) -> Optional[float]:
            try:
                data = self._get_fapi_json(path, params)
            except Exception as e:  # noqa: BLE001 - fall back to non-binance source
                errors.append(f"futures_{label}: {e}")
                return None
            try:
                return extract(data)
            except (TypeError, ValueError, IndexError, KeyError) as e:
                errors.append(f"futures_{label}: parse {e}")
                return None

        funding = _binance_metric(
            "funding_rate",
            "/fapi/v1/premiumIndex",
            {"symbol": symbol},
            lambda d: (
                _safe_float(d.get("lastFundingRate")) if isinstance(d, dict) else None
            ),
        )
        open_interest = _binance_metric(
            "open_interest",
            "/fapi/v1/openInterest",
            {"symbol": symbol},
            lambda d: _safe_float(d.get("openInterest")) if isinstance(d, dict) else None,
        )
        long_short_ratio = _binance_metric(
            "long_short_ratio",
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": "1d", "limit": 1},
            lambda d: _safe_float(d[0].get("longShortRatio")) if isinstance(d, list) and d else None,
        )

        if funding is not None:
            metrics["funding_rate"] = funding
            used_sources.append("币安合约")
        if open_interest is not None:
            metrics["open_interest"] = open_interest
            used_sources.append("币安合约")
        if long_short_ratio is not None:
            metrics["long_short_ratio"] = long_short_ratio
            used_sources.append("币安合约")

        # 二级：CoinGecko 聚合（币安域名被地域封锁时的兜底，仅 crypto）
        if funding is None or open_interest is None:
            cg = self._coingecko_futures_metrics(symbol)
            if funding is None and cg.get("funding_rate") is not None:
                metrics["funding_rate"] = cg["funding_rate"]
                used_sources.append("CoinGecko")
            if open_interest is None and cg.get("open_interest") is not None:
                metrics["open_interest"] = cg["open_interest"]
                used_sources.append("CoinGecko")
            if not cg:
                errors.append(f"futures_coingecko: {symbol} 无匹配的永续合约数据")

        # 二级：多空比用 Bybit 账户多空比代理
        if long_short_ratio is None:
            bybit_ratio = self._bybit_long_short_ratio(symbol)
            if bybit_ratio is not None:
                metrics["long_short_ratio"] = bybit_ratio
                used_sources.append("Bybit")
            else:
                errors.append("futures_long_short_ratio: 币安与 Bybit 均不可用")

        if used_sources:
            metrics["source"] = "/".join(dict.fromkeys(used_sources))

        for reason in errors:
            logger.info("[BinanceFetcher] %s 合约指标不可用: %s", symbol, reason)

        # 只把“最终仍然缺失”的指标作为错误上报：已成功回退的（例如币安 451
        # 但 CoinGecko 已补齐）不应污染上层 errors 与提示词。
        unreported: list = []
        for label, value in (
            ("funding_rate", metrics.get("funding_rate")),
            ("open_interest", metrics.get("open_interest")),
            ("long_short_ratio", metrics.get("long_short_ratio")),
        ):
            if value is None:
                unreported.append(f"futures_{label}: 所有可用数据源均未返回该指标")

        return metrics, unreported

    def get_futures_metrics(self, stock_code: str) -> dict:
        """合约公开指标（best-effort；不可用时返回空字段，不报错）。"""
        symbol = (stock_code or "").strip().upper()
        if not is_crypto_symbol(symbol):
            return {}
        metrics, _errors = self._collect_futures_metrics(symbol)
        return metrics

    def get_capital_flow(self, stock_code: str) -> Optional[dict]:
        """现货净主动买入资金流 + 合约资金面。

        净主动买入（USDT）= 2 × 主动买成交额 − 总成交额（taker buy 口径）。
        返回结构对齐项目里的 ``capital_flow`` 块 data.stock_flow 字段。
        """
        symbol = (stock_code or "").strip().upper()
        if not is_crypto_symbol(symbol):
            return None

        result: dict = {
            "status": "partial",
            "stock_flow": {},
            "futures": {},
            "source_chain": [],
            "errors": [],
        }

        started = time.time()
        try:
            data = self._get_json(
                "/api/v3/klines",
                {"symbol": symbol, "interval": "1d", "limit": 15},
            )
            nets: list = []
            if isinstance(data, list):
                for k in data:
                    quote_volume = float(k[7])
                    taker_buy_quote = float(k[10])
                    nets.append(2 * taker_buy_quote - quote_volume)
            if nets:
                result["stock_flow"] = {
                    "main_net_inflow": round(nets[-1], 2),
                    "inflow_5d": round(sum(nets[-5:]), 2),
                    "inflow_10d": round(sum(nets[-10:]), 2),
                }
                result["status"] = "ok"
                result["source_chain"].append(
                    {
                        "provider": "binance_spot_taker",
                        "result": "ok",
                        "duration_ms": int((time.time() - started) * 1000),
                    }
                )
        except Exception as e:  # noqa: BLE001 - surfaced via errors for fail-open
            result["errors"].append(f"spot_taker: {e}")

        futures, futures_errors = self._collect_futures_metrics(symbol)
        result["errors"].extend(futures_errors)
        if futures:
            result["futures"] = futures
            result["source_chain"].append(
                {"provider": "binance_futures", "result": "ok", "duration_ms": 0}
            )

        return result
