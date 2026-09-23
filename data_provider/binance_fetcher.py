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
# 合约公开数据（资金费率/持仓量/多空比）：与现货不同域，且在部分地域可能被限制，
# 因此全部调用都做 best-effort 降级处理。
_BINANCE_FAPI_BASE_URL = "https://fapi.binance.com"
_KLINE_LIMIT = 1000
_DAY_MS = 86_400_000


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
        self._fapi_url = (
            (os.getenv("BINANCE_FAPI_BASE_URL") or _BINANCE_FAPI_BASE_URL).strip().rstrip("/")
        )
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
        """合约公开数据（best-effort，失败由调用方降级）。"""
        resp = self._session.get(f"{self._fapi_url}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

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
    def _collect_futures_metrics(self, symbol: str) -> tuple:
        """采集合约公开指标（best-effort）。

        Returns:
            (metrics, errors)
            - metrics: 成功获取的字段
            - errors: 不可用原因，向上游汇报，避免“只有 None 没有理由”
        """
        metrics: dict = {}
        errors: list = []

        def _attempt(label: str, path: str, params: dict, extract) -> None:
            try:
                data = self._get_fapi_json(path, params)
            except Exception as e:  # noqa: BLE001 - optional metric
                errors.append(f"futures_{label}: {e}")
                logger.info("[BinanceFetcher] %s %s 不可用: %s", symbol, label, e)
                return
            try:
                value = extract(data)
            except (TypeError, ValueError, IndexError, KeyError) as e:
                errors.append(f"futures_{label}: parse {e}")
                return
            if value is not None:
                metrics[label] = value

        _attempt(
            "funding_rate",
            "/fapi/v1/premiumIndex",
            {"symbol": symbol},
            lambda d: (
                float(d["lastFundingRate"])
                if isinstance(d, dict) and d.get("lastFundingRate") is not None
                else None
            ),
        )
        _attempt(
            "open_interest",
            "/fapi/v1/openInterest",
            {"symbol": symbol},
            lambda d: (
                float(d["openInterest"])
                if isinstance(d, dict) and d.get("openInterest") is not None
                else None
            ),
        )
        _attempt(
            "long_short_ratio",
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": "1d", "limit": 1},
            lambda d: (
                float(d[0]["longShortRatio"])
                if isinstance(d, list) and d and d[0].get("longShortRatio") is not None
                else None
            ),
        )
        return metrics, errors

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
