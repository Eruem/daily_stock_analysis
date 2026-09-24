# -*- coding: utf-8 -*-
"""免费无 key 的加密市场数据源（Fear & Greed + DefiLlama）。

三个数据全部免费、无需 API Key、无注册配额：

- Alternative.me Fear & Greed Index（市场恐慌/贪婪情绪，每日更新）
    GET https://api.alternative.me/fng/?limit=N
- DefiLlama 全链 TVL（DeFi 资金水位，近实时）
    GET https://api.llama.fi/v2/chains
- DefiLlama 稳定币总市值（场外弹药 / 潜在买盘）
    GET https://stablecoins.llama.fi/stablecoins?includePrices=true

所有函数 fail-open：异常返回 None，由调用方决定 block 状态。
结果做短 TTL 进程内缓存——同一批分析里 N 个币共享同一份市场级数据，
避免每个标的一轮重复请求。
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/"
_DEFILLAMA_CHAINS_URL = "https://api.llama.fi/v2/chains"
_DEFILLAMA_STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoins"
_REQUEST_TIMEOUT = 15
_CACHE_TTL_SECONDS = 600  # 10 分钟：FNG 每日更新，TVL/稳定币 10 分钟足够新鲜

_cache_lock = threading.Lock()
_cache: Dict[str, tuple] = {}  # key -> (expires_at, value)


def _cache_get(key: str):
    with _cache_lock:
        item = _cache.get(key)
        if item and item[0] > time.time():
            return item[1]
    return None


def _cache_put(key: str, value) -> None:
    with _cache_lock:
        _cache[key] = (time.time() + _CACHE_TTL_SECONDS, value)


def _http_get_json(url: str, params: Optional[dict] = None) -> Any:
    resp = requests.get(
        url,
        params=params,
        timeout=_REQUEST_TIMEOUT,
        headers={"User-Agent": "daily_stock_analysis/1.0"},
    )
    resp.raise_for_status()
    return resp.json()


def fetch_fear_greed(days: int = 7) -> Optional[Dict[str, Any]]:
    """Alternative.me Fear & Greed 指数。

    Returns:
        {"current": {"value": int, "classification": str, "date": str},
         "series": [{"value": int, "date": str}, ...],
         "avg_7d": Optional[float]}
        失败返回 None（fail-open）。
    """
    days = max(1, min(int(days), 30))
    cache_key = f"fng:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        data = _http_get_json(_FNG_URL, {"limit": days})
        # data 结构: {"name": ..., "data": [ {value, value_classification, timestamp, ...}, ...]}
        raw_items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(raw_items, list) or not raw_items:
            logger.info("[free_crypto] Fear&Greed 返回空数据")
            return None

        series: List[Dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                value = int(item.get("value"))
            except (TypeError, ValueError):
                continue
            ts = item.get("timestamp")
            date_str = ""
            if ts is not None:
                try:
                    from datetime import datetime, timezone

                    date_str = datetime.fromtimestamp(
                        int(ts), tz=timezone.utc
                    ).strftime("%Y-%m-%d")
                except (TypeError, ValueError, OSError):
                    date_str = str(ts)
            series.append({
                "value": value,
                "classification": str(item.get("value_classification") or ""),
                "date": date_str,
            })

        if not series:
            return None

        current = series[0]
        avg_7d = round(sum(x["value"] for x in series[:7]) / min(len(series), 7), 1)
        result = {
            "current": current,
            "series": series,
            "avg_7d": avg_7d,
        }
        _cache_put(cache_key, result)
        return result
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.info("[free_crypto] Fear&Greed 获取失败: %s", e)
        return None


def fetch_defillama_snapshot() -> Optional[Dict[str, Any]]:
    """DefiLlama 全链 TVL + 稳定币总市值（两次轻量请求合并为一个快照）。

    Returns:
        {"total_tvl_usd": float, "stablecoin_mktcap_usd": float,
         "top_chains": [{"name": str, "tvl_usd": float}, ...],
         "top_stablecoins": [{"name": str, "mktcap_usd": float}, ...]}
        失败返回 None（fail-open）。
    """
    cache_key = "defillama"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        chains = _http_get_json(_DEFILLAMA_CHAINS_URL)
        total_tvl = 0.0
        top_chains: List[Dict[str, Any]] = []
        if isinstance(chains, list):
            for c in chains:
                if not isinstance(c, dict):
                    continue
                try:
                    tvl = float(c.get("tvl") or 0)
                except (TypeError, ValueError):
                    continue
                total_tvl += tvl
                top_chains.append({"name": str(c.get("name") or ""), "tvl_usd": tvl})
            top_chains.sort(key=lambda x: x["tvl_usd"], reverse=True)

        stable = _http_get_json(_DEFILLAMA_STABLECOINS_URL, {"includePrices": "true"})
        total_stable = 0.0
        top_stables: List[Dict[str, Any]] = []
        pegged = stable.get("peggedAssets") if isinstance(stable, dict) else None
        if isinstance(pegged, list):
            for s in pegged:
                if not isinstance(s, dict):
                    continue
                circ = s.get("circulating")
                if not isinstance(circ, dict):
                    continue
                try:
                    mktcap = float(circ.get("peggedUSD") or 0)
                except (TypeError, ValueError):
                    continue
                total_stable += mktcap
                top_stables.append({"name": str(s.get("name") or ""), "mktcap_usd": mktcap})
            top_stables.sort(key=lambda x: x["mktcap_usd"], reverse=True)

        if total_tvl <= 0 and total_stable <= 0:
            logger.info("[free_crypto] DefiLlama 返回空数据")
            return None

        result = {
            "total_tvl_usd": round(total_tvl, 0),
            "stablecoin_mktcap_usd": round(total_stable, 0),
            "top_chains": top_chains[:5],
            "top_stablecoins": top_stables[:5],
        }
        _cache_put(cache_key, result)
        return result
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.info("[free_crypto] DefiLlama 获取失败: %s", e)
        return None
