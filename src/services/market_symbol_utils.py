# -*- coding: utf-8 -*-
"""Shared market-symbol helpers for suffix-only offshore markets.

Keep this module dependency-light so it can be used by data providers, market
context, trading calendars, stock-index loading, and API input normalization
without introducing import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SuffixMarketSpec:
    """A suffix-only Yahoo Finance market rule."""

    market: str
    suffixes: tuple[str, ...]
    digit_lengths: tuple[int, ...]


_SUFFIX_MARKET_SPECS: tuple[SuffixMarketSpec, ...] = (
    SuffixMarketSpec("jp", ("T",), (4, 5)),
    SuffixMarketSpec("kr", ("KS", "KQ"), (6,)),
    # Taiwan support mirrors the same suffix-only pattern; keep it here so the
    # shared helpers stay complete for all yfinance-only offshore markets.
    SuffixMarketSpec("tw", ("TW", "TWO"), (4, 5, 6)),
)

_MARKET_TO_SPEC = {spec.market: spec for spec in _SUFFIX_MARKET_SPECS}
_SUFFIX_TO_SPEC = {
    suffix: spec
    for spec in _SUFFIX_MARKET_SPECS
    for suffix in spec.suffixes
}


def split_suffix_symbol(stock_code: str) -> tuple[str, str] | None:
    """Return ``(base, suffix)`` for dotted symbols, upper-cased and stripped."""

    code = (stock_code or "").strip().upper()
    if "." not in code:
        return None
    base, suffix = code.rsplit(".", 1)
    if not base or not suffix:
        return None
    return base, suffix


def get_suffix_market(stock_code: str) -> Optional[str]:
    """Return jp/kr/tw for supported suffix-only Yahoo symbols, else None."""

    parts = split_suffix_symbol(stock_code)
    if parts is None:
        return None
    base, suffix = parts
    spec = _SUFFIX_TO_SPEC.get(suffix)
    if spec is None:
        return None
    if not (base.isdigit() and len(base) in spec.digit_lengths):
        return None
    return spec.market


def is_suffix_market_symbol(stock_code: str, market: Optional[str] = None) -> bool:
    """Return whether a stock code is a supported suffix-only Yahoo symbol."""

    detected = get_suffix_market(stock_code)
    if market is None:
        return detected is not None
    return detected == (market or "").strip().lower()


def is_jp_suffix_symbol(stock_code: str) -> bool:
    return is_suffix_market_symbol(stock_code, "jp")


def is_kr_suffix_symbol(stock_code: str) -> bool:
    return is_suffix_market_symbol(stock_code, "kr")


def is_tw_suffix_symbol(stock_code: str) -> bool:
    return is_suffix_market_symbol(stock_code, "tw")


def normalize_suffix_market_symbol(stock_code: str) -> Optional[str]:
    """Normalize supported suffix-only symbols to upper-case Yahoo form."""

    parts = split_suffix_symbol(stock_code)
    if parts is None:
        return None
    base, suffix = parts
    if get_suffix_market(f"{base}.{suffix}") is None:
        return None
    return f"{base}.{suffix}"


def suffix_base_lookup_allowed(canonical_code: str) -> bool:
    """Return True when a suffix-market code may be resolved from its bare base.

    JP/KR intentionally allow stock-index-backed bare-code lookup to support the
    existing MVP behavior. TW remains strict suffix-only for now because its
    follow-up index work is not part of this issue.
    """

    return get_suffix_market(canonical_code) in {"jp", "kr"}


def market_suffixes(market: str) -> tuple[str, ...]:
    spec = _MARKET_TO_SPEC.get((market or "").strip().lower())
    return spec.suffixes if spec else ()


# --- Cryptocurrency (Binance spot) symbols ---------------------------------
# Kept here so data providers, market context, trading calendars and API input
# normalization share one dependency-light definition of a crypto pair.

_CRYPTO_QUOTE_ASSETS: tuple[str, ...] = (
    "USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI",
    "BTC", "ETH", "BNB", "EUR", "TRY", "BRL", "GBP", "AUD",
)

_CRYPTO_BASE_NAMES: dict[str, str] = {
    "BTC": "比特币(BTC)",
    "ETH": "以太坊(ETH)",
    "BNB": "币安币(BNB)",
    "SOL": "Solana(SOL)",
    "XRP": "瑞波币(XRP)",
    "ADA": "艾达币(ADA)",
    "DOGE": "狗狗币(DOGE)",
    "TON": "Toncoin(TON)",
    "AVAX": "Avalanche(AVAX)",
    "DOT": "波卡(DOT)",
    "LINK": "Chainlink(LINK)",
    "MATIC": "Polygon(MATIC)",
    "POL": "Polygon(POL)",
    "LTC": "莱特币(LTC)",
    "BCH": "比特币现金(BCH)",
    "TRX": "波场(TRX)",
    "SHIB": "柴犬币(SHIB)",
    "PEPE": "Pepe(PEPE)",
    "SUI": "Sui(SUI)",
    "APT": "Aptos(APT)",
    "ARB": "Arbitrum(ARB)",
    "OP": "Optimism(OP)",
}


def split_crypto_symbol(stock_code: str) -> Optional[tuple[str, str]]:
    """Return ``(base, quote)`` for a supported crypto pair, else ``None``.

    Recognizes Binance spot-style concatenated pairs such as ``BTCUSDT`` /
    ``ETHBTC``. Requires an alphanumeric symbol of at least 6 chars ending with
    a known quote asset, which avoids colliding with A-share / HK / US tickers.
    """

    code = (stock_code or "").strip().upper()
    if len(code) < 6 or not code.isalnum() or code.isdigit():
        return None
    for quote in _CRYPTO_QUOTE_ASSETS:
        if code.endswith(quote) and len(code) > len(quote):
            base = code[: -len(quote)]
            if base.isalnum():
                return base, quote
    return None


def is_crypto_symbol(stock_code: str) -> bool:
    """Return whether a code looks like a supported crypto trading pair."""

    return split_crypto_symbol(stock_code) is not None


def crypto_display_name(stock_code: str) -> str:
    """Return a human-friendly display name for a crypto pair."""

    parts = split_crypto_symbol(stock_code)
    if parts is None:
        return (stock_code or "").strip().upper()
    base, quote = parts
    name = _CRYPTO_BASE_NAMES.get(base)
    return f"{name}/{quote}" if name else f"{base}/{quote}"
