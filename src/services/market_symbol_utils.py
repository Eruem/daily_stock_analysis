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


# --- Binance bStocks (tokenized US equities / ETFs) ------------------------
# bStocks trade on Binance Spot as ``<TICKER>B`` + quote asset, e.g.
# ``AAPLBUSDT``. A plain "ends with B" rule is not enough because crypto bases
# also end with B (ARB, BNB, SHIB, TRB, ...), so a curated ticker allowlist is
# used to keep classification deterministic and false-positive free.

_BSTOCK_TICKERS: frozenset[str] = frozenset({
    "AAOI", "AAPL", "ALAB", "AMAT", "AMD", "AMZN", "ARM", "ASML", "ASTS",
    "AVGO", "AXTI", "BABA", "BMNR", "CBR", "COHR", "COIN", "CRCL", "CRDO",
    "CRM", "CRWD", "CRWV", "DELL", "DJT", "DRAM", "EWY", "FLNC", "GLW",
    "GME", "GOOGL", "GPRO", "GS", "HIMS", "HOOD", "IBM", "INTC", "IREN",
    "KORU", "LITE", "META", "MRNA", "MRVL", "MSFT", "MSTR", "MU", "NBIS",
    "NFLX", "NOK", "NVDA", "ORCL", "PLTR", "PYPL", "QCOM", "QQQ", "RDDT",
    "RKLB", "SKHY", "SMCI", "SMH", "SNDK", "SOXL", "SOXS", "SPCX", "SPY",
    "SQQQ", "STX", "TQQQ", "TSM", "TSLA", "USAR", "WDC",
})

_BSTOCK_NAMES: dict[str, str] = {
    "AAPL": "苹果", "NVDA": "英伟达", "TSLA": "特斯拉", "MSFT": "微软",
    "AMZN": "亚马逊", "GOOGL": "谷歌", "META": "Meta", "AMD": "AMD",
    "INTC": "英特尔", "MU": "美光", "TSM": "台积电", "NFLX": "奈飞",
    "PLTR": "Palantir", "COIN": "Coinbase", "MSTR": "Strategy(MicroStrategy)",
    "HOOD": "Robinhood", "ORCL": "甲骨文", "IBM": "IBM", "PYPL": "PayPal",
    "QCOM": "高通", "SMCI": "超微电脑", "DELL": "戴尔", "AVGO": "博通",
    "ASML": "阿斯麦", "ARM": "Arm", "BABA": "阿里巴巴", "GS": "高盛",
    "GME": "游戏驿站", "CRCL": "Circle", "RDDT": "Reddit", "RKLB": "Rocket Lab",
    "SPY": "标普500ETF", "QQQ": "纳斯达克100ETF", "TQQQ": "纳指3倍做多ETF",
    "SQQQ": "纳指3倍做空ETF", "SOXL": "半导体3倍做多ETF", "SOXS": "半导体3倍做空ETF",
    "SMH": "半导体ETF", "GLW": "康宁", "MRNA": "Moderna", "ASTS": "AST SpaceMobile",
    "CRWV": "CoreWeave",
}


def split_bstock_symbol(stock_code: str) -> Optional[tuple[str, str]]:
    """Return ``(underlying_ticker, quote)`` for a Binance bStock, else ``None``."""

    parts = split_crypto_symbol(stock_code)
    if parts is None:
        return None
    base, quote = parts
    if base.endswith("B") and base[:-1] in _BSTOCK_TICKERS:
        return base[:-1], quote
    return None


def is_bstock_symbol(stock_code: str) -> bool:
    """Return whether a code is a Binance bStock (tokenized stock/ETF)."""

    return split_bstock_symbol(stock_code) is not None


def bstock_display_name(stock_code: str) -> str:
    """Return a display name for a bStock, e.g. ``苹果(AAPL)代币化股票``."""

    parts = split_bstock_symbol(stock_code)
    if parts is None:
        return (stock_code or "").strip().upper()
    ticker, _quote = parts
    name = _BSTOCK_NAMES.get(ticker)
    return f"{name}({ticker})代币化股票" if name else f"{ticker}代币化股票"


def binance_display_name(stock_code: str) -> str:
    """Unified display name for Binance symbols (bStocks or crypto pairs)."""

    if is_bstock_symbol(stock_code):
        return bstock_display_name(stock_code)
    return crypto_display_name(stock_code)


def crypto_query_terms(stock_code: str) -> str:
    """Return search-engine friendly terms for a Binance symbol.

    Crypto:    ``BTCUSDT``   -> ``比特币 BTC``
    bStock:    ``AAPLBUSDT`` -> ``苹果 AAPL``
    """

    bstock = split_bstock_symbol(stock_code)
    if bstock is not None:
        ticker, _quote = bstock
        name = _BSTOCK_NAMES.get(ticker)
        return f"{name} {ticker}" if name else ticker

    parts = split_crypto_symbol(stock_code)
    if parts is not None:
        base, _quote = parts
        name = _CRYPTO_BASE_NAMES.get(base)
        if name:
            cn = name.split("(")[0]
            return f"{cn} {base}"
        return base

    return (stock_code or "").strip().upper()
