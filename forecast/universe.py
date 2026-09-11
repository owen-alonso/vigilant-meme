"""Train-era-locked liquid US names for the cross-sectional book.

Membership is a 2018-era large-cap + sector-ETF list, not today's index.
SPY is the market benchmark and is downloaded separately; it is not a
trading name here.
"""

from __future__ import annotations

# Mega-caps and other liquid names that existed (under these or predecessor
# tickers Yahoo still serves) through the 2018 train cut.
LIQUID_NAMES: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "AVGO",
    "TSLA",
    "JPM",
    "JNJ",
    "UNH",
    "XOM",
    "V",
    "MA",
    "PG",
    "HD",
    "CVX",
    "MRK",
    "ABBV",
    "PEP",
    "KO",
    "COST",
    "WMT",
    "BAC",
    "CRM",
    "CSCO",
    "ACN",
    "LIN",
    "MCD",
    "ABT",
    "TMO",
    "DHR",
    "ADBE",
    "WFC",
    "DIS",
    "VZ",
    "CMCSA",
    "NFLX",
    "TXN",
    "NKE",
    "PM",
    "RTX",
    "HON",
    "UNP",
    "IBM",
    "GE",
    "CAT",
    "AMGN",
    "QCOM",
    "LOW",
    "INTU",
    "AMAT",
    "NOW",
    "ISRG",
    "BKNG",
    "GS",
    "SPGI",
    "BLK",
    "AXP",
    "SYK",
    "TJX",
    "MDLZ",
    "GILD",
    "LMT",
    "DE",
    "ADP",
    "VRTX",
    "C",
    "PGR",
    "PLD",
    "ELV",
    "CB",
    "CI",
    "SO",
    "DUK",
    "MO",
    "BMY",
    "PFE",
    "T",
    "INTC",
    "BA",
    "SBUX",
    "MDT",
    "CVS",
    "SCHW",
    # Liquid sector / style ETFs that existed before 2018.
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "EEM",
    "EFA",
    "GLD",
    "TLT",
    "HYG",
    "LQD",
)

BENCHMARK_SYMBOL = "SPY"


def trading_symbols() -> list[str]:
    return list(LIQUID_NAMES)


def download_symbols(*, include_benchmark: bool = True) -> list[str]:
    names = list(LIQUID_NAMES)
    if include_benchmark and BENCHMARK_SYMBOL not in names:
        names = [BENCHMARK_SYMBOL, *names]
    return names


def allowed_symbols(
    universe: str,
    *,
    include_benchmark: bool = True,
) -> frozenset[str] | None:
    """Membership set for a named universe, or None to keep every parquet."""
    raw = (universe or "").strip().lower()
    if not raw:
        return None
    if raw != "liquid":
        raise ValueError(
            f"unknown universe {universe!r}; expected '' or 'liquid'"
        )
    names = set(LIQUID_NAMES)
    if include_benchmark:
        names.add(BENCHMARK_SYMBOL)
    return frozenset(names)
