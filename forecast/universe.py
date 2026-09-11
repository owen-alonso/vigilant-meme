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

INDEX_ETFS: tuple[str, ...] = ("QQQ", "IWM", "DIA")
SECTOR_ETFS: tuple[str, ...] = (
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
)
MACRO_HEDGES: tuple[str, ...] = ("GLD", "TLT", "HYG", "LQD", "EEM", "EFA")

# Approximate GICS-ish map used only as a feature / residual hedge.
# Unknown equities and missing sector ETFs fall back to SPY. XLC names use XLK.
SECTOR_ETF_BY_SYMBOL: dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "CRM": "XLK",
    "CSCO": "XLK", "ACN": "XLK", "ADBE": "XLK", "TXN": "XLK", "IBM": "XLK",
    "QCOM": "XLK", "INTU": "XLK", "AMAT": "XLK", "NOW": "XLK", "INTC": "XLK",
    "GOOGL": "XLK", "GOOG": "XLK", "META": "XLK", "NFLX": "XLK", "DIS": "XLK",
    "CMCSA": "XLK", "T": "XLK", "VZ": "XLK",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "BLK": "XLF", "SCHW": "XLF", "AXP": "XLF", "V": "XLF",
    "MA": "XLF", "SPGI": "XLF", "PGR": "XLF", "CB": "XLF",
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "PFE": "XLV", "TMO": "XLV", "ABT": "XLV", "DHR": "XLV", "AMGN": "XLV",
    "ISRG": "XLV", "SYK": "XLV", "MDT": "XLV", "GILD": "XLV", "VRTX": "XLV",
    "ELV": "XLV", "CI": "XLV", "BMY": "XLV", "CVS": "XLV",
    "XOM": "XLE", "CVX": "XLE",
    "CAT": "XLI", "GE": "XLI", "HON": "XLI", "UNP": "XLI", "DE": "XLI",
    "BA": "XLI", "RTX": "XLI", "LMT": "XLI", "ADP": "XLI",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "SBUX": "XLY", "LOW": "XLY", "TJX": "XLY", "BKNG": "XLY",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    "PM": "XLP", "MO": "XLP", "MDLZ": "XLP",
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU",
    "LIN": "XLB",
    "AMT": "XLRE", "PLD": "XLRE",
}

_NON_EQUITY: frozenset[str] = frozenset(
    (BENCHMARK_SYMBOL,) + INDEX_ETFS + SECTOR_ETFS + MACRO_HEDGES
)


def is_equity_name(symbol: str) -> bool:
    """True for single-name equities (not SPY / index / sector / macro ETFs)."""
    return str(symbol).upper() not in _NON_EQUITY


def hedge_symbol_for(symbol: str, *, sector_residual: bool, benchmark: str = "SPY") -> str:
    """Causal hedge ticker for residual labels. SPY if sector map or ETF is unused."""
    bench = str(benchmark or "SPY").upper()
    if not sector_residual:
        return bench
    mapped = SECTOR_ETF_BY_SYMBOL.get(str(symbol).upper())
    return mapped if mapped else bench


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
