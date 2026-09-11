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
# Size / small-cap factor used only when size_residual=True. Hedge, not a book name.
SIZE_HEDGE = "IWM"

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
# Optional industry ETFs (pre-2018). Used only when industry_residual=True.
INDUSTRY_ETFS: tuple[str, ...] = ("SMH", "KBE", "XBI", "IYR", "XRT")

# Additional 2018-era large-caps for the 150–200 name book. No post-2018 IPOs.
# Sized so liquid + extras lands ~170–180 equities (plan band), not 200+.
# MET/TROW replace MMC/BK: Yahoo chart 404s on those two tickers in-cloud.
LIQUID_WIDE_EXTRA: tuple[str, ...] = (
    "ORCL", "AMD", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "ADI", "NXPI",
    "ADSK", "APH", "FTNT", "PANW", "MSI", "HPQ",
    "MS", "USB", "PNC", "TFC", "COF", "AIG", "MET", "ICE", "CME", "MCO",
    "AON", "TROW", "STT",
    "LLY", "BDX", "BSX", "EW", "REGN", "ZTS", "HCA", "IDXX", "BIIB",
    "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    "UPS", "FDX", "MMM", "ITW", "ETN", "EMR", "GD", "NOC", "WM", "NSC", "CSX",
    "CMG", "MAR", "YUM", "GM", "F", "DAL", "ORLY", "AZO", "ROST", "TGT",
    "CL", "KMB", "GIS", "STZ", "MNST", "HSY",
    "NEE", "D", "AEP", "SRE", "EXC",
    "APD", "SHW", "ECL", "FCX",
    "AMT", "EQIX", "CCI", "SPG", "O", "DLR", "WELL",
    "TMUS", "CHTR",
)

LIQUID_WIDE_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys((*LIQUID_NAMES, *LIQUID_WIDE_EXTRA, *INDUSTRY_ETFS))
)

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
    "ORCL": "XLK", "AMD": "XLK", "MU": "XLK", "LRCX": "XLK", "KLAC": "XLK",
    "SNPS": "XLK", "CDNS": "XLK", "ADI": "XLK", "NXPI": "XLK", "ADSK": "XLK",
    "APH": "XLK", "TEL": "XLK", "FTNT": "XLK", "PANW": "XLK", "MSI": "XLK",
    "HPQ": "XLK", "ANSS": "XLK", "TMUS": "XLK", "CHTR": "XLK",
    "MS": "XLF", "USB": "XLF", "PNC": "XLF", "TFC": "XLF", "COF": "XLF",
    "MET": "XLF", "AIG": "XLF", "PRU": "XLF", "AFL": "XLF", "ALL": "XLF",
    "TRV": "XLF", "ICE": "XLF", "CME": "XLF", "MCO": "XLF",
    "AON": "XLF", "STT": "XLF", "TROW": "XLF",
    "LLY": "XLV", "BDX": "XLV", "BSX": "XLV", "EW": "XLV", "REGN": "XLV",
    "ZTS": "XLV", "HCA": "XLV", "MCK": "XLV", "IQV": "XLV", "IDXX": "XLV",
    "RMD": "XLV", "BIIB": "XLV", "HUM": "XLV", "CNC": "XLV",
    "COP": "XLE", "SLB": "XLE", "EOG": "XLE", "MPC": "XLE", "PSX": "XLE",
    "VLO": "XLE", "OXY": "XLE", "WMB": "XLE", "HAL": "XLE",
    "UPS": "XLI", "FDX": "XLI", "MMM": "XLI", "ITW": "XLI", "ETN": "XLI",
    "EMR": "XLI", "GD": "XLI", "NOC": "XLI", "WM": "XLI", "CTAS": "XLI",
    "NSC": "XLI", "CSX": "XLI", "PCAR": "XLI", "CMI": "XLI", "FAST": "XLI",
    "CMG": "XLY", "MAR": "XLY", "YUM": "XLY", "GM": "XLY", "F": "XLY",
    "DAL": "XLY", "ORLY": "XLY", "AZO": "XLY", "ROST": "XLY", "DG": "XLY",
    "EBAY": "XLY", "EA": "XLY",
    "TGT": "XLP", "CL": "XLP", "KMB": "XLP", "GIS": "XLP", "KR": "XLP",
    "SYY": "XLP", "ADM": "XLP", "STZ": "XLP", "MNST": "XLP", "HSY": "XLP",
    "NEE": "XLU", "D": "XLU", "AEP": "XLU", "SRE": "XLU", "EXC": "XLU",
    "XEL": "XLU", "PEG": "XLU", "ED": "XLU",
    "APD": "XLB", "SHW": "XLB", "ECL": "XLB", "NEM": "XLB", "FCX": "XLB",
    "NUE": "XLB",
    "EQIX": "XLRE", "CCI": "XLRE", "SPG": "XLRE", "O": "XLRE",
    "WELL": "XLRE", "PSA": "XLRE", "DLR": "XLRE",
}

INDUSTRY_ETF_BY_SYMBOL: dict[str, str] = {
    "NVDA": "SMH", "AVGO": "SMH", "TXN": "SMH", "QCOM": "SMH", "AMAT": "SMH",
    "INTC": "SMH", "AMD": "SMH", "MU": "SMH", "LRCX": "SMH", "KLAC": "SMH",
    "ADI": "SMH", "NXPI": "SMH",
    "JPM": "KBE", "BAC": "KBE", "WFC": "KBE", "C": "KBE", "USB": "KBE",
    "PNC": "KBE", "TFC": "KBE", "COF": "KBE", "BK": "KBE", "STT": "KBE",
    "FITB": "KBE",
    "GILD": "XBI", "VRTX": "XBI", "REGN": "XBI", "BIIB": "XBI", "AMGN": "XBI",
    "PLD": "IYR", "AMT": "IYR", "EQIX": "IYR", "CCI": "IYR", "SPG": "IYR",
    "O": "IYR", "WELL": "IYR", "PSA": "IYR", "DLR": "IYR",
    "HD": "XRT", "LOW": "XRT", "TJX": "XRT", "ORLY": "XRT", "AZO": "XRT",
    "ROST": "XRT", "DG": "XRT", "TGT": "XRT", "EBAY": "XRT", "BBY": "XRT",
}

_NON_EQUITY: frozenset[str] = frozenset(
    (BENCHMARK_SYMBOL,) + INDEX_ETFS + SECTOR_ETFS + MACRO_HEDGES + INDUSTRY_ETFS
)


def is_equity_name(symbol: str) -> bool:
    """True for single-name equities (not SPY / index / sector / macro ETFs)."""
    return str(symbol).upper() not in _NON_EQUITY


def universe_name_list(universe: str) -> tuple[str, ...]:
    raw = (universe or "").strip().lower()
    if raw in ("", "liquid"):
        return LIQUID_NAMES
    if raw in ("liquid_wide", "wide"):
        return LIQUID_WIDE_NAMES
    raise ValueError(
        f"unknown universe {universe!r}; expected '', 'liquid', or 'liquid_wide'"
    )


def hedge_symbol_for(symbol: str, *, sector_residual: bool, benchmark: str = "SPY") -> str:
    """Causal hedge ticker for residual labels. SPY if sector map or ETF is unused."""
    bench = str(benchmark or "SPY").upper()
    if not sector_residual:
        return bench
    mapped = SECTOR_ETF_BY_SYMBOL.get(str(symbol).upper())
    return mapped if mapped else bench


def industry_symbol_for(symbol: str) -> str | None:
    """Optional finer hedge. None if the name has no industry ETF map."""
    return INDUSTRY_ETF_BY_SYMBOL.get(str(symbol).upper())


def trading_symbols(universe: str = "liquid") -> list[str]:
    return list(universe_name_list(universe))


def download_symbols(*, include_benchmark: bool = True, universe: str = "liquid") -> list[str]:
    names = list(universe_name_list(universe or "liquid"))
    for extra in INDUSTRY_ETFS:
        if extra not in names:
            names.append(extra)
    if include_benchmark and BENCHMARK_SYMBOL not in names:
        names = [BENCHMARK_SYMBOL, *names]
    elif include_benchmark and names and names[0] != BENCHMARK_SYMBOL:
        names = [BENCHMARK_SYMBOL, *[s for s in names if s != BENCHMARK_SYMBOL]]
    return names


def allowed_symbols(
    universe: str,
    *,
    include_benchmark: bool = True,
) -> frozenset[str] | None:
    """Membership set for a named universe, or None to keep every parquet.

    Industry ETFs are allowed as residual hedges; ``equities_only`` keeps
    them out of the trading book.
    """
    raw = (universe or "").strip().lower()
    if not raw:
        return None
    names = set(universe_name_list(raw))
    names.update(INDUSTRY_ETFS)
    if include_benchmark:
        names.add(BENCHMARK_SYMBOL)
    return frozenset(names)
