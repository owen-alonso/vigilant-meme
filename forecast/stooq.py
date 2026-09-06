"""Long daily/weekly/monthly history from Stooq (no API key).

Stooq CSV is the free way to get more than Alpha Vantage's 100 daily bars.
US tickers are requested as ``aapl.us``.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from typing import Any, Callable

import pandas as pd

from forecast.alphavantage import AlphaVantageError, bars_to_canonical

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i={code}"
INTERVAL_CODE = {"daily": "d", "weekly": "w", "monthly": "m"}


def stooq_ticker(symbol: str) -> str:
    raw = str(symbol).strip().lower()
    if "." not in raw:
        return f"{raw}.us"
    return raw


def parse_stooq_csv(text: str, *, interval: str = "daily") -> pd.DataFrame:
    body = (text or "").strip()
    if not body or body.lower().startswith("<!doctype") or body.lower().startswith("<html"):
        raise AlphaVantageError("Stooq returned HTML instead of CSV")
    if body.lower().startswith("no data") or "exceeded" in body.lower():
        raise AlphaVantageError(f"Stooq: {body[:200]}")
    df = pd.read_csv(io.StringIO(body))
    if df.empty:
        raise AlphaVantageError("Stooq CSV had no rows")
    renamed = {str(c): str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=renamed)
    if "date" not in df.columns:
        raise AlphaVantageError(f"Stooq CSV missing Date column: {list(df.columns)}")
    df = df.rename(columns={"date": "datetime"})
    return bars_to_canonical(df, interval=interval, source="stooq")


def fetch_stooq_bars(
    symbol: str,
    *,
    interval: str = "daily",
    opener: Callable[[str], Any] | None = None,
    timeout_sec: float = 30.0,
) -> pd.DataFrame:
    if interval not in INTERVAL_CODE:
        raise AlphaVantageError(
            f"Stooq supports daily/weekly/monthly, not {interval!r}"
        )
    ticker = stooq_ticker(symbol)
    url = STOOQ_URL.format(symbol=ticker, code=INTERVAL_CODE[interval])
    try:
        if opener is not None:
            raw = opener(url)
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        else:
            request = urllib.request.Request(
                url, headers={"User-Agent": "vigilant-meme/forecast"}
            )
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                text = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise AlphaVantageError(f"Stooq request failed: {exc}") from exc
    return parse_stooq_csv(text, interval=interval)
