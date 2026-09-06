"""Long daily/weekly/monthly history from Yahoo Finance chart API (no key)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

import pandas as pd

from forecast.alphavantage import AlphaVantageError, bars_to_canonical

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
INTERVAL_CODE = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}


def parse_yahoo_chart(payload: dict[str, Any], *, interval: str = "daily") -> pd.DataFrame:
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        raise AlphaVantageError("Yahoo chart payload was not an object")
    error = chart.get("error")
    if error:
        raise AlphaVantageError(f"Yahoo chart error: {error}")
    results = chart.get("result")
    if not results:
        raise AlphaVantageError("Yahoo chart returned no result")
    row = results[0]
    stamps = row.get("timestamp") or []
    quote = ((row.get("indicators") or {}).get("quote") or [{}])[0]
    if not stamps:
        raise AlphaVantageError("Yahoo chart had no timestamps")
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(stamps, unit="s", utc=True),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    adj = ((row.get("indicators") or {}).get("adjclose") or [{}])
    if adj and adj[0].get("adjclose") is not None:
        df["adjustedclose"] = adj[0]["adjclose"]
    return bars_to_canonical(df, interval=interval, source="yahoo")


def fetch_yahoo_bars(
    symbol: str,
    *,
    interval: str = "daily",
    opener: Callable[[str], Any] | None = None,
    timeout_sec: float = 30.0,
) -> pd.DataFrame:
    if interval not in INTERVAL_CODE:
        raise AlphaVantageError(f"Yahoo source supports daily/weekly/monthly, not {interval!r}")
    query = urllib.parse.urlencode(
        {
            "period1": "0",
            "period2": "9999999999",
            "interval": INTERVAL_CODE[interval],
            "events": "history",
        }
    )
    url = f"{YAHOO_CHART.format(symbol=urllib.parse.quote(str(symbol).upper()))}?{query}"
    try:
        if opener is not None:
            raw = opener(url)
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        else:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                text = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise AlphaVantageError(f"Yahoo request failed: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AlphaVantageError("Yahoo returned non-JSON") from exc
    if not isinstance(payload, dict):
        raise AlphaVantageError("Yahoo JSON was not an object")
    return parse_yahoo_chart(payload, interval=interval)
