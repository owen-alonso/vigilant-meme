"""Alpha Vantage payload parsing and parquet cache merge (no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from forecast.alphavantage import (
    AlphaVantageClient,
    AlphaVantageError,
    bars_to_canonical,
    download_symbol,
    month_labels,
    parse_intraday_payload,
    symbol_parquet_path,
    upsert_bars,
    write_bars,
)


def _payload(n: int = 3) -> dict:
    series = {}
    for i in range(n):
        stamp = f"2026-04-18 09:{30 + i:02d}:00"
        px = 100.0 + i
        series[stamp] = {
            "1. open": str(px),
            "2. high": str(px + 0.1),
            "3. low": str(px - 0.1),
            "4. close": str(px + 0.05),
            "5. volume": str(1000 + i),
        }
    return {
        "Meta Data": {
            "1. Information": "Intraday (1min)",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2026-04-18 09:32:00",
            "4. Interval": "1min",
            "5. Output Size": "Compact",
            "6. Time Zone": "US/Eastern",
        },
        "Time Series (1min)": series,
    }


def test_parse_intraday_payload_canonical():
    bars = parse_intraday_payload(_payload())
    assert list(bars.columns) == [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "interval",
    ]
    assert len(bars) == 3
    assert bars["source"].eq("alphavantage").all()
    assert bars["interval"].eq("1min").all()
    assert str(getattr(bars["datetime"].dtype, "tz", None)) == "America/New_York"
    assert bars["close"].iloc[0] == pytest.approx(100.05)


def test_parse_daily_adjusted_rescales_ohlc():
    payload = {
        "Meta Data": {"2. Symbol": "IBM"},
        "Time Series (Daily)": {
            "2026-04-17": {
                "1. open": "100.0",
                "2. high": "110.0",
                "3. low": "90.0",
                "4. close": "100.0",
                "5. adjusted close": "50.0",
                "6. volume": "1000",
            }
        },
    }
    bars = parse_intraday_payload(payload, interval="daily")
    assert len(bars) == 1
    assert bars["close"].iloc[0] == pytest.approx(50.0)
    assert bars["open"].iloc[0] == pytest.approx(50.0)
    assert bars["high"].iloc[0] == pytest.approx(55.0)
    assert bars["interval"].iloc[0] == "daily"
    with pytest.raises(AlphaVantageError, match="Note"):
        parse_intraday_payload({"Note": "Thank you for using Alpha Vantage"})


def test_parse_rejects_error_message():
    with pytest.raises(AlphaVantageError, match="Error Message"):
        parse_intraday_payload({"Error Message": "Invalid API call"})


def test_month_labels_wraps_year():
    assert month_labels(3, end=date(2026, 1, 15)) == ["2026-01", "2025-12", "2025-11"]


def test_upsert_merges_and_dedupes(tmp_path: Path):
    first = bars_to_canonical(
        pd.DataFrame(
            {
                "datetime": ["2026-04-18 09:30:00", "2026-04-18 09:31:00"],
                "open": [10.0, 10.1],
                "high": [10.1, 10.2],
                "low": [9.9, 10.0],
                "close": [10.05, 10.15],
                "volume": [100, 110],
            }
        ),
    )
    path = tmp_path / "IBM_1min.parquet"
    upsert_bars(path, first)
    second = bars_to_canonical(
        pd.DataFrame(
            {
                "datetime": ["2026-04-18 09:31:00", "2026-04-18 09:32:00"],
                "open": [10.2, 10.3],
                "high": [10.3, 10.4],
                "low": [10.1, 10.2],
                "close": [10.25, 10.35],
                "volume": [999, 120],
            }
        ),
    )
    merged = upsert_bars(path, second)
    assert len(merged) == 3
    row = merged.loc[merged["datetime"] == pd.Timestamp("2026-04-18 09:31:00", tz="America/New_York")]
    assert float(row["close"].iloc[0]) == pytest.approx(10.25)


def test_write_bars_replace_does_not_upsert(tmp_path: Path):
    first = bars_to_canonical(
        pd.DataFrame(
            {
                "datetime": ["2026-04-18 09:30:00"],
                "open": [10.0],
                "high": [10.1],
                "low": [9.9],
                "close": [10.05],
                "volume": [100],
            }
        ),
    )
    path = tmp_path / "AAPL_weekly.parquet"
    write_bars(path, first, replace=True)
    second = bars_to_canonical(
        pd.DataFrame(
            {
                "datetime": ["2026-04-25 09:30:00"],
                "open": [11.0],
                "high": [11.1],
                "low": [10.9],
                "close": [11.05],
                "volume": [120],
            }
        ),
    )
    out = write_bars(path, second, replace=True)
    assert len(out) == 1
    assert float(out["close"].iloc[0]) == pytest.approx(11.05)


def test_download_symbol_uses_client_and_writes_cache(tmp_path: Path):
    payload = json.dumps(_payload()).encode("utf-8")

    def opener(_url: str) -> bytes:
        return payload

    client = AlphaVantageClient(api_key="test-key", min_interval_sec=0, opener=opener)
    bars = download_symbol("aapl", tmp_path, client=client, interval="1min", log_fn=None)
    path = symbol_parquet_path(tmp_path, "AAPL", interval="1min")
    assert path.exists()
    assert len(bars) == 3
    assert path.name == "AAPL_1min.parquet"


def test_fetch_bars_falls_back_from_premium_full():
    premium = json.dumps(
        {
            "Information": (
                "Thank you for using Alpha Vantage! The outputsize=full "
                "parameter value is a premium feature for the TIME_SERIES_DAILY endpoint."
            )
        }
    ).encode("utf-8")
    compact = json.dumps(
        {
            "Meta Data": {"2. Symbol": "IBM"},
            "Time Series (Daily)": {
                "2026-04-17": {
                    "1. open": "10",
                    "2. high": "11",
                    "3. low": "9",
                    "4. close": "10.5",
                    "5. volume": "100",
                }
            },
        }
    ).encode("utf-8")
    seen: list[str] = []

    def opener(url: str) -> bytes:
        seen.append(url)
        if "outputsize=full" in url:
            return premium
        return compact

    client = AlphaVantageClient(api_key="test-key", min_interval_sec=0, opener=opener)
    bars = client.fetch_bars("IBM", interval="daily", outputsize="full")
    assert len(bars) == 1
    assert any("outputsize=full" in u for u in seen)
    assert any("outputsize=compact" in u for u in seen)


def test_parse_weekly_time_series():
    payload = {
        "Meta Data": {"2. Symbol": "AAPL"},
        "Weekly Time Series": {
            "2020-01-10": {
                "1. open": "70",
                "2. high": "75",
                "3. low": "69",
                "4. close": "74",
                "5. volume": "1000",
            },
            "2020-01-17": {
                "1. open": "74",
                "2. high": "80",
                "3. low": "73",
                "4. close": "79",
                "5. volume": "1100",
            },
        },
    }
    bars = parse_intraday_payload(payload, interval="weekly")
    assert len(bars) == 2
    assert bars["interval"].eq("weekly").all()
