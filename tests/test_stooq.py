"""Stooq CSV parsing (no network)."""

from __future__ import annotations

from forecast.stooq import parse_stooq_csv, stooq_ticker


def test_stooq_ticker_adds_us_suffix():
    assert stooq_ticker("AAPL") == "aapl.us"
    assert stooq_ticker("vod.uk") == "vod.uk"


def test_parse_stooq_csv_canonical():
    text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2020-01-02,75.0,76.0,74.0,75.5,1000\n"
        "2020-01-03,75.5,77.0,75.0,76.0,1100\n"
    )
    bars = parse_stooq_csv(text, interval="daily")
    assert len(bars) == 2
    assert bars["source"].eq("stooq").all()
    assert float(bars["close"].iloc[-1]) == 76.0
