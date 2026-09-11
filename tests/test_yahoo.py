"""Yahoo chart payload parsing (no network)."""

from __future__ import annotations

from forecast.yahoo import parse_yahoo_chart


def test_parse_yahoo_chart_canonical():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577923200, 1578009600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [75.0, 76.0],
                                "high": [76.0, 77.0],
                                "low": [74.0, 75.0],
                                "close": [75.5, 76.5],
                                "volume": [1000, 1100],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }
    bars = parse_yahoo_chart(payload, interval="daily")
    assert len(bars) == 2
    assert bars["source"].eq("yahoo").all()
    assert float(bars["close"].iloc[-1]) == 76.5


def test_parse_yahoo_chart_uses_adjclose():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577923200],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0],
                                "high": [110.0],
                                "low": [90.0],
                                "close": [100.0],
                                "volume": [1000],
                            }
                        ],
                        "adjclose": [{"adjclose": [50.0]}],
                    },
                }
            ],
            "error": None,
        }
    }
    bars = parse_yahoo_chart(payload, interval="daily")
    assert float(bars["close"].iloc[0]) == 50.0
    assert float(bars["open"].iloc[0]) == 50.0
    assert float(bars["high"].iloc[0]) == 55.0


def test_liquid_universe_is_train_era_locked():
    from forecast.universe import (
        BENCHMARK_SYMBOL,
        INDUSTRY_ETFS,
        LIQUID_NAMES,
        SECTOR_ETF_BY_SYMBOL,
        allowed_symbols,
        download_symbols,
        is_equity_name,
    )

    names = download_symbols(include_benchmark=True)
    assert BENCHMARK_SYMBOL == "SPY"
    assert names[0] == "SPY"
    assert "AAPL" in LIQUID_NAMES
    assert 80 <= len(LIQUID_NAMES) <= 150
    assert "SPY" not in LIQUID_NAMES
    for etf in INDUSTRY_ETFS:
        assert etf in names
        assert not is_equity_name(etf)
    allowed = allowed_symbols("liquid")
    assert allowed is not None
    assert set(INDUSTRY_ETFS).issubset(allowed)
    equities = [s for s in LIQUID_NAMES if is_equity_name(s)]
    assert 50 <= len(equities) <= 200
    missing = [s for s in equities if s not in SECTOR_ETF_BY_SYMBOL]
    assert missing == []


def test_liquid_wide_is_train_era_locked_and_mapped():
    from forecast.universe import (
        LIQUID_NAMES,
        LIQUID_WIDE_NAMES,
        SECTOR_ETF_BY_SYMBOL,
        download_symbols,
        is_equity_name,
    )

    wide = download_symbols(include_benchmark=True, universe="liquid_wide")
    assert wide[0] == "SPY"
    assert set(LIQUID_NAMES).issubset(LIQUID_WIDE_NAMES)
    equities = [s for s in LIQUID_WIDE_NAMES if is_equity_name(s)]
    assert 150 <= len(equities) <= 200
    assert "UBER" not in LIQUID_WIDE_NAMES
    assert "SNOW" not in LIQUID_WIDE_NAMES
    missing = [s for s in equities if s not in SECTOR_ETF_BY_SYMBOL]
    assert missing == []


def test_download_parser_replace_and_universe():
    from forecast.download import build_arg_parser

    args = build_arg_parser().parse_args(
        ["--universe", "liquid", "--source", "yahoo", "--replace", "--interval", "daily"]
    )
    assert args.universe == "liquid"
    assert args.replace is True
    assert args.source == "yahoo"
    wide = build_arg_parser().parse_args(
        ["--universe", "liquid_wide", "--source", "yahoo"]
    )
    assert wide.universe == "liquid_wide"
    skip = build_arg_parser().parse_args(
        ["--universe", "liquid_wide", "--source", "yahoo", "--skip-existing"]
    )
    assert skip.skip_existing is True
