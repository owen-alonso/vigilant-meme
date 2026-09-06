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


def test_liquid_universe_is_train_era_locked():
    from forecast.universe import LIQUID_NAMES, BENCHMARK_SYMBOL, download_symbols

    names = download_symbols(include_benchmark=True)
    assert BENCHMARK_SYMBOL == "SPY"
    assert names[0] == "SPY"
    assert "AAPL" in LIQUID_NAMES
    assert 80 <= len(LIQUID_NAMES) <= 150
    assert "SPY" not in LIQUID_NAMES


def test_download_parser_replace_and_universe():
    from forecast.download import build_arg_parser

    args = build_arg_parser().parse_args(
        ["--universe", "liquid", "--source", "yahoo", "--replace", "--interval", "daily"]
    )
    assert args.universe == "liquid"
    assert args.replace is True
    assert args.source == "yahoo"
