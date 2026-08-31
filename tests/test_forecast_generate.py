"""Multi-stock generate tables: one predicted-move column per ticker."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forecast.data import symbol_from_path
from forecast.generate import (
    latest_snapshot,
    parse_symbols,
    predicted_move_wide,
    resolve_data_files,
)


def _bars(symbol: str, bps: list[float], start: str = "2026-08-21 15:50") -> pd.DataFrame:
    n = len(bps)
    when = pd.date_range(start, periods=n, freq="min")
    close = 10.0 + pd.Series(range(n), dtype=float)
    return pd.DataFrame(
        {
            "datetime": when,
            "close": close,
            "traded": True,
            "pred_return_bps": bps,
            "pred_price_1h": close * 1.001,
            "realized_bps": [float("nan")] * n,
        }
    )


def test_parse_symbols():
    assert parse_symbols(None) is None
    assert parse_symbols(" aapl, MSFT,nvda ") == ["AAPL", "MSFT", "NVDA"]


def test_predicted_move_wide_one_column_per_stock():
    by_symbol = {
        "MSFT": _bars("MSFT", [1.0, 2.0]),
        "AAPL": _bars("AAPL", [-0.4, -0.5]),
    }
    wide = predicted_move_wide(by_symbol)
    assert list(wide.columns) == ["AAPL", "MSFT"]
    assert list(wide["AAPL"]) == pytest.approx([-0.4, -0.5])
    assert list(wide["MSFT"]) == pytest.approx([1.0, 2.0])


def test_predicted_move_wide_aligns_missing_timestamps():
    aapl = _bars("AAPL", [1.0], start="2026-08-21 15:59")
    msft = _bars("MSFT", [2.0, 3.0], start="2026-08-21 15:58")
    wide = predicted_move_wide({"AAPL": aapl, "MSFT": msft})
    assert wide.shape == (2, 2)
    assert pd.isna(wide.loc[pd.Timestamp("2026-08-21 15:58"), "AAPL"])
    assert wide.loc[pd.Timestamp("2026-08-21 15:59"), "AAPL"] == pytest.approx(1.0)


def test_latest_snapshot_stocks_are_columns():
    by_symbol = {
        "AAPL": _bars("AAPL", [-0.4, -1.2]),
        "MSFT": _bars("MSFT", [3.0]),
    }
    snap = latest_snapshot(by_symbol)
    assert list(snap.columns) == ["AAPL", "MSFT"]
    assert "pred (bp)" in snap.index
    assert snap.loc["pred (bp)", "AAPL"] == "-1.2"
    assert snap.loc["pred (bp)", "MSFT"] == "+3.0"
    assert snap.loc["last $", "AAPL"].replace(",", "") == "11.0000"


def test_resolve_data_files_filters_symbols(tmp_path: Path):
    a = tmp_path / "AAPL_clean_1min.parquet"
    b = tmp_path / "MSFT_clean_1min.parquet"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    files = resolve_data_files(str(tmp_path), tmp_path, ["MSFT"])
    assert [symbol_from_path(p) for p in files] == ["MSFT"]


def test_resolve_data_files_unknown_symbol(tmp_path: Path):
    (tmp_path / "AAPL_clean_1min.parquet").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="ZZZZ"):
        resolve_data_files(str(tmp_path), tmp_path, ["ZZZZ"])
