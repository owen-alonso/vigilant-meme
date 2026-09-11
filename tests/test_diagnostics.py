"""Mix-jump diagnostics for poisoned weekly caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.config import DataConfig
from forecast.diagnostics import (
    assert_calendar_price_quality,
    close_return_diagnostics,
    looks_like_mixed_scale,
)


def test_mixed_scale_weekly_fails_autocorr_gate(tmp_path: Path):
    n = 80
    close = np.where(np.arange(n) % 2 == 0, 100.0, 160.0).astype(np.float64)
    dates = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    path = tmp_path / "AAPL_weekly.parquet"
    pd.DataFrame(
        {
            "datetime": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1.0),
        }
    ).to_parquet(path)
    stats = close_return_diagnostics(close)
    assert stats["lag1_autocorr"] < -0.2
    assert looks_like_mixed_scale(stats)
    cfg = DataConfig(interval="weekly", allow_mixed_prices=False)
    with pytest.raises(ValueError, match="mixed"):
        assert_calendar_price_quality(path, cfg)


def test_clean_weekly_passes_autocorr_gate(tmp_path: Path):
    n = 80
    close = 100.0 * (1.002 ** np.arange(n))
    dates = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    path = tmp_path / "AAPL_weekly.parquet"
    pd.DataFrame(
        {
            "datetime": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1.0),
        }
    ).to_parquet(path)
    cfg = DataConfig(interval="weekly", allow_mixed_prices=False)
    stats = assert_calendar_price_quality(path, cfg)
    assert not looks_like_mixed_scale(stats)


def test_delete_mixed_weekly_removes_file(tmp_path: Path):
    n = 80
    close = np.where(np.arange(n) % 2 == 0, 100.0, 160.0).astype(np.float64)
    dates = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    path = tmp_path / "AAPL_weekly.parquet"
    pd.DataFrame(
        {
            "datetime": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1.0),
        }
    ).to_parquet(path)
    from forecast.diagnostics import main as diag_main

    rc = diag_main(["--data-dir", str(tmp_path), "--interval", "weekly", "--delete-mixed"])
    assert rc == 0
    assert not path.exists()


def test_overnight_rejects_unadjusted_open_vs_adj_close(tmp_path: Path):
    n = 80
    close = np.full(n, 50.0)
    open_px = np.full(n, 100.0)
    dates = pd.bdate_range("2018-01-02", periods=n)
    path = tmp_path / "AAPL_daily.parquet"
    pd.DataFrame(
        {
            "datetime": dates,
            "open": open_px,
            "high": open_px,
            "low": close,
            "close": close,
            "volume": np.full(n, 1.0),
        }
    ).to_parquet(path)
    cfg = DataConfig(interval="daily", label_return="overnight", allow_mixed_prices=False)
    with pytest.raises(ValueError, match="unadjusted opens"):
        assert_calendar_price_quality(path, cfg)
    ok = DataConfig(interval="daily", label_return="close", allow_mixed_prices=False)
    assert_calendar_price_quality(path, ok)
