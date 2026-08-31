"""Forecast data causality, horizon prints, session ffill, timezone, close>0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.config import DataConfig, ForecastModelConfig, ForecastTrainConfig, validate_loss_head
from forecast.data import (
    _session_naive_datetime,
    build_session_grid,
    compute_features,
    load_bars,
)
from forecast.model import ReturnForecaster
from forecast.training import masked_loss


def _tiny_cfg(**kwargs) -> DataConfig:
    defaults = dict(
        horizon=2,
        warmup_bars=3,
        vol_halflife=2,
        z_window=5,
        z_min_periods=2,
        min_session_bars=2,
        require_horizon_traded=True,
        max_session_gap_days=4,
    )
    defaults.update(kwargs)
    return DataConfig(**defaults)


def _grid(n: int = 20, traded: np.ndarray | None = None) -> pd.DataFrame:
    close = 10.0 + np.linspace(0, 0.2, n)
    if traded is None:
        traded = np.ones(n)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-02 09:30", periods=n, freq="min"),
            "session": pd.Timestamp("2024-01-02"),
            "mos": np.arange(n),
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": np.full(n, 100.0),
            "traded": traded.astype(np.float64),
        }
    )


def test_valid_requires_horizon_print():
    traded = np.ones(20)
    traded[10] = 0.0
    panel = compute_features(_grid(20, traded), _tiny_cfg(horizon=2))
    # mos=8 looks 2 bars ahead at mos=10, which is untraded.
    row = panel.loc[panel["mos"] == 8].iloc[0]
    assert row["horizon_traded"] == 0
    assert not bool(row["valid"])
    # A pair of traded bars with a traded horizon can be valid after warmup.
    later = panel.loc[panel["mos"] == 12].iloc[0]
    assert later["horizon_traded"] == 1


def test_stale_horizon_allowed_when_flag_off():
    traded = np.ones(20)
    traded[10] = 0.0
    panel = compute_features(_grid(20, traded), _tiny_cfg(horizon=2, require_horizon_traded=False))
    row = panel.loc[panel["mos"] == 8].iloc[0]
    assert row["horizon_traded"] == 0
    # May still fail warmup/features; the extra constraint is not applied.
    assert row["traded"] == 1


def test_ffill_does_not_cross_sessions(tmp_path: Path):
    rows = []
    for i, minute in enumerate([0, 1]):
        rows.append(
            {
                "datetime": pd.Timestamp("2024-01-02 09:30") + pd.Timedelta(minutes=minute),
                "Open": 10.0,
                "High": 10.0,
                "Low": 10.0,
                "Close": 10.0 + i,
                "Volume": 1.0,
            }
        )
    rows.append(
        {
            "datetime": pd.Timestamp("2024-01-09 10:00"),
            "Open": 50.0,
            "High": 50.0,
            "Low": 50.0,
            "Close": 50.0,
            "Volume": 1.0,
        }
    )
    path = tmp_path / "TEST_clean_1min.parquet"
    pd.DataFrame(rows).to_parquet(path)
    df = load_bars(path)
    grid = build_session_grid(df, _tiny_cfg(min_session_bars=1))
    day2 = grid[grid["session"] == pd.Timestamp("2024-01-09")]
    morning = day2[day2["mos"] < 30]
    assert morning.empty
    first = day2.iloc[0]
    assert int(first["mos"]) == 30
    assert float(first["close"]) == 50.0


def test_large_gap_does_not_become_one_minute_return():
    n = 12
    g1 = _grid(n)
    g2 = _grid(n)
    g2["session"] = pd.Timestamp("2024-01-20")
    g2["datetime"] = pd.date_range("2024-01-20 09:30", periods=n, freq="min")
    g2["close"] = g2["close"] + 5.0
    grid = pd.concat([g1, g2], ignore_index=True)
    panel = compute_features(grid, _tiny_cfg(max_session_gap_days=4, warmup_bars=0))
    join = panel[panel["session"] == pd.Timestamp("2024-01-20")].iloc[0]
    # ret_1 is filled to 0 after validity; the join itself must not be labelled.
    assert not bool(join["valid"])


def test_utc_timestamps_convert_to_eastern():
    # 14:30 UTC in January is 09:30 EST.
    utc = pd.Series(pd.to_datetime(["2024-01-02 14:30:00"]).tz_localize("UTC"))
    naive = _session_naive_datetime(utc)
    assert naive.dt.tz is None
    assert int(naive.dt.hour.iloc[0]) == 9
    assert int(naive.dt.minute.iloc[0]) == 30


def test_nonpositive_close_raises(tmp_path: Path):
    path = tmp_path / "BAD_clean_1min.parquet"
    pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-02 09:30")],
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [0.0],
            "Volume": [1.0],
        }
    ).to_parquet(path)
    with pytest.raises(ValueError, match="close <= 0"):
        load_bars(path)


def test_features_ignore_future_price_spike():
    grid = _grid(16)
    cfg = _tiny_cfg(warmup_bars=4, horizon=2)
    base = compute_features(grid, cfg)
    spiked = grid.copy()
    spiked.loc[15, "close"] = 999.0
    alt = compute_features(spiked, cfg)
    # Position 8 must not see the spike at 15 (beyond its causal past).
    cols = ["ret_1", "ret_5", "vol_level"]
    assert np.allclose(
        base.loc[8, cols].to_numpy(dtype=np.float64),
        alt.loc[8, cols].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_validate_loss_head_rejects_mismatches():
    with pytest.raises(ValueError):
        validate_loss_head(
            ForecastModelConfig(heteroscedastic=True),
            ForecastTrainConfig(loss="huber", sigma_aux_weight=0.0),
        )
    with pytest.raises(ValueError):
        validate_loss_head(
            ForecastModelConfig(heteroscedastic=False),
            ForecastTrainConfig(loss="gaussian"),
        )
    validate_loss_head(
        ForecastModelConfig(heteroscedastic=True),
        ForecastTrainConfig(loss="huber", sigma_aux_weight=0.5),
    )


def test_predict_restores_training_flag():
    import torch

    model = ReturnForecaster(
        ForecastModelConfig(n_features=18, d_model=16, n_layer=1, dropout=0.5)
    )
    model.train()
    x = torch.randn(2, 8, 18)
    model.predict(x)
    assert model.training


def test_masked_loss_all_masked_is_nan():
    import torch

    mean = torch.zeros(2, 4)
    log_sigma = torch.zeros_like(mean)
    target = torch.ones_like(mean)
    mask = torch.zeros_like(mean)
    loss = masked_loss(mean, log_sigma, target, mask, ForecastTrainConfig(loss="huber"))
    assert torch.isnan(loss)
