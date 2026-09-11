"""Locked CS residual protocol: skip-only ridge, selection, planted signal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forecast.config import DataConfig, ForecastModelConfig, ForecastTrainConfig
from forecast.synthetic import write_cs_momentum_universe
from forecast.training import selection_score, train


def test_skip_only_cs_ridge_recovers_planted_residual_signal(tmp_path: Path):
    data_dir = tmp_path / "data"
    write_cs_momentum_universe(data_dir, n_names=12, n_days=200, seed=0, rho=0.6)
    ckpt_dir = tmp_path / "ckpt"
    data_cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=24,
        stride=1,
        min_context=6,
        warmup_bars=12,
        vol_halflife=8,
        z_window=16,
        z_min_periods=6,
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=8,
        allow_mixed_prices=True,
        cs_zscore=True,
    )
    summary = train(
        data_cfg,
        ForecastModelConfig(d_model=16, n_layer=1, d_state=8, linear_skip=True),
        ForecastTrainConfig(
            skip_only=True,
            ridge_skip=1.0,
            freeze_skip=True,
            ridge_cs_demean=True,
            precision="fp32",
            checkpoint_dir=str(ckpt_dir),
        ),
        device=torch.device("cpu"),
        log_fn=None,
    )
    test = summary["test"]
    assert summary["cross_section"] is True
    assert summary["skip_only"] is True
    assert (ckpt_dir / "best.pt").exists()
    assert np.isfinite(test["cs_ic"])
    # Planted CS momentum should show up as positive mean CS IC, not coin-flip.
    assert test["cs_ic"] > 0.05
    assert test["cs_n_dates"] >= 5
    assert summary["best_val_ic"] == pytest.approx(selection_score(summary["skip_only_val"]))


def test_skip_only_selects_cs_ic_not_pooled_pearson():
    val = {"ic": 0.20, "cs_ic": 0.04, "loss": 0.5, "r2": 0.0, "direction": 0.5, "pred_std_bps": 1.0, "n": 10}
    assert selection_score(val) == pytest.approx(0.04)
