"""Forecast checkpoint round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from forecast.checkpoint import load_forecast_checkpoint, save_forecast_checkpoint
from forecast.config import DataConfig, ForecastModelConfig, ForecastTrainConfig
from forecast.model import ReturnForecaster


def test_forecast_checkpoint_round_trip(tmp_path: Path):
    model_cfg = ForecastModelConfig(n_features=4, heteroscedastic=False)
    model = ReturnForecaster(model_cfg)
    mean = np.zeros(4, dtype=np.float32)
    std = np.ones(4, dtype=np.float32)
    path = tmp_path / "best.pt"
    save_forecast_checkpoint(
        path,
        model=model,
        model_cfg=model_cfg,
        data_cfg=DataConfig(),
        train_cfg=ForecastTrainConfig(),
        feature_names=["a", "b", "c", "d"],
        feature_mean=mean,
        feature_std=std,
        symbols=[{"symbol": "TEST"}],
        step=7,
        metrics={"ic": 0.1},
    )
    loaded = ReturnForecaster(model_cfg)
    state = load_forecast_checkpoint(path, model=loaded)
    assert state["step"] == 7
    assert state["feature_mean"].shape == (4,)
    assert torch.allclose(model.state_dict()["head.weight"], loaded.state_dict()["head.weight"])
