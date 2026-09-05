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
    assert torch.allclose(model.state_dict()["skip.weight"], loaded.state_dict()["skip.weight"])


def test_forecast_checkpoint_loads_pre_skip_weights(tmp_path: Path):
    model_cfg = ForecastModelConfig(n_features=4, heteroscedastic=False, linear_skip=True)
    model = ReturnForecaster(model_cfg)
    path = tmp_path / "old.pt"
    payload = {
        "schema_version": 1,
        "model": {k: v for k, v in model.state_dict().items() if not k.startswith("skip.")},
        "model_config": model_cfg.to_dict(),
        "data_config": DataConfig().to_dict(),
        "train_config": ForecastTrainConfig().to_dict(),
        "feature_names": ["a", "b", "c", "d"],
        "feature_mean": [0.0, 0.0, 0.0, 0.0],
        "feature_std": [1.0, 1.0, 1.0, 1.0],
        "symbols": [],
        "step": 1,
        "metrics": {},
    }
    torch.save(payload, path)
    loaded = ReturnForecaster(model_cfg)
    load_forecast_checkpoint(path, model=loaded)
    assert torch.equal(loaded.skip.weight, torch.zeros_like(loaded.skip.weight))
