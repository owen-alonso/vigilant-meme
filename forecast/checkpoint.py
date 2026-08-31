"""Forecast checkpoint save/load with validated deserialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from forecast.config import DataConfig, ForecastModelConfig, ForecastTrainConfig
from forecast.model import ReturnForecaster
from mamba_lm.checkpoint_io import load_checkpoint_dict


def _to_numpy_list(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, list):
        return value
    return np.asarray(value, dtype=np.float32).tolist()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def save_forecast_checkpoint(
    path: str | Path,
    *,
    model: ReturnForecaster,
    model_cfg: ForecastModelConfig,
    data_cfg: DataConfig,
    train_cfg: ForecastTrainConfig,
    feature_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    symbols: list[dict[str, Any]],
    step: int,
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "data_config": data_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "feature_names": feature_names,
        "feature_mean": _to_numpy_list(feature_mean),
        "feature_std": _to_numpy_list(feature_std),
        "symbols": symbols,
        "step": step,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    torch.save(payload, path)


def load_forecast_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    model: ReturnForecaster | None = None,
) -> dict[str, Any]:
    state = load_checkpoint_dict(path, map_location=map_location)
    required = {
        "model",
        "model_config",
        "data_config",
        "feature_mean",
        "feature_std",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(f"forecast checkpoint missing keys: {sorted(missing)}")

    state["feature_mean"] = _to_numpy(state["feature_mean"])
    state["feature_std"] = _to_numpy(state["feature_std"])

    if model is not None:
        model.load_state_dict(state["model"])

    return state


def load_forecaster(
    checkpoint: str | Path, device: torch.device
) -> tuple[ReturnForecaster, dict[str, Any]]:
    state = load_forecast_checkpoint(checkpoint, map_location=device)
    model_cfg = ForecastModelConfig.from_dict(state["model_config"])
    model = ReturnForecaster(model_cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def uncertainty_is_trained(model: ReturnForecaster, state: dict[str, Any]) -> bool:
    """Only the gaussian NLL trains log_sigma; otherwise the column is noise."""
    train_cfg = state.get("train_config") or {}
    return bool(model.config.heteroscedastic) and train_cfg.get("loss") == "gaussian"
