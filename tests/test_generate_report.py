"""Readable generate.py report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecast.config import DataConfig, ForecastModelConfig
from forecast.generate import format_stock_column_report, uncertainty_is_trained
from forecast.model import ReturnForecaster


def test_format_stock_column_report_headings():
    result = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-02 15:59")],
            "close": [25.0],
            "traded": [True],
            "pred_norm": [0.1],
            "scale": [0.01],
            "pred_return_bps": [10.0],
            "pred_price_1h": [25.025],
            "realized_bps": [float("nan")],
        }
    )
    text = format_stock_column_report(
        {"SPAB": result},
        trained_on="SPAB",
        checkpoint=Path("checkpoints/forecast/best.pt"),
        data_cfg=DataConfig(),
        context=256,
        device=torch.device("cpu"),
        notes=["No trained uncertainty: the loss was not gaussian NLL."],
    )
    assert "NEXT-HOUR RETURN FORECAST" in text
    assert "SPAB" in text
    assert "Predicted move" in text or "pred (bp)" in text
    assert "basis points" in text
    assert "No trained uncertainty" in text
    assert "pred_uncertainty" not in text


def test_uncertainty_is_trained_requires_gaussian():
    model = ReturnForecaster(ForecastModelConfig(n_features=18, heteroscedastic=False))
    assert not uncertainty_is_trained(model, {"train_config": {"loss": "huber"}})
    model_h = ReturnForecaster(ForecastModelConfig(n_features=18, heteroscedastic=True))
    assert not uncertainty_is_trained(model_h, {"train_config": {"loss": "huber"}})
    assert uncertainty_is_trained(model_h, {"train_config": {"loss": "gaussian"}})
