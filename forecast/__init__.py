"""Next-hour equity return forecasting on top of the Mamba backbone."""

from forecast.config import (
    DataConfig,
    ForecastModelConfig,
    ForecastTrainConfig,
)
from forecast.data import FEATURE_NAMES, build_datasets, build_panel
from forecast.model import ReturnForecaster

__all__ = [
    "DataConfig",
    "ForecastModelConfig",
    "ForecastTrainConfig",
    "FEATURE_NAMES",
    "build_datasets",
    "build_panel",
    "ReturnForecaster",
]
