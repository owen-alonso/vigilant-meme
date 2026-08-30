"""Configuration for the next-hour return forecaster."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from mamba_lm.config import MambaConfig


Precision = Literal["fp32", "fp16", "bf16"]
LossName = Literal["huber", "mse", "gaussian"]

# Regular US equity trading session, matching the bars in data/.
SESSION_START_MINUTE = 9 * 60 + 30  # 09:30
BARS_PER_SESSION = 390  # 09:30 .. 15:59 inclusive


@dataclass
class DataConfig:
    """How raw OHLCV bars become model inputs and a next-hour target.

    Every field here changes the meaning of the features, so the whole object is
    stored in each checkpoint and reused verbatim at inference time.
    """

    data_dir: str = "data"
    horizon: int = 60  # target lookahead, in bars (= 1 hour of 1-minute bars)
    seq_len: int = 256
    stride: int = 64  # window stride; < seq_len gives overlapping windows
    # Leading bars of each window are unsupervised: the SSM state is still
    # warming up there, so scoring them would train and measure a model that
    # has barely any history.
    min_context: int = 64

    # Volatility scaling. Returns are divided by an EWM realized-vol estimate so
    # the model sees (and predicts) unitless, roughly stationary quantities.
    vol_halflife: int = BARS_PER_SESSION
    vol_floor: float = 1e-5

    # Rolling window used to standardize slow-moving level features.
    z_window: int = 5 * BARS_PER_SESSION
    z_min_periods: int = BARS_PER_SESSION

    # Sessions with fewer real (non-forward-filled) bars than this are dropped:
    # a day with 5 prints cannot support a meaningful 60-minute target.
    min_session_bars: int = 30
    warmup_bars: int = 2 * BARS_PER_SESSION

    clip: float = 8.0  # feature clamp, in standardized units
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # A label is only valid if the horizon bar itself is a real print, not a
    # forward-filled hole. Otherwise "next hour return" can be a stale zero.
    require_horizon_traded: bool = True
    # Session joins wider than this (calendar days) are not treated as a
    # one-minute return; diffs that cross the gap become NaN.
    max_session_gap_days: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class ForecastModelConfig:
    """Mamba backbone sized for continuous financial features."""

    n_features: int = 18
    d_model: int = 96
    n_layer: int = 4
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    dropout: float = 0.1
    # Extra log-sigma head. Only trained when ForecastTrainConfig.loss is
    # "gaussian"; keep False unless you opt into that loss.
    heteroscedastic: bool = False

    dynamic_weights: bool = False
    dynamic_A: bool = True
    dynamic_strength: float = 0.1
    dynamic_controller_dim: int | None = None

    def mamba_config(self) -> MambaConfig:
        """Backbone config. Vocab fields are unused by the regression model."""
        return MambaConfig(
            d_model=self.d_model,
            n_layer=self.n_layer,
            d_state=self.d_state,
            expand=self.expand,
            d_conv=self.d_conv,
            dynamic_weights=self.dynamic_weights,
            dynamic_A=self.dynamic_A,
            dynamic_strength=self.dynamic_strength,
            dynamic_controller_dim=self.dynamic_controller_dim,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastModelConfig:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class ForecastTrainConfig:
    batch_size: int = 16
    epochs: int = 8
    max_steps: int | None = None
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    loss: LossName = "huber"
    huber_delta: float = 1.0
    precision: Precision = "bf16"
    seed: int = 42
    log_interval: int = 25
    eval_interval: int = 250
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints/forecast"
    early_stop_evals: int = 8  # stop after N evals with no val-IC improvement

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastTrainConfig:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def validate_loss_head(model_cfg: ForecastModelConfig, train_cfg: ForecastTrainConfig) -> None:
    """Huber/MSE must not ship an untrained uncertainty head; gaussian needs one."""
    if train_cfg.loss == "gaussian" and not model_cfg.heteroscedastic:
        raise ValueError("loss='gaussian' requires ForecastModelConfig.heteroscedastic=True")
    if train_cfg.loss != "gaussian" and model_cfg.heteroscedastic:
        raise ValueError(
            "heteroscedastic=True is only trained under loss='gaussian'; "
            "set heteroscedastic=False or switch the loss"
        )
