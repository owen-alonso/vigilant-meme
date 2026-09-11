"""Configuration for the equity return forecaster."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

BarInterval = Literal["daily", "weekly", "monthly", "1min", "5min", "15min", "30min", "60min"]

from mamba_lm.config import MambaConfig
from mamba_lm.config_utils import filter_dataclass_fields


Precision = Literal["fp32", "fp16", "bf16"]
LossName = Literal["huber", "mse", "gaussian"]

# Regular US equity trading session, matching the bars in data/.
SESSION_START_MINUTE = 9 * 60 + 30  # 09:30
BARS_PER_SESSION = 390  # 09:30 .. 15:59 inclusive


@dataclass
class DataConfig:
    """How raw OHLCV bars become model inputs and a forward-return target.

    Default vendor is Alpha Vantage daily bars (free TIME_SERIES_DAILY).
    Split-adjusted daily and 1-minute TIME_SERIES_INTRADAY are premium
    endpoints. 1-minute uses the session-grid path when ``interval='1min'``.

    Every field here changes the meaning of the features, so the whole object is
    stored in each checkpoint and reused verbatim at inference time.
    """

    data_dir: str = "data"
    interval: BarInterval = "daily"
    horizon: int = 1  # target lookahead, in bars (1 daily bar = next session)
    seq_len: int = 128
    stride: int = 16  # window stride; < seq_len gives overlapping windows
    # Leading bars of each window are unsupervised: the SSM state is still
    # warming up there, so scoring them would train and measure a model that
    # has barely any history.
    min_context: int = 32

    # Volatility scaling. Returns are divided by an EWM realized-vol estimate so
    # the model sees (and predicts) unitless, roughly stationary quantities.
    vol_halflife: int = 21
    vol_floor: float = 1e-5

    # Rolling window used to standardize slow-moving level features.
    z_window: int = 252
    z_min_periods: int = 21

    # Intraday only: sessions with fewer real bars than this are dropped.
    min_session_bars: int = 1
    warmup_bars: int = 21

    clip: float = 8.0  # feature clamp, in standardized units
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # A label is only valid if the horizon bar itself is a real print, not a
    # forward-filled hole. Otherwise "next hour return" can be a stale zero.
    require_horizon_traded: bool = True
    # Session joins wider than this (calendar days) are not treated as a
    # one-minute return; diffs that cross the gap become NaN.
    max_session_gap_days: int = 10
    # Drop labels whose forward log-return looks like a split / halt, not a
    # tradeable move. 0 disables. 0.40 catches 2:1 and larger corporate actions
    # without dropping ordinary crash weeks.
    max_abs_log_return: float = 0.40
    # Drop labels whose vol-normalized target is outside this band (same scale
    # as feature ``clip``). Pearson IC is otherwise dominated by one jump.
    max_abs_target: float = 8.0
    # Train only the last K bars of each window (0 = every bar after min_context).
    # Calendar defaults use 1 so train/eval match generate.py (last bar only).
    supervise_last: int = 1
    # Score unique last bars (the trading object), not every labelled position.
    eval_last_bar: bool = True
    # One train_end / val_end for every symbol (union of session dates).
    global_calendar_split: bool = True
    # Same-bar market feature + residual label vs this ticker when its parquet exists.
    benchmark_symbol: str = "SPY"
    # y = (r_{t+h} - beta_t * r_mkt_{t+h}) / sigma. beta uses data through t only.
    residual_target: bool = True
    beta_halflife: int = 63
    # Cross-section batches when at least this many names print on a date.
    # 30 keeps the ridge off sparse 1970s panels; 8 is the absolute floor in tests.
    cross_section_min_names: int = 30
    # Weekly mixed adjusted/raw files have lag-1 autocorr << 0. Set True to skip.
    allow_mixed_prices: bool = False
    # Restrict loaded parquets to a train-era-locked list or all.
    universe: str = ""
    # Same-day cross-sectional z-scores of momentum / volume (known at close).
    cs_zscore: bool = True
    # y = r_{t+h} - beta_t * r_sector_{t+h} when the sector ETF parquet exists.
    # beta still uses data through t only; missing sector ETFs fall back to SPY.
    sector_residual: bool = True
    # Train / score single-name equities only. SPY and sector/macro ETFs still
    # load for features and hedges, but they are not book names.
    equities_only: bool = True
    # Drop *train* labels before this date. Empty = keep every train session.
    # Val/test calendar cuts are unchanged (locked test window).
    train_from: str = "1999-01-01"
    # y = r - b_mkt * SPY_fwd - b_sec * sector_fwd (two-factor, causal betas).
    double_residual: bool = False
    # Also residualize ret_* features vs same-bar hedges (not labels).
    residualize_features: bool = False
    # Optional third factor vs a mapped industry ETF when that parquet exists.
    industry_residual: bool = False

    def is_daily(self) -> bool:
        return self.interval == "daily"

    def is_intraday(self) -> bool:
        return str(self.interval).endswith("min")

    def is_calendar(self) -> bool:
        """Daily / weekly / monthly bars (no 390-minute session grid)."""
        return not self.is_intraday()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        payload = dict(data)
        # Older checkpoints were 1-minute / next-hour and had no interval field.
        if "interval" not in payload and int(payload.get("horizon", 1)) >= 60:
            payload["interval"] = "1min"
        # New protocol flags: absent on old checkpoints means the old behavior.
        if "sector_residual" not in payload:
            payload["sector_residual"] = False
        if "equities_only" not in payload:
            payload["equities_only"] = False
        if "train_from" not in payload:
            payload["train_from"] = ""
        if "double_residual" not in payload:
            payload["double_residual"] = False
        if "residualize_features" not in payload:
            payload["residualize_features"] = False
        if "industry_residual" not in payload:
            payload["industry_residual"] = False
        return cls(**filter_dataclass_fields(cls, payload))


def interval_data_kwargs(interval: str) -> dict[str, Any]:
    """Lookbacks and window sizes that match the bar interval.

    Dataclass defaults on ``DataConfig`` are the daily bundle. Passing
    ``--interval weekly`` without these would keep a 128-bar / 252-bar daily
    context, which leaves ~two val windows and a 5-year z-score on weekly bars.
    """
    if interval == "weekly":
        return {
            "seq_len": 52,
            "stride": 1,
            "min_context": 13,
            "vol_halflife": 12,
            "z_window": 52,
            "z_min_periods": 8,
            "warmup_bars": 26,
            "max_abs_log_return": 0.40,
            "supervise_last": 1,
        }
    if interval == "monthly":
        return {
            "seq_len": 36,
            "stride": 1,
            "min_context": 8,
            "vol_halflife": 6,
            "z_window": 24,
            "z_min_periods": 6,
            "warmup_bars": 12,
            "max_abs_log_return": 0.50,
            "supervise_last": 1,
        }
    if str(interval).endswith("min"):
        bars = {
            "1min": BARS_PER_SESSION,
            "5min": 78,
            "15min": 26,
            "30min": 13,
            "60min": 7,
        }.get(interval, BARS_PER_SESSION)
        minute = interval == "1min"
        return {
            "seq_len": 256 if minute else 128,
            "stride": 64 if minute else 16,
            "min_context": 64 if minute else 32,
            "vol_halflife": bars,
            "z_window": 5 * bars,
            "z_min_periods": max(8, bars // 2),
            "warmup_bars": 2 * bars,
            "max_abs_log_return": 0.15,
            "supervise_last": 0,
        }
    return {
        "seq_len": 128,
        "stride": 1,
        "min_context": 32,
        "vol_halflife": 21,
        "z_window": 252,
        "z_min_periods": 21,
        "warmup_bars": 21,
        "max_abs_log_return": 0.40,
        "supervise_last": 1,
    }


def interval_model_kwargs(interval: str) -> dict[str, Any]:
    """SSM step-size prior and a tiny residual for calendar bars."""
    if interval in ("weekly", "monthly"):
        return {
            "dt_min": 0.05,
            "dt_max": 1.0,
            "d_model": 32,
            "n_layer": 1,
            "d_state": 8,
        }
    if interval == "daily":
        return {
            "dt_min": 1e-3,
            "dt_max": 0.1,
            "d_model": 32,
            "n_layer": 1,
            "d_state": 8,
        }
    return {"dt_min": 1e-3, "dt_max": 0.1}


@dataclass
class ForecastModelConfig:
    """Mamba backbone sized for continuous financial features."""

    n_features: int = 44
    d_model: int = 96
    n_layer: int = 4
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    # Stem dropout sits on the only residual that could carry ``ret_*``.
    # Off by default so the linear skip (and the identity Mamba path) survive.
    dropout: float = 0.0
    # Residual-std head. Trained by gaussian NLL, or by sigma_aux_weight when
    # the mean loss is Huber/MSE. generate.py maps it to a confidence score.
    heteroscedastic: bool = False
    # Direct features -> mean. The Mamba head stays zero-init so training
    # starts as a linear readout of the already vol-normalized lags.
    linear_skip: bool = True
    dt_min: float = 1e-3
    dt_max: float = 0.1

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
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dynamic_weights=self.dynamic_weights,
            dynamic_A=self.dynamic_A,
            dynamic_strength=self.dynamic_strength,
            dynamic_controller_dim=self.dynamic_controller_dim,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastModelConfig:
        return cls(**filter_dataclass_fields(cls, data))


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
    # Huber owns prediction scale. Keep it smaller than the IC term so the
    # optimizer does not trade rank agreement for MSE.
    location_loss_weight: float = 0.4
    # Mix 1 - Pearson over the pooled labelled batch (same IC val reports).
    ic_loss_weight: float = 2.0
    # Clip pred/target to +/- this many vol units before Pearson / IC loss.
    ic_winsor: float = 3.0
    # Direction on moves larger than sign_min_abs (volatility units).
    sign_loss_weight: float = 0.4
    sign_min_abs: float = 0.25
    # Pairwise RankNet on labelled bars in the batch (Spearman-like).
    # Default matches the CS ranking objective; do not raise ic_loss_weight.
    rank_loss_weight: float = 1.0
    # Match pred std to target std so Pearson cannot explode |pred|.
    pred_std_weight: float = 0.5
    # Closed-form ridge readout copied into the linear skip at step 0.
    # 0 keeps Xavier init.
    ridge_skip: float = 10.0
    # Date-demean features/targets before ridge (the CS linear baseline).
    ridge_cs_demean: bool = True
    # Huber/MSE/NLL on within-date demeaned pred/target when a date has breadth.
    cs_center_loss: bool = True
    # After ridge, freeze the skip so AdamW cannot decay the linear baseline.
    skip_lr_mult: float = 0.0
    freeze_skip: bool = True
    # Apply ridge, log last-bar train/val/test IC, write best.pt, exit (no AdamW).
    skip_only: bool = False
    # Frozen skip vs causal expanding/rolling refit (walk-forward uses labels < t).
    ridge_window: str = "frozen"
    # Trailing calendar days for rolling ridge. Ignored when window=frozen.
    ridge_lookback_days: int = 1260
    # Fit ridge on within-date ranks of y (the CS trading object).
    ridge_rank_target: bool = True
    # Within-date z-score features in the ridge design (kills calendar constants).
    ridge_cs_zscore: bool = False
    # Drop long TS + calendar from the skip (val-selected with CS products).
    ridge_features: str = "no_long_ts"
    # Exponential recency weights on train dates (0 = uniform).
    ridge_date_halflife: float = 0.0
    # Winsorize raw y within date before the ridge (ignored when rank_target).
    ridge_y_winsor: float = 0.0
    # Winsorize features within date (in residual-std units). Val-selected 3.
    ridge_feat_winsor: float = 3.0
    # Drop the top this fraction of train dates by residual dispersion.
    ridge_drop_disp_q: float = 0.0
    # Huber IRLS delta in MAD units (0 = closed-form ridge only).
    ridge_huber: float = 0.0
    # Zero weights that flip the train univariate CS IC sign.
    ridge_sign_constrain: bool = False
    # Drop dot-com + GFC dates from the frozen skip fit.
    ridge_drop_crashes: bool = False
    # ``ridge`` / ``listnet`` / ``ranknet`` skip objective.
    ridge_objective: str = "ridge"
    # Equalize per-year total weight in the frozen skip (train only).
    ridge_year_balance: bool = False
    # Year-sign stability mask: ``train`` or ``train_val`` (empty = off).
    ridge_year_stable: str = ""
    # ListNet (softmax CE) within date. 0 keeps RankNet-only ranking.
    listnet_loss_weight: float = 0.0
    # Residual-std head. Trained by gaussian NLL, or by sigma_aux_weight when
    # the mean loss is Huber/MSE. Default 0 matches heteroscedastic=False.
    sigma_aux_weight: float = 0.0
    precision: Precision = "bf16"
    seed: int = 42
    log_interval: int = 25
    eval_interval: int = 250
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints/forecast"
    # 0 = never stop. Default stops after N evals with no new best val IC.
    # Plateau must not reset this counter or a 1e6-step run never ends.
    early_stop_evals: int = 24
    # After this many evals with no new best val IC, multiply the scheduled LR
    # (warmup/cosine) by lr_plateau_factor and restore best.pt. 0 disables.
    lr_plateau_evals: int = 8
    lr_plateau_factor: float = 0.5
    lr_plateau_min_scale: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastTrainConfig:
        return cls(**filter_dataclass_fields(cls, data))


def validate_loss_head(model_cfg: ForecastModelConfig, train_cfg: ForecastTrainConfig) -> None:
    """Gaussian NLL needs a sigma head; Huber/MSE need aux weight if that head exists."""
    if train_cfg.loss == "gaussian" and not model_cfg.heteroscedastic:
        raise ValueError("loss='gaussian' requires ForecastModelConfig.heteroscedastic=True")
    for name in (
        "location_loss_weight",
        "ic_loss_weight",
        "sign_loss_weight",
        "rank_loss_weight",
        "listnet_loss_weight",
        "pred_std_weight",
        "ridge_skip",
        "skip_lr_mult",
    ):
        if getattr(train_cfg, name) < 0:
            raise ValueError(f"{name} must be >= 0")
    if train_cfg.ic_winsor <= 0:
        raise ValueError("ic_winsor must be > 0")
    if train_cfg.sign_min_abs < 0:
        raise ValueError("sign_min_abs must be >= 0")
    if model_cfg.heteroscedastic and train_cfg.loss != "gaussian" and train_cfg.sigma_aux_weight <= 0:
        raise ValueError(
            "heteroscedastic=True with Huber/MSE needs sigma_aux_weight > 0 "
            "so the confidence head is actually trained"
        )
    if not model_cfg.heteroscedastic and train_cfg.sigma_aux_weight > 0:
        raise ValueError(
            "sigma_aux_weight > 0 requires ForecastModelConfig.heteroscedastic=True"
        )
