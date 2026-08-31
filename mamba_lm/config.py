"""Model and training configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from mamba_lm.config_utils import filter_dataclass_fields


Precision = Literal["fp32", "fp16", "bf16"]
DynamicParameterization = Literal["elementwise", "diagonal", "low_rank"]


@dataclass
class MambaConfig:
    """Mamba-1 language-model hyperparameters plus Dynamic A flags.

    Tensor conventions used throughout the codebase:
        B  batch
        L  sequence length
        D  d_inner = expand * d_model   (SSM channels)
        N  d_state                      (SSM state dimension)
    """

    d_model: int = 128
    n_layer: int = 4
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    vocab_size: int = 256
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    dt_rank: int | str = "auto"
    dt_min: float = 1e-3
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    rms_eps: float = 1e-5

    # Dynamic weight controller (V1: Dynamic A only).
    dynamic_weights: bool = False
    dynamic_A: bool = True
    dynamic_B: bool = False
    dynamic_C: bool = False
    dynamic_dt: bool = False
    dynamic_controller_dim: int | None = None
    dynamic_strength: float = 0.1
    dynamic_parameterization: DynamicParameterization = "elementwise"
    dynamic_gate: bool = False
    dynamic_rank: int = 8
    dynamic_scale_eps: float = 1e-4

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.d_state <= 0:
            raise ValueError(f"d_state must be positive, got {self.d_state}")
        if self.expand <= 0:
            raise ValueError(f"expand must be positive, got {self.expand}")
        if self.dynamic_strength < 0:
            raise ValueError("dynamic_strength must be non-negative")
        # Reject reserved flags even when Dynamic A is off, so they cannot
        # sit silently unused on a baseline config.
        self._reject_unimplemented_flags()
        if self.dynamic_weights:
            self._validate_dynamic_v1()

    def _reject_unimplemented_flags(self) -> None:
        if self.dynamic_parameterization != "elementwise":
            raise NotImplementedError(
                "V1 only implements dynamic_parameterization='elementwise'; "
                f"got {self.dynamic_parameterization!r}"
            )
        unimplemented = []
        if self.dynamic_B:
            unimplemented.append("dynamic_B")
        if self.dynamic_C:
            unimplemented.append("dynamic_C")
        if self.dynamic_dt:
            unimplemented.append("dynamic_dt")
        if self.dynamic_gate:
            unimplemented.append("dynamic_gate")
        if unimplemented:
            raise NotImplementedError(
                "V1 only implements Dynamic A. Disable: " + ", ".join(unimplemented)
            )

    def _validate_dynamic_v1(self) -> None:
        self._reject_unimplemented_flags()
        if not self.dynamic_A:
            raise ValueError(
                "dynamic_weights=True requires dynamic_A=True in V1 "
                "(B/C/dt are not implemented yet)"
            )

    @property
    def d_inner(self) -> int:
        return int(self.expand * self.d_model)

    def resolved_dt_rank(self) -> int:
        if self.dt_rank == "auto":
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)

    def resolved_controller_dim(self) -> int:
        if self.dynamic_controller_dim is not None:
            return int(self.dynamic_controller_dim)
        return max(32, self.d_model // 4)

    def padded_vocab_size(self) -> int:
        multiple = self.pad_vocab_size_multiple
        return math.ceil(self.vocab_size / multiple) * multiple

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MambaConfig:
        return cls(**filter_dataclass_fields(cls, data))


@dataclass
class TrainConfig:
    """Optimization / data settings shared by baseline and Dynamic A runs."""

    batch_size: int = 8
    seq_len: int = 256
    max_steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 20
    grad_clip: float = 1.0
    precision: Precision = "fp32"
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 50
    eval_batches: int = 8
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    val_fraction: float = 0.1
    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        return cls(**filter_dataclass_fields(cls, data))
