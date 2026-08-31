"""Checkpoint save/load. Baseline checkpoints remain loadable into Dynamic A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mamba_lm.checkpoint_io import load_checkpoint_dict
from mamba_lm.config import MambaConfig
from mamba_lm.model import MambaLM


def save_checkpoint(
    path: str | Path,
    *,
    model: MambaLM,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": model.config.to_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    model: MambaLM | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    strict: bool | None = None,
) -> dict[str, Any]:
    """Load a checkpoint.

    If ``strict`` is None, missing/unexpected keys are allowed when the
    checkpoint and the live model disagree on ``dynamic_weights`` so that a
    baseline checkpoint can initialize a Dynamic A model (controller stays at
    its zero init).
    """
    ckpt = load_checkpoint_dict(path, map_location=map_location)
    config = MambaConfig.from_dict(ckpt["config"])

    if model is not None:
        if strict is None:
            ckpt_dynamic = bool(config.dynamic_weights)
            model_dynamic = bool(model.config.dynamic_weights)
            strict = ckpt_dynamic == model_dynamic
        incompatible = model.load_state_dict(ckpt["model"], strict=strict)
        ckpt["incompatible_keys"] = incompatible

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    ckpt["parsed_config"] = config
    return ckpt


def model_from_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    override: dict[str, Any] | None = None,
) -> tuple[MambaLM, dict[str, Any]]:
    ckpt = load_checkpoint_dict(path, map_location=map_location)
    cfg_dict = dict(ckpt["config"])
    if override:
        cfg_dict.update(override)
    config = MambaConfig.from_dict(cfg_dict)
    model = MambaLM(config)
    strict = override is None or "dynamic_weights" not in (override or {})
    if override and override.get("dynamic_weights") != ckpt["config"].get("dynamic_weights"):
        strict = False
    model.load_state_dict(ckpt["model"], strict=strict)
    return model, ckpt
