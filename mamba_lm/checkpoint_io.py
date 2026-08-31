"""Safe checkpoint loading with structural validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_SAFE_SCALAR_TYPES = (bool, int, float, str, type(None))


def _validate_checkpoint_value(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, _SAFE_SCALAR_TYPES):
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"checkpoint at {path}: dict keys must be strings, got {type(key)!r}"
                )
            _validate_checkpoint_value(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_checkpoint_value(nested, path=f"{path}[{index}]")
        return
    raise ValueError(
        f"checkpoint at {path}: unsupported type {type(value)!r}. "
        "Only tensors and plain Python collections are allowed."
    )


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    """Reject checkpoints that contain unexpected pickled object types."""
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint root must be a dict, got {type(payload)!r}")
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"checkpoint root keys must be strings, got {type(key)!r}")
        _validate_checkpoint_value(value, path=key)


def load_checkpoint_dict(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and validate that it contains only safe value types."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a dict, got {type(payload)!r}")
    validate_checkpoint_payload(payload)
    return payload
