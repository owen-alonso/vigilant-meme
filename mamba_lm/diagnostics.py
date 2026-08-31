"""Thread-local SSM diagnostic capture without mutating ``nn.Module`` state."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

_tls = threading.local()


@dataclass(frozen=True)
class SsmDiagnostics:
    delta_A: torch.Tensor | None = None
    a_scale: torch.Tensor | None = None


def _store() -> dict[int, SsmDiagnostics | None]:
    bucket = getattr(_tls, "ssm_by_block", None)
    if bucket is None:
        bucket = {}
        _tls.ssm_by_block = bucket
    return bucket


def record_ssm_diagnostics(block: object, diagnostics: SsmDiagnostics | None) -> None:
    """Record diagnostics for ``block`` from the current forward pass."""
    store = _store()
    key = id(block)
    if diagnostics is None:
        store.pop(key, None)
    else:
        store[key] = diagnostics


def get_ssm_diagnostics(block: object) -> SsmDiagnostics | None:
    """Return diagnostics recorded for ``block`` on this thread, if any."""
    return _store().get(id(block))


def clear_ssm_diagnostics() -> None:
    """Drop all recorded diagnostics on this thread."""
    _tls.ssm_by_block = {}
