"""Structured dynamic SSM parameter tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DynamicParameters:
    """Controller outputs consumed by DynamicParameterModulator.

    V1 populates ``delta_A`` only. Remaining fields are extension points for
    later dynamic B / C / dt / gate work and stay None.

    Shapes (V1):
        delta_A: [B, L, N]  raw controller logits for state-dimension A scale
        delta_B: [B, L, N]  (future)
        delta_C: [B, L, N]  (future)
        delta_dt: [B, L, D] (future)
        gate:    [B, L, 1]  (future)
    """

    delta_A: torch.Tensor | None = None
    delta_B: torch.Tensor | None = None
    delta_C: torch.Tensor | None = None
    delta_dt: torch.Tensor | None = None
    gate: torch.Tensor | None = None
