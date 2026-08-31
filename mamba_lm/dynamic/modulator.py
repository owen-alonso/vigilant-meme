"""Apply controller outputs to base SSM parameters without scattering logic."""

from __future__ import annotations

import torch

from mamba_lm.config import MambaConfig
from mamba_lm.dynamic.params import DynamicParameters


class DynamicParameterModulator:
    """Turn raw controller logits into a stable, token-dependent A_t.

    V1 (elementwise)::

        scale   = 1 + strength * tanh(delta_A)     # [B, L, N], bounded
        scale   = clamp(scale, min=eps)            # keep A negative
        A_t     = A_base * scale                   # [B, L, D, N]

    ``A_base = -exp(A_log)`` is formed *before* this module. We never add the
    controller output to ``A_log``.
    """

    def __init__(self, config: MambaConfig) -> None:
        if config.dynamic_parameterization != "elementwise":
            raise NotImplementedError(
                "V1 modulator only supports parameterization='elementwise'"
            )
        self.parameterization = config.dynamic_parameterization
        self.strength = float(config.dynamic_strength)
        self.eps = float(config.dynamic_scale_eps)
        self.d_state = config.d_state

    def modulate_A(
        self,
        A_base: torch.Tensor,
        params: DynamicParameters,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale the stable base A by a positive, token-dependent factor.

        Args:
            A_base: [D, N], already equal to -exp(A_log) (negative)
            params: controller outputs; ``delta_A`` is [B, L, N]

        Returns:
            A_t:    [B, L, D, N]  token-dependent continuous-time A
            scale:  [B, L, N]     the positive multiplier (for diagnostics)
        """
        if params.delta_A is None:
            raise ValueError("modulate_A requires params.delta_A")

        delta_A = params.delta_A
        if delta_A.dim() != 3 or delta_A.shape[-1] != self.d_state:
            raise ValueError(
                f"delta_A must be [B, L, N={self.d_state}], got {tuple(delta_A.shape)}"
            )
        if A_base.dim() != 2 or A_base.shape[-1] != self.d_state:
            raise ValueError(
                f"A_base must be [D, N={self.d_state}], got {tuple(A_base.shape)}"
            )

        delta_fp32 = delta_A.float()
        scale = 1.0 + self.strength * torch.tanh(delta_fp32)
        scale = scale.clamp(min=self.eps)

        A_t = A_base.unsqueeze(0).unsqueeze(0) * scale.unsqueeze(2)
        return A_t.to(dtype=A_base.dtype), scale

    def __repr__(self) -> str:
        return (
            f"DynamicParameterModulator("
            f"parameterization={self.parameterization!r}, "
            f"strength={self.strength}, eps={self.eps})"
        )
