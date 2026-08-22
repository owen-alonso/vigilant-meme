"""Small input-dependent network that emits raw dynamic SSM parameters."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_lm.config import MambaConfig
from mamba_lm.dynamic.params import DynamicParameters


class DynamicWeightController(nn.Module):
    """Map the SSM input sequence to token-dependent SSM parameter logits.

    Architecture (V1, Dynamic A only)::

        x [B, L, d_in]
          -> Linear(d_in, controller_dim)
          -> SiLU
          -> Linear(controller_dim, d_state)   # zero-initialized
          -> delta_A [B, L, N]

    The output projection is zero-initialized so that at step 0
    ``tanh(delta_A) = 0`` and Dynamic A matches the static baseline.
    """

    def __init__(self, config: MambaConfig, d_in: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_state = config.d_state
        self.controller_dim = config.resolved_controller_dim()
        self.dynamic_A = config.dynamic_A
        self.dynamic_B = config.dynamic_B
        self.dynamic_C = config.dynamic_C
        self.dynamic_dt = config.dynamic_dt
        self.dynamic_gate = config.dynamic_gate

        self.in_proj = nn.Linear(d_in, self.controller_dim, bias=True)
        # V1: a single head producing N logits per token (broadcast over D).
        out_dim = 0
        if self.dynamic_A:
            out_dim += self.d_state
        if self.dynamic_B or self.dynamic_C or self.dynamic_dt or self.dynamic_gate:
            raise NotImplementedError("V1 controller emits Dynamic A only")
        if out_dim == 0:
            raise ValueError("Controller constructed with no dynamic outputs")

        self.out_proj = nn.Linear(self.controller_dim, out_dim, bias=True)
        self._zero_init_output()

    def _zero_init_output(self) -> None:
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> DynamicParameters:
        """Run the controller over the full sequence.

        Args:
            x: SSM input after conv + SiLU, shape [B, L, d_in]

        Returns:
            DynamicParameters with delta_A of shape [B, L, N]
        """
        batch, seq_len, d_in = x.shape
        if d_in != self.d_in:
            raise ValueError(f"controller expected d_in={self.d_in}, got {d_in}")

        hidden = F.silu(self.in_proj(x))  # [B, L, controller_dim]
        raw = self.out_proj(hidden)  # [B, L, out_dim]

        delta_A = None
        offset = 0
        if self.dynamic_A:
            delta_A = raw[..., offset : offset + self.d_state]
            offset += self.d_state
            # delta_A: [B, L, N] — one scale per state dim, per token.
            if delta_A.shape != (batch, seq_len, self.d_state):
                raise RuntimeError(
                    f"delta_A shape {tuple(delta_A.shape)} != "
                    f"{(batch, seq_len, self.d_state)}"
                )

        return DynamicParameters(delta_A=delta_A)
