"""Mamba-1 block: projections, causal conv, selective SSM, optional Dynamic A."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_lm.config import MambaConfig
from mamba_lm.diagnostics import SsmDiagnostics, record_ssm_diagnostics
from mamba_lm.dynamic.controller import DynamicWeightController
from mamba_lm.dynamic.modulator import DynamicParameterModulator
from mamba_lm.rmsnorm import RMSNorm
from mamba_lm.scan import ssm_forward


class MambaBlock(nn.Module):
    """Selective SSM mixer.

    Input/output shape: [B, L, d_model].

    When ``config.dynamic_weights`` is False the controller is not constructed
    and the numerical path is the standard Mamba-1 S6 block.

    When True, a batched DynamicWeightController produces ``delta_A [B, L, N]``
    which scales ``A_base = -exp(A_log)`` before discretization.
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.dt_rank = config.resolved_dt_rank()
        self.dynamic_weights = config.dynamic_weights

        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )
        # x_proj produces dt (low rank), B, and C from the SSM input.
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A_log: [D, N].  A = -exp(A_log) is always negative (stable timescales).
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, self.d_state).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=config.bias)

        self._init_dt_proj(config)

        if self.dynamic_weights:
            # Isolate controller init from the global RNG so a shared seed still
            # produces identical baseline Mamba weights (needed for fair ablations).
            rng_state = torch.get_rng_state()
            cuda_states = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            self.controller = DynamicWeightController(config, d_in=self.d_inner)
            self.modulator = DynamicParameterModulator(config)
            torch.set_rng_state(rng_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        else:
            self.controller = None
            self.modulator = None

    def _init_dt_proj(self, config: MambaConfig) -> None:
        dt_init_std = config.resolved_dt_rank() ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unknown dt_init: {config.dt_init}")

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # Inverse of softplus so softplus(bias) ≈ dt at init.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Args/returns: hidden [B, L, d_model]."""
        batch, seq_len, d_model = hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        projected = self.in_proj(hidden)  # [B, L, 2 * d_inner]
        x, residual = projected.split(self.d_inner, dim=-1)

        # Causal depthwise conv: pad d_conv-1, then crop back to L.
        x = x.transpose(1, 2)  # [B, d_inner, L]
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # [B, L, d_inner]
        x = F.silu(x)

        y = self._ssm(x)
        y = y * F.silu(residual)
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective SSM. ``x`` is the post-conv SiLU input, shape [B, L, D]."""
        batch, seq_len, d_inner = x.shape

        # Continuous-time A in fp32: A_base [D, N] is negative.
        A_base = -torch.exp(self.A_log.float())

        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2N]
        dt_rank, n = self.dt_rank, self.d_state
        dt_raw, B, C = x_dbl.split([dt_rank, n, n], dim=-1)
        # dt: [B, L, D], strictly positive via softplus.
        dt = F.softplus(self.dt_proj(dt_raw).float())

        if self.controller is not None and self.modulator is not None:
            params = self.controller(x)
            A, scale = self.modulator.modulate_A(A_base, params)
            record_ssm_diagnostics(
                self,
                SsmDiagnostics(
                    delta_A=params.delta_A.detach() if params.delta_A is not None else None,
                    a_scale=scale.detach(),
                ),
            )
        else:
            A = A_base  # [D, N]
            record_ssm_diagnostics(self, None)

        y = ssm_forward(x.float(), dt, A, B.float(), C.float(), self.D.float())
        return y.to(dtype=x.dtype)


class MambaLayer(nn.Module):
    """Pre-norm residual wrapper: x + mixer(RMSNorm(x))."""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.rms_eps)
        self.mixer = MambaBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x
