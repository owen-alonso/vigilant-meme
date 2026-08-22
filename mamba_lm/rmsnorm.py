"""RMSNorm used as the pre-norm around every Mamba block."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: x * rsqrt(mean(x^2) + eps) * weight."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] (or any tensor whose last dim is d_model)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight
