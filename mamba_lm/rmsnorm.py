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
        # Variance in fp32 so fp16 autocast cannot overflow x^2.
        x32 = x.float()
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x32 * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight.float()).to(dtype=x.dtype)
