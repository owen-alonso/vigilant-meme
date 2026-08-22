"""Full Mamba-1 language model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mamba_lm.config import MambaConfig
from mamba_lm.mamba import MambaLayer
from mamba_lm.rmsnorm import RMSNorm


class MambaLM(nn.Module):
    """Token embedding → N residual Mamba layers → RMSNorm → tied LM head.

    Forward:
        input_ids [B, L] -> logits [B, L, padded_vocab]
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.padded_vocab = config.padded_vocab_size()
        self.embedding = nn.Embedding(self.padded_vocab, config.d_model)
        self.layers = nn.ModuleList([MambaLayer(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model, eps=config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, self.padded_vocab, bias=False)
        self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        x = self.embedding(input_ids)  # [B, L, d_model]
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def collect_dynamic_diagnostics(self) -> list[dict[str, Any]]:
        """Per-layer A-scale stats from the most recent dynamic forward."""
        reports: list[dict[str, Any]] = []
        for i, layer in enumerate(self.layers):
            mixer = layer.mixer
            if mixer.last_A_scale is None:
                continue
            stats = {
                "layer": i,
                **_scale_stats(mixer.last_A_scale),
            }
            reports.append(stats)
        return reports


def _scale_stats(scale: torch.Tensor) -> dict[str, float]:
    s = scale.detach().float()
    return {
        "dynamic_A_mean": float(s.mean().item()),
        "dynamic_A_std": float(s.std(unbiased=False).item()),
        "dynamic_A_min": float(s.min().item()),
        "dynamic_A_max": float(s.max().item()),
    }
