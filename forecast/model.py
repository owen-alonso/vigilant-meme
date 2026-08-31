"""Mamba backbone adapted from next-token classification to return regression.

The language model embeds discrete tokens and produces a distribution over the
vocabulary. Here the input is a continuous feature vector per minute bar and the
output is a scalar per bar: the expected next-hour return, in volatility units.

The backbone (``MambaLayer``) is reused unchanged, so the Dynamic A ablation
carries over to this task.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from forecast.config import ForecastModelConfig
from mamba_lm.mamba import MambaLayer
from mamba_lm.model import collect_mixer_diagnostics
from mamba_lm.rmsnorm import RMSNorm


class ReturnForecaster(nn.Module):
    """Sequence-to-sequence regressor over 1-minute bars.

    Forward:
        features [B, L, n_features] -> mean [B, L], log_sigma [B, L]

    The model is causal, so position ``t`` of the output only sees bars ``<= t``.
    Every position is supervised during training, and only the last position is
    read at inference time.
    """

    def __init__(self, config: ForecastModelConfig) -> None:
        super().__init__()
        self.config = config
        mamba_cfg = config.mamba_config()

        self.input_proj = nn.Linear(config.n_features, config.d_model)
        self.input_norm = RMSNorm(config.d_model, eps=mamba_cfg.rms_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MambaLayer(mamba_cfg) for _ in range(config.n_layer)]
        )
        self.norm_f = RMSNorm(config.d_model, eps=mamba_cfg.rms_eps)

        n_out = 2 if config.heteroscedastic else 1
        self.head = nn.Linear(config.d_model, n_out)
        # Start at "no edge": predicting zero excess return is the right prior
        # for a return series, and it keeps early gradients small.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.dim() != 3:
            raise ValueError(
                f"features must be [B, L, n_features], got {tuple(features.shape)}"
            )
        if features.size(-1) != self.config.n_features:
            raise ValueError(
                f"expected n_features={self.config.n_features}, "
                f"got {features.size(-1)}"
            )

        x = self.dropout(self.input_norm(self.input_proj(features)))
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        out = self.head(x)  # [B, L, 1 or 2]

        mean = out[..., 0]
        if self.config.heteroscedastic:
            # Bounded so the NLL cannot escape by predicting infinite variance.
            # Unit-vol targets make [-3, 1] a usable residual-std range.
            log_sigma = out[..., 1].clamp(-3.0, 1.0)
        else:
            log_sigma = torch.zeros_like(mean)
        return mean, log_sigma

    @staticmethod
    def confidence_from_std(residual_std: torch.Tensor) -> torch.Tensor:
        """Map residual std (volatility units) to a score in ``(0, 1]``.

        A typical unit-vol residual (std = 1) scores 0.5. Tighter predicted
        error raises confidence; ``std = 0`` is 1.0, ``std = 3`` is 0.25.
        """
        return 1.0 / (1.0 + residual_std.clamp(min=0.0))

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prediction at the final bar of each sequence: ``([B], [B])``."""
        was_training = self.training
        self.eval()
        try:
            mean, log_sigma = self(features)
            return mean[:, -1], log_sigma[:, -1]
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_with_confidence(
        self,
        features: torch.Tensor,
        *,
        mc_samples: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Final-bar mean, log-sigma, and confidence in ``(0, 1]``.

        ``mc_samples > 1`` turns input dropout on and uses the spread of
        sampled means as the residual std (for checkpoints with no trained
        sigma head). Otherwise confidence comes from ``exp(log_sigma)``.
        """
        was_training = self.training
        try:
            if mc_samples > 1:
                self.train()
                samples: list[torch.Tensor] = []
                for _ in range(mc_samples):
                    mean, _log_sigma = self(features)
                    samples.append(mean[:, -1].float())
                stacked = torch.stack(samples, dim=0)
                mean_hat = stacked.mean(dim=0)
                spread = stacked.std(dim=0, unbiased=False)
                log_sigma_hat = torch.log(spread.clamp(min=1e-6))
                confidence = self.confidence_from_std(spread)
                return mean_hat, log_sigma_hat, confidence

            self.eval()
            mean, log_sigma = self(features)
            mean_hat = mean[:, -1]
            log_sigma_hat = log_sigma[:, -1]
            confidence = self.confidence_from_std(torch.exp(log_sigma_hat.float()))
            return mean_hat, log_sigma_hat, confidence
        finally:
            self.train(was_training)

    def collect_dynamic_diagnostics(self) -> list[dict[str, Any]]:
        """Per-layer Dynamic A scale stats from the most recent forward pass."""
        return collect_mixer_diagnostics(self.layers)
