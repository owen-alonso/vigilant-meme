"""Numerical stability in fp32 and mixed precision."""

from __future__ import annotations

import pytest
import torch

from mamba_lm.mamba import MambaBlock
from mamba_lm.model import MambaLM
from tests.helpers import dynamic_config, perturb_controller, tiny_config


def test_hundreds_of_fp32_forwards_finite():
    cfg = dynamic_config()
    block = MambaBlock(cfg)
    perturb_controller(block)
    block.eval()
    with torch.no_grad():
        for i in range(200):
            x = torch.randn(2, 16, cfg.d_model)
            y = block(x)
            assert torch.isfinite(y).all(), f"non-finite output at iter {i}"
            assert y.abs().max() < 1e6, f"exploding output at iter {i}: {y.abs().max()}"
            assert torch.isfinite(block.last_A_scale).all()
            assert (block.last_A_scale > 0).all()


def test_mixed_precision_forwards_finite():
    cfg = dynamic_config()
    block = MambaBlock(cfg)
    perturb_controller(block)
    block.eval()
    device_type = "cpu"
    with torch.no_grad():
        for i in range(50):
            x = torch.randn(2, 12, cfg.d_model)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                y = block(x)
            assert torch.isfinite(y.float()).all(), f"non-finite AMP output at iter {i}"


def test_lm_logits_finite_under_amp():
    cfg = dynamic_config(vocab_size=16)
    model = MambaLM(cfg)
    perturb_controller(model.layers[0].mixer)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(ids)
    assert torch.isfinite(logits.float()).all()


def test_baseline_fp32_finite():
    block = MambaBlock(tiny_config(dynamic_weights=False))
    x = torch.randn(4, 20, 32)
    y = block(x)
    assert torch.isfinite(y).all()
