"""Tests for shared training utilities."""

from __future__ import annotations

import pytest
import torch

from mamba_lm.training_utils import (
    autocast_context,
    grads_finite,
    lr_linear_warmup,
    lr_warmup_cosine,
)


def test_lr_linear_warmup_ramps_then_plateaus():
    assert lr_linear_warmup(0, lr=1.0, warmup_steps=4) == 0.25
    assert lr_linear_warmup(3, lr=1.0, warmup_steps=4) == 1.0
    assert lr_linear_warmup(10, lr=1.0, warmup_steps=4) == 1.0


def test_lr_warmup_cosine_decays():
    warmup_end = lr_warmup_cosine(9, total_steps=100, lr=1.0, warmup_frac=0.1)
    end = lr_warmup_cosine(99, total_steps=100, lr=1.0, warmup_frac=0.1)
    assert warmup_end == pytest.approx(1.0, rel=1e-3)
    assert end == pytest.approx(0.1, rel=1e-2)
    assert end < warmup_end


def test_autocast_context_cpu_fp16_falls_back():
    device = torch.device("cpu")
    with autocast_context(device, "fp16") as ctx:
        assert ctx is not None


def test_grads_finite_detects_nan():
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.tensor([float("nan"), 1.0])
    model = torch.nn.Linear(2, 1)
    model.weight = param
    assert not grads_finite(model, unique=False)
