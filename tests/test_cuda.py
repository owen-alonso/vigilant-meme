"""CUDA path: skipped when no GPU is present."""

from __future__ import annotations

import pytest
import torch

from mamba_lm.mamba import MambaBlock
from tests.helpers import dynamic_config, perturb_controller, tiny_config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@cuda
def test_cuda_baseline_and_dynamic():
    device = torch.device("cuda")
    x = torch.randn(2, 8, 32, device=device)

    base = MambaBlock(tiny_config(dynamic_weights=False)).to(device)
    y = base(x)
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()

    dyn = MambaBlock(dynamic_config()).to(device)
    perturb_controller(dyn)
    y2 = dyn(x)
    assert y2.device.type == "cuda"
    assert torch.isfinite(y2).all()
    y2.sum().backward()
    assert dyn.controller.out_proj.weight.grad is not None
