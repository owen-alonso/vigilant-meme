"""Changing one batch element must not affect another."""

from __future__ import annotations

import torch

from mamba_lm.mamba import MambaBlock
from tests.helpers import dynamic_config, perturb_controller


def test_batch_independence_of_dynamic_A():
    cfg = dynamic_config()
    block = MambaBlock(cfg)
    perturb_controller(block)
    block.eval()

    x = torch.randn(2, 10, cfg.d_model)
    with torch.no_grad():
        block(x)
        scale_orig = block.last_A_scale.detach().clone()

    x_mut = x.clone()
    x_mut[0] = torch.randn_like(x_mut[0])
    with torch.no_grad():
        block(x_mut)
        scale_mut = block.last_A_scale.detach().clone()

    assert not torch.allclose(scale_orig[0], scale_mut[0])
    assert torch.allclose(scale_orig[1], scale_mut[1], rtol=1e-5, atol=1e-5)
