"""Dynamic A must be input-dependent and token-dependent."""

from __future__ import annotations

import torch

from mamba_lm.mamba import MambaBlock
from tests.helpers import dynamic_config, perturb_controller


def test_different_inputs_different_A():
    cfg = dynamic_config()
    block = MambaBlock(cfg)
    perturb_controller(block)
    x1 = torch.randn(2, 8, cfg.d_model)
    x2 = torch.randn(2, 8, cfg.d_model)
    block(x1)
    a1 = block.last_A_scale.detach().clone()
    block(x2)
    a2 = block.last_A_scale.detach().clone()
    assert not torch.allclose(a1, a2), "controller collapsed to a static map"


def test_tokens_in_sequence_differ():
    cfg = dynamic_config()
    block = MambaBlock(cfg)
    perturb_controller(block)
    # Distinct tokens along L so the SSM input (and thus A) can vary.
    x = torch.randn(1, 12, cfg.d_model)
    block(x)
    scale = block.last_A_scale  # [1, 12, N]
    assert scale is not None
    # At least one pair of timesteps should differ.
    diffs = (scale[:, 1:] - scale[:, :-1]).abs().sum(dim=-1)
    assert (diffs > 1e-8).any(), "all tokens received identical dynamic A"


def test_delta_A_not_constant_after_perturb():
    cfg = dynamic_config()
    block = MambaBlock(cfg)
    perturb_controller(block)
    x = torch.randn(2, 9, cfg.d_model)
    block(x)
    assert block.last_delta_A is not None
    assert block.last_delta_A.abs().sum() > 0
    assert not block.last_A_scale.requires_grad
    assert not block.last_delta_A.requires_grad
