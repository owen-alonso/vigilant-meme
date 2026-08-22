"""Baseline path must match Dynamic A at zero controller contribution."""

from __future__ import annotations

import torch

from mamba_lm.mamba import MambaBlock
from mamba_lm.model import MambaLM
from tests.helpers import dynamic_config, tiny_config


def test_zero_init_block_matches_baseline():
    torch.manual_seed(1)
    base_cfg = tiny_config(dynamic_weights=False)
    dyn_cfg = dynamic_config()
    baseline = MambaBlock(base_cfg)
    dynamic = MambaBlock(dyn_cfg)
    dynamic.load_state_dict(baseline.state_dict(), strict=False)

    x = torch.randn(3, 11, base_cfg.d_model)
    with torch.no_grad():
        y_base = baseline(x)
        y_dyn = dynamic(x)
    assert torch.allclose(y_base, y_dyn, rtol=1e-5, atol=1e-5)
    # Zero-init controller => scale identically 1.
    assert dynamic.last_A_scale is not None
    assert torch.allclose(dynamic.last_A_scale, torch.ones_like(dynamic.last_A_scale))


def test_zero_strength_matches_baseline():
    torch.manual_seed(2)
    base_cfg = tiny_config(dynamic_weights=False)
    dyn_cfg = dynamic_config(dynamic_strength=0.0)
    baseline = MambaBlock(base_cfg)
    dynamic = MambaBlock(dyn_cfg)
    dynamic.load_state_dict(baseline.state_dict(), strict=False)
    # Non-zero controller weights still cannot move A when strength is 0.
    torch.nn.init.normal_(dynamic.controller.out_proj.weight, std=0.2)

    x = torch.randn(2, 8, base_cfg.d_model)
    y_base = baseline(x)
    y_dyn = dynamic(x)
    assert torch.allclose(y_base, y_dyn, rtol=1e-5, atol=1e-5)


def test_lm_zero_init_matches_baseline():
    torch.manual_seed(3)
    base = MambaLM(tiny_config(dynamic_weights=False))
    dyn = MambaLM(dynamic_config())
    dyn.load_state_dict(base.state_dict(), strict=False)
    ids = torch.randint(0, 16, (2, 12))
    with torch.no_grad():
        logits_b = base(ids)
        logits_d = dyn(ids)
    assert torch.allclose(logits_b, logits_d, rtol=1e-4, atol=1e-4)
