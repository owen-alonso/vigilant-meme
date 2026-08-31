"""Baseline path must match Dynamic A at zero controller contribution."""

from __future__ import annotations

import torch

from mamba_lm.mamba import MambaBlock
from mamba_lm.model import MambaLM
from tests.helpers import block_scale, dynamic_config, tiny_config


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
    scale = block_scale(dynamic)
    assert torch.allclose(scale, torch.ones_like(scale))


def test_zero_strength_matches_baseline():
    torch.manual_seed(2)
    base_cfg = tiny_config(dynamic_weights=False)
    dyn_cfg = dynamic_config(dynamic_strength=0.0)
    baseline = MambaBlock(base_cfg)
    dynamic = MambaBlock(dyn_cfg)
    dynamic.load_state_dict(baseline.state_dict(), strict=False)
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


def test_same_seed_constructs_matching_baseline_weights():
    """Controller init must not steal RNG from later Mamba layers."""
    torch.manual_seed(11)
    base = MambaLM(tiny_config(dynamic_weights=False, n_layer=3))
    torch.manual_seed(11)
    dyn = MambaLM(dynamic_config(n_layer=3))
    base_sd = base.state_dict()
    dyn_sd = dyn.state_dict()
    shared = [k for k in base_sd if k in dyn_sd]
    assert shared
    for key in shared:
        assert torch.equal(base_sd[key], dyn_sd[key]), key
