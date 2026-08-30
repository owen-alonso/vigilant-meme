"""Tied-weight grad clip and unique parameter counting."""

from __future__ import annotations

import torch

from mamba_lm.model import MambaLM
from mamba_lm.reporting import clip_grad_norm_unique, unique_parameters
from mamba_lm.rmsnorm import RMSNorm
from tests.helpers import tiny_config


def test_tied_embedding_is_deduped():
    model = MambaLM(tiny_config())
    emb = model.embedding.weight
    assert emb is model.lm_head.weight
    # Default Module.parameters() already drops duplicates; the trap is
    # named_parameters(remove_duplicate=False) or a raw list of both names.
    listed_twice = sum(
        1 for _, p in model.named_parameters(remove_duplicate=False) if p is emb
    )
    uniq = sum(1 for p in unique_parameters(model) if p is emb)
    assert listed_twice == 2
    assert uniq == 1


def test_clip_does_not_scale_tied_weight_twice():
    torch.manual_seed(0)
    model = MambaLM(tiny_config())
    ids = torch.randint(0, 16, (2, 8))
    model(ids).sum().backward()
    unique = clip_grad_norm_unique(model, 1.0)

    model.zero_grad(set_to_none=True)
    model(ids).sum().backward()
    dup = [p for _, p in model.named_parameters(remove_duplicate=False)]
    naive = float(torch.nn.utils.clip_grad_norm_(dup, 1.0))
    # Duplicate embedding inflates the naive norm.
    assert naive >= unique - 1e-6


def test_rmsnorm_finite_under_autocast():
    norm = RMSNorm(16)
    x = torch.randn(2, 8, 16) * 40
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = norm(x)
    assert y.dtype == x.dtype
    assert torch.isfinite(y.float()).all()
