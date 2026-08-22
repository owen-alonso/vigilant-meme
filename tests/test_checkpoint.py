"""Checkpoint round-trip and baseline -> dynamic compatibility."""

from __future__ import annotations

from pathlib import Path

import torch

from mamba_lm.checkpoint import load_checkpoint, save_checkpoint
from mamba_lm.model import MambaLM
from tests.helpers import dynamic_config, tiny_config


def test_round_trip_dynamic(tmp_path: Path):
    torch.manual_seed(0)
    cfg = dynamic_config()
    model = MambaLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 7))
    with torch.no_grad():
        before = model(ids)

    path = tmp_path / "dyn.pt"
    save_checkpoint(path, model=model, step=3)
    loaded = MambaLM(dynamic_config())
    ckpt = load_checkpoint(path, model=loaded, strict=True)
    assert ckpt["step"] == 3
    assert ckpt["parsed_config"].dynamic_weights is True
    loaded.eval()
    with torch.no_grad():
        after = loaded(ids)
    assert torch.allclose(before, after, rtol=1e-6, atol=1e-6)


def test_baseline_checkpoint_loads_into_dynamic(tmp_path: Path):
    torch.manual_seed(4)
    baseline = MambaLM(tiny_config(dynamic_weights=False))
    path = tmp_path / "base.pt"
    save_checkpoint(path, model=baseline, step=1)

    dynamic = MambaLM(dynamic_config())
    ckpt = load_checkpoint(path, model=dynamic, strict=None)
    missing = ckpt["incompatible_keys"].missing_keys
    assert any("controller" in k for k in missing)

    ids = torch.randint(0, 16, (2, 6))
    with torch.no_grad():
        y_base = baseline(ids)
        y_dyn = dynamic(ids)
    # Controller remains zero-init, so outputs stay aligned.
    assert torch.allclose(y_base, y_dyn, rtol=1e-4, atol=1e-4)


def test_config_preserved(tmp_path: Path):
    cfg = dynamic_config(dynamic_strength=0.25, dynamic_controller_dim=48)
    model = MambaLM(cfg)
    path = tmp_path / "cfg.pt"
    save_checkpoint(path, model=model)
    ckpt = load_checkpoint(path)
    parsed = ckpt["parsed_config"]
    assert parsed.dynamic_weights is True
    assert parsed.dynamic_A is True
    assert parsed.dynamic_strength == 0.25
    assert parsed.dynamic_controller_dim == 48
