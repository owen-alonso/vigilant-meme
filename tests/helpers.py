from __future__ import annotations

import torch

from mamba_lm.config import MambaConfig


def tiny_config(**kwargs) -> MambaConfig:
    defaults = dict(
        d_model=32,
        n_layer=2,
        d_state=8,
        expand=2,
        d_conv=4,
        vocab_size=16,
        dynamic_weights=False,
    )
    defaults.update(kwargs)
    return MambaConfig(**defaults)


def dynamic_config(**kwargs) -> MambaConfig:
    kwargs.setdefault("dynamic_weights", True)
    kwargs.setdefault("dynamic_A", True)
    return tiny_config(**kwargs)


def perturb_controller(block) -> None:
    """Make Dynamic A actually token-dependent (it is zero-init by default)."""
    assert block.controller is not None
    torch.nn.init.normal_(block.controller.out_proj.weight, mean=0.0, std=0.05)
    torch.nn.init.zeros_(block.controller.out_proj.bias)
