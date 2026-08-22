"""Gradient flow into Mamba parameters and the Dynamic A controller."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mamba_lm.mamba import MambaBlock
from mamba_lm.model import MambaLM
from tests.helpers import dynamic_config, perturb_controller, tiny_config


def test_block_gradients_reach_controller_and_A_log():
    cfg = dynamic_config()
    block = MambaBlock(cfg)
    perturb_controller(block)
    x = torch.randn(2, 6, cfg.d_model, requires_grad=True)
    y = block(x)
    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert block.A_log.grad is not None
    assert torch.isfinite(block.A_log.grad).all()
    assert block.controller.out_proj.weight.grad is not None
    assert block.controller.out_proj.bias.grad is not None
    assert block.controller.in_proj.weight.grad is not None
    assert not torch.allclose(
        block.controller.out_proj.weight.grad, torch.zeros_like(block.controller.out_proj.weight.grad)
    )


def test_lm_backward_no_nan():
    cfg = dynamic_config(vocab_size=16)
    model = MambaLM(cfg)
    perturb_controller(model.layers[0].mixer)
    ids = torch.randint(0, cfg.vocab_size, (3, 10))
    targets = torch.randint(0, cfg.vocab_size, (3, 10))
    logits = model(ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"

    # Controller and A_log in every layer should receive gradients.
    for layer in model.layers:
        mixer = layer.mixer
        assert mixer.A_log.grad is not None
        assert mixer.controller.out_proj.weight.grad is not None


def test_baseline_A_log_gets_grad():
    cfg = tiny_config(dynamic_weights=False)
    block = MambaBlock(cfg)
    x = torch.randn(2, 5, cfg.d_model, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None
    assert block.A_log.grad is not None
