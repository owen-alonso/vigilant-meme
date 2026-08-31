"""Shape tests for the Mamba block, LM, and Dynamic A tensors."""

from __future__ import annotations

import pytest
import torch

from mamba_lm.mamba import MambaBlock
from mamba_lm.model import MambaLM
from tests.helpers import (
    block_delta_a,
    block_scale,
    dynamic_config,
    perturb_controller,
    tiny_config,
)


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("seq_len", [1, 8, 17])
def test_block_preserves_shape(batch, seq_len):
    cfg = tiny_config()
    block = MambaBlock(cfg)
    x = torch.randn(batch, seq_len, cfg.d_model)
    y = block(x)
    assert y.shape == (batch, seq_len, cfg.d_model)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("seq_len", [4, 16])
def test_dynamic_block_shapes(batch, seq_len):
    cfg = dynamic_config()
    block = MambaBlock(cfg)
    perturb_controller(block)
    x = torch.randn(batch, seq_len, cfg.d_model)
    y = block(x)
    assert y.shape == (batch, seq_len, cfg.d_model)
    assert block_delta_a(block) is not None
    scale = block_scale(block)
    assert block_delta_a(block).shape == (batch, seq_len, cfg.d_state)
    assert scale.shape == (batch, seq_len, cfg.d_state)


def test_lm_logits_shape():
    cfg = tiny_config(vocab_size=20)
    model = MambaLM(cfg)
    B, L = 2, 9
    ids = torch.randint(0, cfg.vocab_size, (B, L))
    logits = model(ids)
    assert logits.shape == (B, L, model.padded_vocab)


def test_baseline_has_no_controller():
    block = MambaBlock(tiny_config(dynamic_weights=False))
    assert block.controller is None
    assert block.modulator is None
    x = torch.randn(2, 5, 32)
    block(x)
    assert block_delta_a(block) is None
    from mamba_lm.diagnostics import get_ssm_diagnostics

    assert get_ssm_diagnostics(block) is None
