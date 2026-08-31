"""Tests for safe checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mamba_lm.checkpoint_io import load_checkpoint_dict, validate_checkpoint_payload
from mamba_lm.model import MambaLM
from tests.helpers import tiny_config


def test_validate_checkpoint_payload_rejects_arbitrary_objects():
    with pytest.raises(ValueError, match="unsupported type"):
        validate_checkpoint_payload({"model": object()})


def test_load_checkpoint_dict_round_trip(tmp_path: Path):
    model = MambaLM(tiny_config())
    path = tmp_path / "safe.pt"
    torch.save({"model": model.state_dict(), "config": model.config.to_dict(), "step": 0}, path)
    payload = load_checkpoint_dict(path)
    assert payload["step"] == 0
    assert "model" in payload
