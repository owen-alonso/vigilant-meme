"""Tests for shared training utilities."""

from __future__ import annotations

import pytest
import torch

from mamba_lm.training_utils import (
    _ES_CONTINUOUS,
    _KEEP_AWAKE_FLAGS,
    autocast_context,
    grads_finite,
    keep_awake,
    lr_linear_warmup,
    lr_warmup_cosine,
)


def test_lr_linear_warmup_ramps_then_plateaus():
    assert lr_linear_warmup(0, lr=1.0, warmup_steps=4) == 0.25
    assert lr_linear_warmup(3, lr=1.0, warmup_steps=4) == 1.0
    assert lr_linear_warmup(10, lr=1.0, warmup_steps=4) == 1.0


def test_lr_warmup_cosine_decays():
    warmup_end = lr_warmup_cosine(9, total_steps=100, lr=1.0, warmup_frac=0.1)
    end = lr_warmup_cosine(99, total_steps=100, lr=1.0, warmup_frac=0.1)
    assert warmup_end == pytest.approx(1.0, rel=1e-3)
    assert end == pytest.approx(0.1, rel=1e-2)
    assert end < warmup_end


def test_autocast_context_cpu_fp16_falls_back():
    device = torch.device("cpu")
    with autocast_context(device, "fp16") as ctx:
        assert ctx is not None


def test_keep_awake_inhibits_sleep_then_restores(monkeypatch):
    calls: list[int] = []

    def fake_set(flags: int) -> bool:
        calls.append(flags)
        return True

    monkeypatch.setattr("mamba_lm.training_utils.sys.platform", "win32")
    monkeypatch.setattr(
        "mamba_lm.training_utils._set_windows_execution_state", fake_set
    )
    logs: list[str] = []
    with keep_awake(log_fn=logs.append, interval_sec=60.0):
        assert calls == [_KEEP_AWAKE_FLAGS]
        assert any("sleep and display inhibited" in line for line in logs)
    assert calls[-1] == _ES_CONTINUOUS
    assert any("sleep and display restored" in line for line in logs)


def test_keep_awake_clears_after_error(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr("mamba_lm.training_utils.sys.platform", "win32")
    monkeypatch.setattr(
        "mamba_lm.training_utils._set_windows_execution_state",
        lambda flags: calls.append(flags) or True,
    )
    with pytest.raises(RuntimeError, match="boom"):
        with keep_awake(interval_sec=60.0):
            raise RuntimeError("boom")
    assert calls[-1] == _ES_CONTINUOUS


def test_keep_awake_is_noop_off_windows(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr("mamba_lm.training_utils.sys.platform", "linux")
    monkeypatch.setattr(
        "mamba_lm.training_utils._set_windows_execution_state",
        lambda flags: calls.append(flags) or True,
    )
    with keep_awake():
        pass
    assert calls == []


def test_grads_finite_detects_nan():
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.tensor([float("nan"), 1.0])
    model = torch.nn.Linear(2, 1)
    model.weight = param
    assert not grads_finite(model, unique=False)
