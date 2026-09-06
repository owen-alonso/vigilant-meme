"""Fused Triton scan: forward/backward parity with the PyTorch reference."""

from __future__ import annotations

import pytest
import torch

from mamba_lm.scan import discretize, selective_scan

if not torch.cuda.is_available():
    pytest.skip("fused scan requires CUDA", allow_module_level=True)

from mamba_lm.scan_fused import fused_scan_available, fused_ssm_forward

if not fused_scan_available():
    pytest.skip("Triton not available", allow_module_level=True)


def _reference(u, dt, A, B, C, D):
    A_bar, deltaB_u = discretize(u, dt, A, B)
    return selective_scan(u, A_bar, deltaB_u, C, D, algorithm="sequential")


def _make_inputs(batch, seq_len, d_inner, d_state, *, dynamic, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    u = torch.randn(batch, seq_len, d_inner, device=dev)
    dt = torch.rand(batch, seq_len, d_inner, device=dev) * 0.1 + 1e-3
    A = -torch.rand(d_inner, d_state, device=dev) - 0.5
    if dynamic:
        A = A.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1).contiguous()
        A = A * (0.8 + 0.4 * torch.rand_like(A))
    B = torch.randn(batch, seq_len, d_state, device=dev)
    C = torch.randn(batch, seq_len, d_state, device=dev)
    D = torch.randn(d_inner, device=dev)
    return u, dt, A, B, C, D


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 7, 4, 3),      # tiny, odd everything
        (2, 64, 48, 16),   # mid, non-pow2 D
        (3, 130, 256, 16), # odd L, training-like D/N
    ],
)
def test_fused_forward_matches_reference(shape, dynamic):
    u, dt, A, B, C, D = _make_inputs(*shape, dynamic=dynamic)
    with torch.no_grad():
        y_ref = _reference(u, dt, A, B, C, D)
        y_fused = fused_ssm_forward(u, dt, A, B, C, D)
    assert y_fused.shape == y_ref.shape
    torch.testing.assert_close(y_fused, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("shape", [(2, 33, 24, 8), (2, 64, 48, 16)])
def test_fused_backward_matches_reference(shape, dynamic):
    inputs_ref = _make_inputs(*shape, dynamic=dynamic, seed=1)
    inputs_fused = tuple(t.detach().clone() for t in inputs_ref)
    for t in (*inputs_ref, *inputs_fused):
        t.requires_grad_(True)

    torch.manual_seed(2)
    dy = torch.randn(
        inputs_ref[0].shape, device=inputs_ref[0].device
    )
    _reference(*inputs_ref).backward(dy)
    fused_ssm_forward(*inputs_fused).backward(dy)

    names = ["u", "dt", "A", "B", "C", "D"]
    for name, ref, fused in zip(names, inputs_ref, inputs_fused):
        assert fused.grad is not None, name
        torch.testing.assert_close(
            fused.grad, ref.grad, rtol=2e-3, atol=2e-4,
            msg=lambda m, n=name: f"grad mismatch for {n}: {m}",
        )


def test_fused_no_grad_skips_h_buffer():
    u, dt, A, B, C, D = _make_inputs(2, 32, 16, 8, dynamic=False)
    with torch.no_grad():
        y = fused_ssm_forward(u, dt, A, B, C, D)
    assert y.requires_grad is False
