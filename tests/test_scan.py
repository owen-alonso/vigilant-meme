"""Scan recurrence: parallel vs sequential, and discretization shapes."""

from __future__ import annotations

import torch

from mamba_lm.scan import (
    discretize,
    linear_recurrence_parallel,
    linear_recurrence_sequential,
    selective_scan,
)


def test_parallel_matches_sequential():
    torch.manual_seed(0)
    a = torch.rand(2, 7, 4, 3) * 0.5 + 0.4  # (0.4, 0.9) stable
    b = torch.randn(2, 7, 4, 3)
    h_seq = linear_recurrence_sequential(a, b)
    h_par = linear_recurrence_parallel(a, b)
    assert torch.allclose(h_seq, h_par, rtol=1e-5, atol=1e-5)


def test_discretize_static_and_dynamic_shapes():
    B, L, D, N = 2, 5, 6, 4
    u = torch.randn(B, L, D)
    dt = torch.rand(B, L, D) * 0.05 + 0.01
    B_in = torch.randn(B, L, N)
    A_static = -torch.rand(D, N)
    A_bar, deltaB_u = discretize(u, dt, A_static, B_in)
    assert A_bar.shape == (B, L, D, N)
    assert deltaB_u.shape == (B, L, D, N)

    A_dyn = A_static.unsqueeze(0).unsqueeze(0).expand(B, L, D, N).contiguous()
    A_bar2, deltaB_u2 = discretize(u, dt, A_dyn, B_in)
    assert torch.allclose(A_bar, A_bar2)
    assert torch.allclose(deltaB_u, deltaB_u2)


def test_selective_scan_output_shape():
    B, L, D, N = 2, 5, 6, 4
    u = torch.randn(B, L, D)
    A_bar = torch.rand(B, L, D, N) * 0.5 + 0.3
    deltaB_u = torch.randn(B, L, D, N)
    C = torch.randn(B, L, N)
    D_skip = torch.ones(D)
    y = selective_scan(u, A_bar, deltaB_u, C, D_skip, algorithm="parallel")
    assert y.shape == (B, L, D)
    y2 = selective_scan(u, A_bar, deltaB_u, C, D_skip, algorithm="sequential")
    assert torch.allclose(y, y2, rtol=1e-5, atol=1e-5)
