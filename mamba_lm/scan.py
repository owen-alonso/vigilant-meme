"""Selective SSM scan shared by baseline and Dynamic A modes.

The recurrence is the standard diagonal S6 step:

    h_t = A_bar_t * h_{t-1} + deltaB_u_t
    y_t = (h_t * C_t).sum(N) + D * u_t

A_bar is always shaped [B, L, D, N] so a static A and a token-dependent A_t
share one scan implementation.
"""

from __future__ import annotations

from typing import Literal

import torch


ScanAlgorithm = Literal["parallel", "sequential"]


def discretize(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-order-hold discretization of the diagonal SSM.

    Args:
        u:  SSM input, shape [B, L, D]
        dt: step size, shape [B, L, D]
        A:  continuous-time state matrix, [D, N] (static) or [B, L, D, N] (dynamic)
        B:  input-dependent B, shape [B, L, N]

    Returns:
        A_bar:    [B, L, D, N] = exp(dt * A)
        deltaB_u: [B, L, D, N] = dt * B * u
    """
    if A.dim() == 2:
        # A: [D, N] broadcasts against dt: [B, L, D, 1] -> [B, L, D, N]
        dt_A = dt.unsqueeze(-1) * A
    elif A.dim() == 4:
        # A already token-dependent: [B, L, D, N]
        dt_A = dt.unsqueeze(-1) * A
    else:
        raise ValueError(f"A must be [D, N] or [B, L, D, N], got {tuple(A.shape)}")

    A_bar = torch.exp(dt_A)
    # dt [B,L,D,1] * B [B,L,1,N] * u [B,L,D,1] -> [B, L, D, N]
    deltaB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    return A_bar, deltaB_u


def linear_recurrence_sequential(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + b_t with h_{-1} = 0. Sequential over L.

    a, b, h: [B, L, D, N]
    """
    batch, seq_len, d_inner, d_state = a.shape
    h_t = torch.zeros(batch, d_inner, d_state, dtype=a.dtype, device=a.device)
    outputs = []
    for t in range(seq_len):
        h_t = a[:, t] * h_t + b[:, t]
        outputs.append(h_t)
    return torch.stack(outputs, dim=1)


def linear_recurrence_parallel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized Hillis–Steele prefix scan for h_t = a_t * h_{t-1} + b_t.

    Pair operator: (a1, b1) ⊕ (a2, b2) = (a2 * a1, a2 * b1 + b2)

    a, b, h: [B, L, D, N]
    """
    seq_len = a.shape[1]
    h_a = a
    h_b = b
    k = 1
    while k < seq_len:
        a_next = h_a.clone()
        b_next = h_b.clone()
        # Combine the length-k prefix ending at t-k into the prefix ending at t.
        a_next[:, k:] = h_a[:, k:] * h_a[:, :-k]
        b_next[:, k:] = h_a[:, k:] * h_b[:, :-k] + h_b[:, k:]
        h_a, h_b = a_next, b_next
        k *= 2
    return h_b


def selective_scan(
    u: torch.Tensor,
    A_bar: torch.Tensor,
    deltaB_u: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    *,
    algorithm: ScanAlgorithm = "parallel",
) -> torch.Tensor:
    """Run the selective SSM scan.

    Args:
        u:        [B, L, D]
        A_bar:    [B, L, D, N]
        deltaB_u: [B, L, D, N]
        C:        [B, L, N]
        D:        [D]
        algorithm: "parallel" (default) or "sequential" (reference)

    Returns:
        y: [B, L, D]
    """
    if algorithm == "parallel":
        h = linear_recurrence_parallel(A_bar, deltaB_u)
    elif algorithm == "sequential":
        h = linear_recurrence_sequential(A_bar, deltaB_u)
    else:
        raise ValueError(f"Unknown scan algorithm: {algorithm!r}")

    # y_t = C_t · h_t  (sum over N) + D * u_t
    y = (h * C.unsqueeze(2)).sum(dim=-1)
    y = y + u * D
    return y


def ssm_forward(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    *,
    algorithm: ScanAlgorithm = "parallel",
) -> torch.Tensor:
    """Discretize then scan. A may be static [D, N] or dynamic [B, L, D, N]."""
    A_bar, deltaB_u = discretize(u, dt, A, B)
    return selective_scan(u, A_bar, deltaB_u, C, D, algorithm=algorithm)
