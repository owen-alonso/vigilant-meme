"""Selective SSM scan shared by baseline and Dynamic A modes.

The recurrence is the standard diagonal S6 step:

    h_t = A_bar_t * h_{t-1} + deltaB_u_t
    y_t = (h_t * C_t).sum(N) + D * u_t

A_bar is always shaped [B, L, D, N] so a static A and a token-dependent A_t
share one scan implementation.

The default scan is a chunked associative scan. Hillis–Steele over the full
length clones [B, L, D, N] log2(L) times and dominates training time; scanning
short chunks serially (vectorized over batch) then combining chunk carries is
mathematically the same prefix and much cheaper on GPU.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


ScanAlgorithm = Literal["parallel", "sequential"]

# Chosen from GPU fwd+bwd timings on [B=8, L=256, D=256, N=16] (RTX 4070).
_SCAN_CHUNK = 8


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
        dt_A = dt.unsqueeze(-1) * A
    elif A.dim() == 4:
        dt_A = dt.unsqueeze(-1) * A
    else:
        raise ValueError(f"A must be [D, N] or [B, L, D, N], got {tuple(A.shape)}")

    A_bar = torch.exp(dt_A)
    deltaB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    return A_bar, deltaB_u


def linear_recurrence_sequential(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + b_t with h_{-1} = 0. Sequential over L.

    a, b, h: [B, L, D, N]
    """
    return _serial_scan(a, b)


def _serial_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + b_t along dim 1. a, b, h: [B, T, D, N]."""
    batch, seq_len, d_inner, d_state = a.shape
    h = a.new_zeros(batch, d_inner, d_state)
    outputs = []
    for t in range(seq_len):
        h = a[:, t] * h + b[:, t]
        outputs.append(h)
    return torch.stack(outputs, dim=1)


def _associative_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hillis–Steele prefix scan along dim 1.

    Cheap when T is small (chunk carries). The same algorithm over full L
    clones [B, L, D, N] log2(L) times and is the training bottleneck we
    replaced; do not use it as the default full-length scan.
    """
    seq_len = a.shape[1]
    h_a, h_b = a, b
    k = 1
    while k < seq_len:
        a_next = h_a.clone()
        b_next = h_b.clone()
        a_next[:, k:] = h_a[:, k:] * h_a[:, :-k]
        b_next[:, k:] = h_a[:, k:] * h_b[:, :-k] + h_b[:, k:]
        h_a, h_b = a_next, b_next
        k *= 2
    return h_b


def linear_recurrence_parallel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Chunked associative scan for h_t = a_t * h_{t-1} + b_t.

    Pair operator: (a1, b1) ⊕ (a2, b2) = (a2 * a1, a2 * b1 + b2)

    Short chunks are scanned serially (vectorized over batch and chunk).
    Chunk carries are combined with Hillis–Steele on the much smaller
    [B, n_chunks, D, N] tensors, then injected so the result matches a
    full-length prefix scan without log2(L) clones of [B, L, D, N].
    """
    batch, seq_len, d_inner, d_state = a.shape
    chunk = _SCAN_CHUNK
    if seq_len <= chunk:
        return _serial_scan(a, b)

    pad = (chunk - seq_len % chunk) % chunk
    if pad:
        a = F.pad(a, (0, 0, 0, 0, 0, pad), value=1.0)
        b = F.pad(b, (0, 0, 0, 0, 0, pad), value=0.0)
    padded = a.shape[1]
    n_chunks = padded // chunk
    a = a.reshape(batch, n_chunks, chunk, d_inner, d_state)
    b = b.reshape(batch, n_chunks, chunk, d_inner, d_state)

    h = a.new_zeros(batch, n_chunks, d_inner, d_state)
    acc_a = a.new_ones(batch, n_chunks, d_inner, d_state)
    hs: list[torch.Tensor] = []
    prefs: list[torch.Tensor] = []
    for t in range(chunk):
        acc_a = acc_a * a[:, :, t]
        h = a[:, :, t] * h + b[:, :, t]
        hs.append(h)
        prefs.append(acc_a)
    h_local = torch.stack(hs, dim=2)
    a_pref = torch.stack(prefs, dim=2)

    # State after each chunk, then incoming carry = previous chunk's state.
    h_chunks = _associative_scan(a_pref[:, :, -1], h_local[:, :, -1])
    zeros = a.new_zeros(batch, 1, d_inner, d_state)
    hin = torch.cat([zeros, h_chunks[:, :-1]], dim=1).unsqueeze(2)
    out = a_pref * hin + h_local
    return out.reshape(batch, padded, d_inner, d_state)[:, :seq_len]


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
        algorithm: "parallel" (default, chunked) or "sequential" (reference)

    Returns:
        y: [B, L, D]
    """
    if algorithm == "parallel":
        h = linear_recurrence_parallel(A_bar, deltaB_u)
    elif algorithm == "sequential":
        h = linear_recurrence_sequential(A_bar, deltaB_u)
    else:
        raise ValueError(f"Unknown scan algorithm: {algorithm!r}")

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
