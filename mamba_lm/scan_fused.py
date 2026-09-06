"""Fused Triton GPU kernels for the selective SSM scan.

Replaces the PyTorch chunked scan on CUDA. One kernel computes the whole
    a_t = exp(dt_t * A)
    h_t = a_t * h_{t-1} + (dt_t * B_t * u_t)
    y_t = (h_t * C_t).sum(N) + D * u_t
forward in a single launch, keeping h in registers and never materializing
the [B, L, D, N] A_bar / deltaB_u tensors. A matching backward kernel runs
the gradient recurrence g_t = dh_t + a_{t+1} * g_{t+1} in reverse.

Only h (needed by backward) is written to memory, so forward memory traffic
drops from ~5x [B,L,D,N] tensors to 1, and the Python-loop kernel-launch
overhead of the chunked scan disappears.

Supports static A [D, N] and dynamic A [B, L, D, N] (Dynamic A mode).
All math is fp32, matching the existing scan path (mamba.py casts to fp32
before ssm_forward).

Gradients for B and C are accumulated with fp32 atomics across D-blocks, so
they are not bitwise deterministic run-to-run (same as cuBLAS reductions).
"""

from __future__ import annotations

import os

import torch

try:  # Triton ships with CUDA builds of PyTorch; guard for CPU-only installs.
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover - exercised only on CPU-only installs
    triton = None
    tl = None
    _TRITON_OK = False


def fused_scan_available(device: torch.device | None = None) -> bool:
    """True if the fused Triton scan can run (CUDA + Triton + not disabled)."""
    if not _TRITON_OK or not torch.cuda.is_available():
        return False
    if device is not None and device.type != "cuda":
        return False
    return os.environ.get("MAMBA_FUSED_SCAN", "1") != "0"


if _TRITON_OK:

    @triton.jit
    def _ssm_fwd_kernel(
        u_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, d_ptr,
        y_ptr, h_ptr,
        L, D, N,
        IS_DYNAMIC: tl.constexpr,
        SAVE_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        mask_d = offs_d < D
        mask_n = offs_n < N
        mask_dn = mask_d[:, None] & mask_n[None, :]

        d_skip = tl.load(d_ptr + offs_d, mask=mask_d, other=0.0)
        a_mat = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
        if not IS_DYNAMIC:
            a_mat = tl.load(
                a_ptr + offs_d[:, None] * N + offs_n[None, :],
                mask=mask_dn, other=0.0,
            )

        h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
        for t in range(L):
            base = pid_b * L + t
            u_t = tl.load(u_ptr + base * D + offs_d, mask=mask_d, other=0.0)
            dt_t = tl.load(dt_ptr + base * D + offs_d, mask=mask_d, other=0.0)
            b_t = tl.load(b_ptr + base * N + offs_n, mask=mask_n, other=0.0)
            c_t = tl.load(c_ptr + base * N + offs_n, mask=mask_n, other=0.0)
            if IS_DYNAMIC:
                a_mat = tl.load(
                    a_ptr + (base * D + offs_d)[:, None] * N + offs_n[None, :],
                    mask=mask_dn, other=0.0,
                )
            a_bar = tl.exp(dt_t[:, None] * a_mat)
            h = a_bar * h + (dt_t * u_t)[:, None] * b_t[None, :]
            y_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * u_t
            tl.store(y_ptr + base * D + offs_d, y_t, mask=mask_d)
            if SAVE_H:
                tl.store(
                    h_ptr + (base * D + offs_d)[:, None] * N + offs_n[None, :],
                    h, mask=mask_dn,
                )

    @triton.jit
    def _ssm_bwd_kernel(
        u_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, d_ptr, h_ptr, dy_ptr,
        du_ptr, ddt_ptr, da_ptr, db_ptr, dc_ptr, dd_ptr,
        L, D, N,
        IS_DYNAMIC: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        mask_d = offs_d < D
        mask_n = offs_n < N
        mask_dn = mask_d[:, None] & mask_n[None, :]

        d_skip = tl.load(d_ptr + offs_d, mask=mask_d, other=0.0)
        a_mat = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
        if not IS_DYNAMIC:
            a_mat = tl.load(
                a_ptr + offs_d[:, None] * N + offs_n[None, :],
                mask=mask_dn, other=0.0,
            )

        da_acc = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
        dd_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        # carry_t = a_{t+1} * g_{t+1} for the reverse recurrence
        # g_t = dy_t * C_t + carry_t.
        carry = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
        for i in range(L):
            t = L - 1 - i
            base = pid_b * L + t
            dy_t = tl.load(dy_ptr + base * D + offs_d, mask=mask_d, other=0.0)
            u_t = tl.load(u_ptr + base * D + offs_d, mask=mask_d, other=0.0)
            dt_t = tl.load(dt_ptr + base * D + offs_d, mask=mask_d, other=0.0)
            b_t = tl.load(b_ptr + base * N + offs_n, mask=mask_n, other=0.0)
            c_t = tl.load(c_ptr + base * N + offs_n, mask=mask_n, other=0.0)
            if IS_DYNAMIC:
                a_mat = tl.load(
                    a_ptr + (base * D + offs_d)[:, None] * N + offs_n[None, :],
                    mask=mask_dn, other=0.0,
                )
            a_bar = tl.exp(dt_t[:, None] * a_mat)
            h_t = tl.load(
                h_ptr + (base * D + offs_d)[:, None] * N + offs_n[None, :],
                mask=mask_dn, other=0.0,
            )
            h_prev = tl.load(
                h_ptr + ((base - 1) * D + offs_d)[:, None] * N + offs_n[None, :],
                mask=mask_dn & (t > 0), other=0.0,
            )

            g = dy_t[:, None] * c_t[None, :] + carry

            # deltaB_u = dt * B * u  → shared reduction over N.
            s = tl.sum(g * b_t[None, :], axis=1)
            # a_bar = exp(dt * A)  → d a_bar = a_bar * (dt dA + A ddt).
            da_bar = g * h_prev
            ddt_t = tl.sum(da_bar * a_bar * a_mat, axis=1) + s * u_t
            du_t = dy_t * d_skip + s * dt_t
            db_vec = tl.sum(g * (dt_t * u_t)[:, None], axis=0)
            dc_vec = tl.sum(dy_t[:, None] * h_t, axis=0)

            tl.store(du_ptr + base * D + offs_d, du_t, mask=mask_d)
            tl.store(ddt_ptr + base * D + offs_d, ddt_t, mask=mask_d)
            tl.atomic_add(db_ptr + base * N + offs_n, db_vec, mask=mask_n)
            tl.atomic_add(dc_ptr + base * N + offs_n, dc_vec, mask=mask_n)
            if IS_DYNAMIC:
                tl.store(
                    da_ptr + (base * D + offs_d)[:, None] * N + offs_n[None, :],
                    da_bar * a_bar * dt_t[:, None],
                    mask=mask_dn,
                )
            else:
                da_acc += da_bar * a_bar * dt_t[:, None]
            dd_acc += dy_t * u_t
            carry = a_bar * g

        if not IS_DYNAMIC:
            tl.atomic_add(
                da_ptr + offs_d[:, None] * N + offs_n[None, :],
                da_acc, mask=mask_dn,
            )
        tl.atomic_add(dd_ptr + offs_d, dd_acc, mask=mask_d)


def _block_d(d_inner: int) -> int:
    if d_inner <= 16:
        return 16
    if d_inner <= 32:
        return 32
    return 32  # 32 lanes x BLOCK_N keeps register pressure low; grid covers D.


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class _FusedSelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, B, C, D):
        batch, seq_len, d_inner = u.shape
        n = B.shape[-1]
        is_dynamic = A.dim() == 4

        u = u.contiguous()
        dt = dt.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        D = D.contiguous()

        # Inside Function.forward grad mode is disabled, so is_grad_enabled()
        # cannot be used; needs_input_grad reflects grad mode + requires_grad.
        needs_grad = any(ctx.needs_input_grad)
        y = torch.empty_like(u)
        h = (
            torch.empty(batch, seq_len, d_inner, n, device=u.device, dtype=u.dtype)
            if needs_grad
            else u.new_empty(1)
        )

        block_d = _block_d(d_inner)
        block_n = _next_pow2(n)
        grid = (batch, triton.cdiv(d_inner, block_d))
        _ssm_fwd_kernel[grid](
            u, dt, A, B, C, D, y, h,
            seq_len, d_inner, n,
            IS_DYNAMIC=is_dynamic,
            SAVE_H=needs_grad,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
        )
        if needs_grad:
            ctx.save_for_backward(u, dt, A, B, C, D, h)
            ctx.is_dynamic = is_dynamic
        return y

    @staticmethod
    def backward(ctx, dy):
        u, dt, A, B, C, D, h = ctx.saved_tensors
        batch, seq_len, d_inner = u.shape
        n = B.shape[-1]
        dy = dy.contiguous()

        du = torch.empty_like(u)
        ddt = torch.empty_like(dt)
        if ctx.is_dynamic:
            dA = torch.empty_like(A)
        else:
            dA = torch.zeros_like(A)  # accumulated with atomics across batch
        dB = torch.zeros_like(B)  # accumulated with atomics across D-blocks
        dC = torch.zeros_like(C)
        dD = torch.zeros_like(D)

        block_d = _block_d(d_inner)
        block_n = _next_pow2(n)
        grid = (batch, triton.cdiv(d_inner, block_d))
        _ssm_bwd_kernel[grid](
            u, dt, A, B, C, D, h, dy,
            du, ddt, dA, dB, dC, dD,
            seq_len, d_inner, n,
            IS_DYNAMIC=ctx.is_dynamic,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
        )
        return du, ddt, dA, dB, dC, dD


def fused_ssm_forward(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Fused discretize + selective scan + output. Same contract as ssm_forward.

    u, dt: [B, L, D]; A: [D, N] or [B, L, D, N]; B, C: [B, L, N]; D: [D].
    Requires CUDA tensors; fp32 math (inputs are cast if needed).
    """
    if u.dtype != torch.float32:
        u = u.float()
    dt = dt.float()
    A = A.float()
    B = B.float()
    C = C.float()
    D = D.float()
    return _FusedSelectiveScan.apply(u, dt, A, B, C, D)
