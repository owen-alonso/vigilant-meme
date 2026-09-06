"""Microbenchmark for the selective-scan SSM path.

Times ssm_forward (discretize + scan + output) forward and forward+backward
at training-relevant shapes, so scan implementations can be compared.

Usage:
    python scripts/bench_scan.py [--impl pytorch|triton] [--dynamic]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mamba_lm.scan import ssm_forward


def bench(fn, *, warmup: int = 10, steps: int = 50) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.pstdev(times),
    }


def make_inputs(batch, seq_len, d_inner, d_state, *, dynamic: bool, device):
    torch.manual_seed(0)
    u = torch.randn(batch, seq_len, d_inner, device=device)
    dt = torch.rand(batch, seq_len, d_inner, device=device) * 0.1 + 0.001
    A = -torch.rand(d_inner, d_state, device=device) - 0.5
    if dynamic:
        A = A.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1).contiguous()
    B = torch.randn(batch, seq_len, d_state, device=device)
    C = torch.randn(batch, seq_len, d_state, device=device)
    D = torch.ones(d_inner, device=device)
    for t in (u, dt, A, B, C, D):
        t.requires_grad_(True)
    return u, dt, A, B, C, D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", action="store_true", help="A as [B,L,D,N]")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-inner", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda")
    u, dt, A, B, C, D = make_inputs(
        args.batch, args.seq_len, args.d_inner, args.d_state,
        dynamic=args.dynamic, device=device,
    )

    def fwd():
        with torch.no_grad():
            ssm_forward(u, dt, A, B, C, D)

    def fwd_bwd():
        y = ssm_forward(u, dt, A, B, C, D)
        y.sum().backward()
        for t in (u, dt, A, B, C, D):
            t.grad = None

    torch.cuda.reset_peak_memory_stats()
    f = bench(fwd)
    fb = bench(fwd_bwd)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    shape = f"[B={args.batch}, L={args.seq_len}, D={args.d_inner}, N={args.d_state}]"
    print(f"shape {shape} dynamic={args.dynamic}")
    print(f"forward:      {f['mean_ms']:.3f} ms (median {f['median_ms']:.3f}, std {f['std_ms']:.3f})")
    print(f"forward+bwd:  {fb['mean_ms']:.3f} ms (median {fb['median_ms']:.3f}, std {fb['std_ms']:.3f})")
    print(f"peak memory:  {peak:.1f} MiB")


if __name__ == "__main__":
    main()
