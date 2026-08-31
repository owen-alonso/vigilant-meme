"""Thin wrapper so `python scripts/benchmark.py` works from the repo root."""

from __future__ import annotations

from mamba_lm.benchmark import format_benchmark, run_comparison


def main() -> None:
    baseline, dynamic = run_comparison()
    print(format_benchmark(baseline))
    print()
    print(format_benchmark(dynamic))


if __name__ == "__main__":
    main()
