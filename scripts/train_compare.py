"""Thin wrapper: identical-hparams baseline vs Dynamic A training run."""

from __future__ import annotations

from mamba_lm.experiment import run_training_comparison


def main() -> None:
    run_training_comparison()


if __name__ == "__main__":
    main()
