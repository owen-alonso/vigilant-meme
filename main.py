"""CLI: python main.py train | benchmark | compare | report."""

from __future__ import annotations

import argparse

from mamba_lm.benchmark import format_benchmark, run_comparison
from mamba_lm.config import MambaConfig, TrainConfig
from mamba_lm.experiment import run_training_comparison
from mamba_lm.model import MambaLM
from mamba_lm.reporting import format_parameter_report, parameter_report
from mamba_lm.train import select_device, train


def _model_cfg(args: argparse.Namespace) -> MambaConfig:
    return MambaConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_state=args.d_state,
        expand=args.expand,
        dynamic_weights=args.dynamic_weights,
        dynamic_A=args.dynamic_weights,
        dynamic_strength=args.dynamic_strength,
    )


def _train_cfg(args: argparse.Namespace) -> TrainConfig:
    kwargs = dict(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        lr=args.lr,
        precision=args.precision,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
    )
    return TrainConfig(**{k: v for k, v in kwargs.items() if v is not None})


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--dynamic-weights", action="store_true")
    p.add_argument("--dynamic-strength", type=float, default=0.1)


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cpu", action="store_true")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dynamic A Mamba (V1)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_report = sub.add_parser("report", help="parameter split: Mamba vs Dynamic A")
    _add_model_args(p_report)

    p_bench = sub.add_parser("benchmark", help="forward/backward timing")
    _add_model_args(p_bench)
    p_bench.add_argument("--batch-size", type=int, default=4)
    p_bench.add_argument("--seq-len", type=int, default=256)

    p_compare = sub.add_parser("compare", help="identical-hparams baseline vs Dynamic A")
    _add_model_args(p_compare)
    _add_train_args(p_compare)

    p_train = sub.add_parser("train", help="train one model")
    _add_model_args(p_train)
    _add_train_args(p_train)

    args = parser.parse_args(argv)

    if args.cmd == "report":
        model = MambaLM(_model_cfg(args))
        print(format_parameter_report(parameter_report(model)))
        return

    if args.cmd == "benchmark":
        baseline, dynamic = run_comparison(
            d_model=args.d_model,
            n_layer=args.n_layer,
            d_state=args.d_state,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        print(format_benchmark(baseline))
        print()
        print(format_benchmark(dynamic))
        return

    if args.cmd == "compare":
        run_training_comparison(_model_cfg(args), _train_cfg(args))
        return

    if args.cmd == "train":
        device = select_device(prefer_cuda=not args.cpu)
        train(_model_cfg(args), _train_cfg(args), device=device)
        return


if __name__ == "__main__":
    main()
