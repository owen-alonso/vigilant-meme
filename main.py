#!/usr/bin/env python3
"""CLI: train, eval, benchmark, compare."""

from __future__ import annotations

import argparse
import json

from mamba_lm.benchmark import format_benchmark, run_comparison
from mamba_lm.checkpoint import model_from_checkpoint
from mamba_lm.config import MambaConfig, TrainConfig
from mamba_lm.experiment import run_training_comparison
from mamba_lm.model import MambaLM
from mamba_lm.reporting import format_parameter_report, parameter_report
from mamba_lm.train import evaluate, select_device, train
from mamba_lm.data import build_datasets
from torch.utils.data import DataLoader


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--dynamic-strength", type=float, default=0.1)
    parser.add_argument("--dynamic-controller-dim", type=int, default=None)


def _model_config_from_args(args: argparse.Namespace) -> MambaConfig:
    return MambaConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_state=args.d_state,
        expand=args.expand,
        dynamic_weights=bool(args.dynamic_weights),
        dynamic_A=True,
        dynamic_strength=args.dynamic_strength,
        dynamic_controller_dim=args.dynamic_controller_dim,
    )


def cmd_train(args: argparse.Namespace) -> None:
    model_cfg = _model_config_from_args(args)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        lr=args.lr,
        precision=args.precision,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
    )
    result = train(model_cfg, train_cfg)
    summary = {
        k: result[k]
        for k in (
            "final_train_loss",
            "final_val_loss",
            "tokens_per_sec",
            "parameter_report",
            "device",
        )
    }
    print(json.dumps(summary, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    device = select_device()
    model, ckpt = model_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()
    extra = ckpt.get("extra") or {}
    seq_len = extra.get("train_config", {}).get("seq_len", args.seq_len)
    data_dir = extra.get("train_config", {}).get("data_dir", args.data_dir)
    _, _, val_ds = build_datasets(data_dir, seq_len)
    loader = DataLoader(val_ds, batch_size=args.batch_size)
    metrics = evaluate(model, loader, device, "fp32", args.eval_batches)
    print(json.dumps(metrics, indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    baseline, dynamic = run_comparison(
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_state=args.d_state,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup=args.warmup,
        steps=args.steps,
    )
    print(format_benchmark(baseline))
    print()
    print(format_benchmark(dynamic))


def cmd_compare(args: argparse.Namespace) -> None:
    model_cfg = _model_config_from_args(args)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        lr=args.lr,
        precision=args.precision,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        eval_interval=max(1, args.max_steps // 2),
    )
    run_training_comparison(model_cfg, train_cfg)


def cmd_report(args: argparse.Namespace) -> None:
    cfg = _model_config_from_args(args)
    cfg.vocab_size = args.vocab_size
    model = MambaLM(cfg)
    print(format_parameter_report(parameter_report(model)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic A Mamba language model")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train baseline or Dynamic A Mamba")
    _add_model_args(train_p)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--seq-len", type=int, default=256)
    train_p.add_argument("--max-steps", type=int, default=200)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--checkpoint-dir", default="checkpoints")
    train_p.add_argument("--data-dir", default="data")
    train_p.set_defaults(func=cmd_train)

    eval_p = sub.add_parser("eval", help="Evaluate a checkpoint")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--batch-size", type=int, default=8)
    eval_p.add_argument("--seq-len", type=int, default=256)
    eval_p.add_argument("--eval-batches", type=int, default=20)
    eval_p.add_argument("--data-dir", default="data")
    eval_p.set_defaults(func=cmd_eval)

    bench_p = sub.add_parser("benchmark", help="Compare baseline vs Dynamic A speed")
    _add_model_args(bench_p)
    bench_p.add_argument("--batch-size", type=int, default=4)
    bench_p.add_argument("--seq-len", type=int, default=256)
    bench_p.add_argument("--warmup", type=int, default=5)
    bench_p.add_argument("--steps", type=int, default=20)
    bench_p.set_defaults(func=cmd_benchmark)

    cmp_p = sub.add_parser("compare", help="Train baseline and Dynamic A with identical hparams")
    _add_model_args(cmp_p)
    cmp_p.add_argument("--batch-size", type=int, default=4)
    cmp_p.add_argument("--seq-len", type=int, default=128)
    cmp_p.add_argument("--max-steps", type=int, default=50)
    cmp_p.add_argument("--lr", type=float, default=3e-4)
    cmp_p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    cmp_p.add_argument("--seed", type=int, default=42)
    cmp_p.add_argument("--checkpoint-dir", default="checkpoints")
    cmp_p.add_argument("--data-dir", default="data")
    cmp_p.set_defaults(func=cmd_compare)

    report_p = sub.add_parser("report", help="Print parameter counts")
    _add_model_args(report_p)
    report_p.add_argument("--vocab-size", type=int, default=65)
    report_p.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
