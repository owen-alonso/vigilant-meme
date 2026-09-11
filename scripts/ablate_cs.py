"""Ablate skip-only / CS ridge / residual / tiny encoder on the locked CS protocol.

Default: synthetic CS-momentum universe (CPU, seconds). On Owen's GPU box, point
``--data-dir`` at the liquid daily cache and add ``--universe liquid``.

    python -m scripts.ablate_cs
    python scripts/ablate_cs.py --data-dir data --universe liquid --skip-only-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from forecast.config import (
    DataConfig,
    ForecastModelConfig,
    ForecastTrainConfig,
    interval_data_kwargs,
    interval_model_kwargs,
)
from forecast.synthetic import write_cs_momentum_universe
from forecast.training import train


def _tiny_daily_data(data_dir: str, **overrides: Any) -> DataConfig:
    preset = interval_data_kwargs("daily")
    kwargs = dict(
        data_dir=data_dir,
        interval="daily",
        horizon=1,
        seq_len=32,
        stride=1,
        min_context=8,
        warmup_bars=16,
        vol_halflife=10,
        z_window=21,
        z_min_periods=8,
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=8,
        allow_mixed_prices=True,
        cs_zscore=True,
    )
    kwargs.update({k: preset[k] for k in ("max_abs_log_return",) if k in preset})
    kwargs.update(overrides)
    return DataConfig(**kwargs)


def _tiny_model(**overrides: Any) -> ForecastModelConfig:
    ssm = interval_model_kwargs("daily")
    kwargs = dict(
        d_model=ssm.get("d_model", 32),
        n_layer=ssm.get("n_layer", 1),
        d_state=ssm.get("d_state", 8),
        expand=2,
        dropout=0.0,
        linear_skip=True,
        dynamic_weights=False,
        dt_min=ssm["dt_min"],
        dt_max=ssm["dt_max"],
    )
    kwargs.update(overrides)
    return ForecastModelConfig(**kwargs)


def run_case(
    name: str,
    data_cfg: DataConfig,
    *,
    skip_only: bool = True,
    residual: bool = True,
    cs_demean: bool = True,
    cs_zscore: bool = True,
    linear_skip: bool = True,
    dynamic_weights: bool = False,
    max_steps: int = 40,
    checkpoint_dir: str,
    log_fn: Any | None = None,
) -> dict[str, Any]:
    data_cfg.residual_target = residual
    data_cfg.cs_zscore = cs_zscore
    model_cfg = _tiny_model(
        linear_skip=linear_skip, dynamic_weights=dynamic_weights
    )
    train_cfg = ForecastTrainConfig(
        batch_size=4,
        max_steps=0 if skip_only else max_steps,
        eval_interval=max(10, max_steps // 2) if not skip_only else 250,
        log_interval=25,
        checkpoint_dir=checkpoint_dir,
        skip_only=skip_only,
        ridge_skip=1.0 if linear_skip else 0.0,
        freeze_skip=True,
        ridge_cs_demean=cs_demean,
        cs_center_loss=True,
        ic_loss_weight=2.0,
        rank_loss_weight=1.0,
        precision="fp32",
        early_stop_evals=4,
        lr_plateau_evals=0,
    )
    if skip_only:
        train_cfg.max_steps = None
        train_cfg.epochs = 1
    summary = train(
        data_cfg,
        model_cfg,
        train_cfg,
        device=torch.device("cpu"),
        log_fn=log_fn,
    )
    test = summary.get("test") or {}
    return {
        "name": name,
        "skip_only": skip_only,
        "residual": residual,
        "cs_demean": cs_demean,
        "cs_zscore": cs_zscore,
        "linear_skip": linear_skip,
        "dynamic_weights": dynamic_weights,
        "cross_section": bool(summary.get("cross_section")),
        "test_cs_ic": test.get("cs_ic", float("nan")),
        "test_cs_ic_spearman": test.get("cs_ic_spearman", float("nan")),
        "test_cs_ic_tstat": test.get("cs_ic_tstat", float("nan")),
        "test_ic": test.get("ic", float("nan")),
        "test_n": test.get("n", float("nan")),
        "best_val_ic": summary.get("best_val_ic", float("nan")),
    }


def format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'case':<28} {'cs_ic':>8} {'cs_sp':>8} {'t':>7} {'pooled':>8} {'cs?':>4}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['name']:<28} "
            f"{float(row['test_cs_ic']):+8.4f} "
            f"{float(row['test_cs_ic_spearman']):+8.4f} "
            f"{float(row['test_cs_ic_tstat']):+7.2f} "
            f"{float(row['test_ic']):+8.4f} "
            f"{'Y' if row['cross_section'] else 'N':>4}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CS residual protocol ablations.")
    p.add_argument("--data-dir", default="")
    p.add_argument("--universe", default="", choices=("", "liquid"))
    p.add_argument("--out-dir", default="checkpoints/ablate_cs")
    p.add_argument(
        "--skip-only-only",
        action="store_true",
        help="do not run the tiny encoder case",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    log_fn = None if args.quiet else print

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = str(_REPO_ROOT / "data" / "_synthetic_cs")
        write_cs_momentum_universe(data_dir, n_names=12, n_days=200, seed=0)

    root = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    common = dict(
        universe=args.universe,
        cross_section_min_names=8,
    )

    cases = [
        ("skip_cs_ridge_residual", dict(skip_only=True, residual=True, cs_demean=True)),
        ("skip_pooled_ridge_residual", dict(skip_only=True, residual=True, cs_demean=False)),
        ("skip_cs_ridge_raw_target", dict(skip_only=True, residual=False, cs_demean=True)),
        ("skip_no_cs_z", dict(skip_only=True, residual=True, cs_demean=True, cs_zscore=False)),
    ]
    if not args.skip_only_only:
        cases.append(
            (
                "tiny_mamba_frozen_skip",
                dict(skip_only=False, residual=True, cs_demean=True, max_steps=40),
            )
        )

    for name, kwargs in cases:
        data_cfg = _tiny_daily_data(data_dir, **common)
        if args.data_dir and not args.universe:
            data_cfg.seq_len = 64
            data_cfg.min_context = 16
            data_cfg.warmup_bars = 21
        max_steps = int(kwargs.pop("max_steps", 40))
        row = run_case(
            name,
            data_cfg,
            checkpoint_dir=str(root / name),
            log_fn=log_fn,
            max_steps=max_steps,
            **kwargs,
        )
        rows.append(row)
        if log_fn:
            log_fn(f"== {name} test cs_ic={row['test_cs_ic']:+.4f}")

    table = format_table(rows)
    print(table)
    summary_path = root / "ablation.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
