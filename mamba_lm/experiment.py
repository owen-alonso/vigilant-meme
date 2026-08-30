"""Baseline vs Dynamic A training comparison (identical hparams except the flag)."""

from __future__ import annotations

import copy
from typing import Any

from mamba_lm.config import MambaConfig, TrainConfig
from mamba_lm.reporting import format_parameter_report
from mamba_lm.train import train


def run_training_comparison(
    model_cfg: MambaConfig | None = None,
    train_cfg: TrainConfig | None = None,
    *,
    log_fn=print,
) -> dict[str, Any]:
    model_cfg = model_cfg or MambaConfig(
        d_model=64,
        n_layer=2,
        d_state=8,
        expand=2,
    )
    train_cfg = train_cfg or TrainConfig(
        batch_size=4,
        seq_len=128,
        max_steps=50,
        eval_interval=25,
        log_interval=10,
        eval_batches=4,
        lr=3e-4,
        seed=42,
        precision="fp32",
    )

    results: dict[str, Any] = {}
    for name, dynamic in (("baseline", False), ("dynamic_A", True)):
        if log_fn:
            log_fn(f"\n======== {name} ========")
        cfg = copy.deepcopy(model_cfg)
        cfg.dynamic_weights = dynamic
        cfg.dynamic_A = True
        if cfg.dynamic_weights:
            cfg._validate_dynamic_v1()
        tcfg = copy.deepcopy(train_cfg)
        tcfg.checkpoint_dir = f"{train_cfg.checkpoint_dir}/{name}"
        results[name] = train(cfg, tcfg, log_fn=log_fn)

    if log_fn:
        log_fn("\n======== comparison ========")
        for name, result in results.items():
            log_fn(f"\n{name}:")
            log_fn(format_parameter_report(result["parameter_report"]))
            best = result.get("best_val_loss")
            best_s = f"{best:.4f}" if isinstance(best, float) else "n/a"
            log_fn(
                f"final train loss={result['final_train_loss']:.4f}  "
                f"best val loss={best_s}  "
                f"final val loss={result['final_val_loss']:.4f}  "
                f"tok/s={result['tokens_per_sec']:.0f}  "
                f"mean grad_norm={result['mean_grad_norm']:.3f}  "
                f"peak_mem={result['peak_memory_bytes'] / (1024**2):.1f} MiB"
            )
    return results
