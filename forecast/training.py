"""Train the next-hour return forecaster.

Usage:
    python -m forecast.training
    python -m forecast.training --dynamic-weights --epochs 12
    python -m forecast.training --d-model 128 --n-layer 6 --batch-size 8

Supervision is dense and masked: the model predicts at every bar of a window,
and only bars with a well-defined next-hour target contribute to the loss.

Read `val_ic` rather than `val_loss`. On minute-bar returns the loss is
dominated by irreducible noise and barely moves, while the information
coefficient (correlation between prediction and realized return) is what
actually says whether the model found signal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    # Run as a plain script (`python forecast/training.py`, or from an IDE):
    # put the repo root on sys.path so the package imports below resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast.config import (
    DataConfig,
    ForecastModelConfig,
    ForecastTrainConfig,
)
from forecast.data import FEATURE_NAMES, build_datasets
from forecast.model import ReturnForecaster


# --------------------------------------------------------------------------
# Loss and metrics
# --------------------------------------------------------------------------


def masked_loss(
    mean: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cfg: ForecastTrainConfig,
) -> torch.Tensor:
    """Loss over labelled bars only. Returns a zero-grad-safe scalar."""
    weights = mask.to(mean.dtype)
    denom = weights.sum().clamp(min=1.0)

    if cfg.loss == "mse":
        per_bar = (mean - target).pow(2)
    elif cfg.loss == "huber":
        per_bar = F.huber_loss(
            mean, target, reduction="none", delta=cfg.huber_delta
        )
    elif cfg.loss == "gaussian":
        # Heteroscedastic NLL, constants dropped.
        inv_var = torch.exp(-2.0 * log_sigma)
        per_bar = 0.5 * (inv_var * (mean - target).pow(2)) + log_sigma
    else:
        raise ValueError(f"unknown loss {cfg.loss!r}")
    return (per_bar * weights).sum() / denom


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def compute_metrics(
    pred: np.ndarray, target: np.ndarray, scale: np.ndarray
) -> dict[str, float]:
    """Forecast-quality metrics over labelled bars.

    ``pred`` and ``target`` are in volatility units; ``scale`` converts them
    back to log returns.
    """
    if pred.size == 0:
        return {"ic": float("nan"), "n": 0.0}

    mse = float(np.mean((pred - target) ** 2))
    # Skill against the only honest baseline for returns: predict zero.
    baseline_mse = float(np.mean(target**2))
    moved = target != 0.0
    direction = (
        float(np.mean(np.sign(pred[moved]) == np.sign(target[moved])))
        if moved.any()
        else float("nan")
    )
    pred_bps = pred * scale * 1e4
    target_bps = target * scale * 1e4
    return {
        "ic": _pearson(pred, target),
        "mse": mse,
        "baseline_mse": baseline_mse,
        "r2": 1.0 - mse / baseline_mse if baseline_mse > 0 else float("nan"),
        "direction": direction,
        "rmse_bps": float(np.sqrt(np.mean((pred_bps - target_bps) ** 2))),
        "pred_std_bps": float(np.std(pred_bps)),
        "target_std_bps": float(np.std(target_bps)),
        "n": float(pred.size),
    }


# --------------------------------------------------------------------------
# Training utilities
# --------------------------------------------------------------------------


def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if device.type == "cpu" and precision == "fp16":
        dtype = torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def build_optimizer(
    model: ReturnForecaster, cfg: ForecastTrainConfig
) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.dim() < 2
            or name.endswith("bias")
            or "norm" in name
            or name.endswith("A_log")
            or name.endswith(".D")
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def lr_at(step: int, total_steps: int, cfg: ForecastTrainConfig) -> float:
    warmup = max(1, int(total_steps * cfg.warmup_frac))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 * cfg.lr + 0.9 * cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(
    model: ReturnForecaster,
    loader: DataLoader,
    device: torch.device,
    train_cfg: ForecastTrainConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    total_loss, n_batches = 0.0, 0

    for i, (x, y, mask, scale) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, mask, scale = (t.to(device) for t in (x, y, mask, scale))
        with autocast_context(device, train_cfg.precision):
            mean, log_sigma = model(x)
        mean = mean.float()
        loss = masked_loss(mean, log_sigma.float(), y, mask, train_cfg)
        total_loss += float(loss)
        n_batches += 1
        sel = mask.bool()
        preds.append(mean[sel].cpu().numpy())
        targets.append(y[sel].cpu().numpy())
        scales.append(scale[sel].cpu().numpy())

    model.train()
    metrics = compute_metrics(
        np.concatenate(preds) if preds else np.empty(0),
        np.concatenate(targets) if targets else np.empty(0),
        np.concatenate(scales) if scales else np.empty(0),
    )
    metrics["loss"] = total_loss / max(1, n_batches)
    return metrics


def _fmt(metrics: dict[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.5f} ic={metrics['ic']:+.4f} "
        f"r2={metrics['r2']:+.5f} dir={metrics['direction']:.4f} "
        f"pred_std={metrics['pred_std_bps']:.2f}bps n={int(metrics['n'])}"
    )


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def train(
    data_cfg: DataConfig,
    model_cfg: ForecastModelConfig,
    train_cfg: ForecastTrainConfig,
    *,
    device: torch.device | None = None,
    log_fn: Any | None = print,
) -> dict[str, Any]:
    device = device or select_device()
    set_seed(train_cfg.seed)

    bundle = build_datasets(data_cfg, log_fn=log_fn)
    datasets = bundle["datasets"]
    model_cfg.n_features = len(bundle["feature_names"])

    model = ReturnForecaster(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if log_fn:
        log_fn(
            f"device={device} params={n_params:,} features={model_cfg.n_features} "
            f"seq_len={data_cfg.seq_len} horizon={data_cfg.horizon} "
            f"dynamic_weights={model_cfg.dynamic_weights}"
        )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )

    steps_per_epoch = len(train_loader)
    total_steps = train_cfg.max_steps or steps_per_epoch * train_cfg.epochs
    optimizer = build_optimizer(model, train_cfg)
    use_scaler = train_cfg.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path, step: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": model_cfg.to_dict(),
                "data_config": data_cfg.to_dict(),
                "train_config": train_cfg.to_dict(),
                "feature_names": bundle["feature_names"],
                "feature_mean": bundle["feature_mean"],
                "feature_std": bundle["feature_std"],
                "symbols": bundle["meta"],
                "step": step,
                "metrics": metrics,
            },
            path,
        )

    history: list[dict[str, Any]] = []
    best_ic = -float("inf")
    best_step = -1
    evals_without_gain = 0
    data_iter = _cycle(train_loader)
    running: list[float] = []
    t0 = time.perf_counter()
    model.train()

    if log_fn:
        log_fn(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    for step in range(total_steps):
        lr = lr_at(step, total_steps, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y, mask, _scale = next(data_iter)
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_cfg.precision):
            mean, log_sigma = model(x)
        loss = masked_loss(mean.float(), log_sigma.float(), y, mask, train_cfg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        )
        scaler.step(optimizer)
        scaler.update()
        running.append(float(loss.detach()))

        if (step + 1) % train_cfg.log_interval == 0 or step == 0:
            elapsed = max(time.perf_counter() - t0, 1e-8)
            if log_fn:
                log_fn(
                    f"step {step + 1:6d}/{total_steps}  "
                    f"loss={np.mean(running[-train_cfg.log_interval:]):.5f}  "
                    f"grad={grad_norm:.3f}  lr={lr:.2e}  "
                    f"{(step + 1) / elapsed:.2f} it/s"
                )

        if (step + 1) % train_cfg.eval_interval == 0 or step + 1 == total_steps:
            val = evaluate(model, val_loader, device, train_cfg)
            val["step"] = float(step + 1)
            history.append(val)
            if log_fn:
                log_fn(f"  eval step {step + 1}: {_fmt(val)}")
            save(ckpt_dir / "last.pt", step + 1, val)

            ic = val["ic"]
            if np.isfinite(ic) and ic > best_ic:
                best_ic, best_step = ic, step + 1
                evals_without_gain = 0
                save(ckpt_dir / "best.pt", step + 1, val)
                if log_fn:
                    log_fn(f"  new best val_ic={ic:+.4f} -> {ckpt_dir / 'best.pt'}")
            else:
                evals_without_gain += 1
                if evals_without_gain >= train_cfg.early_stop_evals:
                    if log_fn:
                        log_fn(f"early stop: no val IC gain in {evals_without_gain} evals")
                    break

    # Final test evaluation uses the best checkpoint, not the last one.
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    test = evaluate(model, test_loader, device, train_cfg)
    if log_fn:
        log_fn(f"TEST (best step {best_step}): {_fmt(test)}")

    summary = {
        "best_val_ic": best_ic,
        "best_step": best_step,
        "history": history,
        "test": test,
        "n_params": n_params,
        "device": str(device),
        "elapsed_sec": time.perf_counter() - t0,
        "symbols": bundle["meta"],
    }
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the next-hour return forecaster.")
    d, m, t = DataConfig(), ForecastModelConfig(), ForecastTrainConfig()

    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default=d.data_dir)
    g.add_argument("--horizon", type=int, default=d.horizon, help="bars ahead (60 = 1h)")
    g.add_argument("--seq-len", type=int, default=d.seq_len)
    g.add_argument("--stride", type=int, default=d.stride)
    g.add_argument("--min-context", type=int, default=d.min_context)
    g.add_argument("--min-session-bars", type=int, default=d.min_session_bars)
    g.add_argument("--val-fraction", type=float, default=d.val_fraction)
    g.add_argument("--test-fraction", type=float, default=d.test_fraction)

    g = p.add_argument_group("model")
    g.add_argument("--d-model", type=int, default=m.d_model)
    g.add_argument("--n-layer", type=int, default=m.n_layer)
    g.add_argument("--d-state", type=int, default=m.d_state)
    g.add_argument("--expand", type=int, default=m.expand)
    g.add_argument("--dropout", type=float, default=m.dropout)
    g.add_argument("--dynamic-weights", action="store_true")
    g.add_argument("--dynamic-strength", type=float, default=m.dynamic_strength)
    g.add_argument("--no-heteroscedastic", action="store_true")

    g = p.add_argument_group("optim")
    g.add_argument("--batch-size", type=int, default=t.batch_size)
    g.add_argument("--epochs", type=int, default=t.epochs)
    g.add_argument("--max-steps", type=int, default=t.max_steps)
    g.add_argument("--lr", type=float, default=t.lr)
    g.add_argument("--weight-decay", type=float, default=t.weight_decay)
    g.add_argument("--loss", choices=("huber", "mse", "gaussian"), default=t.loss)
    g.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=t.precision)
    g.add_argument("--seed", type=int, default=t.seed)
    g.add_argument("--eval-interval", type=int, default=t.eval_interval)
    g.add_argument("--log-interval", type=int, default=t.log_interval)
    g.add_argument("--num-workers", type=int, default=t.num_workers)
    g.add_argument("--checkpoint-dir", default=t.checkpoint_dir)
    g.add_argument("--early-stop-evals", type=int, default=t.early_stop_evals)
    g.add_argument("--cpu", action="store_true", help="force CPU")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        horizon=args.horizon,
        seq_len=args.seq_len,
        stride=args.stride,
        min_context=args.min_context,
        min_session_bars=args.min_session_bars,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    model_cfg = ForecastModelConfig(
        n_features=len(FEATURE_NAMES),
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_state=args.d_state,
        expand=args.expand,
        dropout=args.dropout,
        heteroscedastic=not args.no_heteroscedastic,
        dynamic_weights=args.dynamic_weights,
        dynamic_strength=args.dynamic_strength,
    )
    train_cfg = ForecastTrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        precision=args.precision,
        seed=args.seed,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        early_stop_evals=args.early_stop_evals,
    )
    train(
        data_cfg,
        model_cfg,
        train_cfg,
        device=torch.device("cpu") if args.cpu else None,
    )


if __name__ == "__main__":
    main()
