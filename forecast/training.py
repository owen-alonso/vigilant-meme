"""Train the equity return forecaster.

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
from typing import Any

# `python forecast/training.py` (and IDEs) are not package imports.
# Put the repo root on sys.path so `forecast.*` / `mamba_lm.*` resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from forecast.checkpoint import load_forecast_checkpoint, save_forecast_checkpoint
from forecast.config import (
    DataConfig,
    ForecastModelConfig,
    ForecastTrainConfig,
    interval_data_kwargs,
    interval_model_kwargs,
    validate_loss_head,
)
from forecast.data import FEATURE_NAMES, build_datasets
from forecast.model import ReturnForecaster
from mamba_lm.model import format_dynamic_diagnostics
from mamba_lm.paths import anchor_to_repo
from mamba_lm.reporting import clip_grad_norm_unique
from mamba_lm.training_utils import (
    autocast_context,
    build_optimizer,
    cycle_loader,
    grads_finite,
    keep_awake,
    lr_warmup_cosine,
    require_nonempty_loader,
    select_device,
    set_seed,
)


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
    denom = weights.sum()
    if float(denom) <= 0:
        return mean.new_tensor(float("nan"))

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
    if cfg.loss != "gaussian" and cfg.sigma_aux_weight > 0:
        # Mean is detached so Huber/MSE still own the location; sigma learns
        # residual scale, which generate.py turns into a confidence score.
        resid_sq = (mean.detach() - target).pow(2)
        inv_var = torch.exp(-2.0 * log_sigma)
        aux = 0.5 * (inv_var * resid_sq) + log_sigma
        per_bar = per_bar + cfg.sigma_aux_weight * aux
    location = (per_bar * weights).sum() / denom
    if cfg.ic_loss_weight <= 0:
        return location
    ic_term = masked_correlation_loss(
        mean, target, mask, winsor=cfg.ic_winsor
    )
    if not torch.isfinite(ic_term):
        return location
    return location + cfg.ic_loss_weight * ic_term


def masked_correlation_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    winsor: float = 3.0,
) -> torch.Tensor:
    """``1 - Pearson`` on labelled bars, averaged over sequences in the batch.

    Pred and target are winsorized so a split-sized jump cannot dominate.
    Huber still owns prediction scale; this term is scale-free.
    """
    weights = mask.to(dtype=mean.dtype)
    cap = float(winsor)
    pred = mean.clamp(-cap, cap)
    y = target.clamp(-cap, cap)
    n = weights.sum(dim=-1)
    denom_n = n.clamp(min=1.0)
    mu_p = (pred * weights).sum(dim=-1) / denom_n
    mu_y = (y * weights).sum(dim=-1) / denom_n
    pc = (pred - mu_p.unsqueeze(-1)) * weights
    yc = (y - mu_y.unsqueeze(-1)) * weights
    cov = (pc * yc).sum(dim=-1)
    var_p = (pc * pc).sum(dim=-1)
    var_y = (yc * yc).sum(dim=-1)
    valid = (n >= 2) & (var_p > 1e-8) & (var_y > 1e-8)
    if not bool(valid.any()):
        return mean.new_zeros(())
    rho = cov[valid] / torch.sqrt(var_p[valid] * var_y[valid]).clamp(min=1e-8)
    return 1.0 - rho.mean()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Ordinal ranks (ties keep first-come order). Good enough for Spearman IC."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return _pearson(_rankdata(a), _rankdata(b))


def compute_metrics(
    pred: np.ndarray, target: np.ndarray, scale: np.ndarray, *, winsor: float = 3.0
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
    cap = float(winsor)
    ic_raw = _pearson(pred, target)
    ic_winsor = _pearson(np.clip(pred, -cap, cap), np.clip(target, -cap, cap))
    ic_spearman = _spearman(pred, target)
    select_parts = [v for v in (ic_winsor, ic_spearman) if np.isfinite(v)]
    ic_select = float(np.mean(select_parts)) if select_parts else float("nan")
    return {
        "ic": ic_select,
        "ic_raw": ic_raw,
        "ic_winsor": ic_winsor,
        "ic_spearman": ic_spearman,
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
# Training utilities (forecast-specific)
# --------------------------------------------------------------------------


def lr_at(step: int, total_steps: int, cfg: ForecastTrainConfig) -> float:
    return lr_warmup_cosine(
        step,
        total_steps=total_steps,
        lr=cfg.lr,
        warmup_frac=cfg.warmup_frac,
    )


def decide_val_plateau(
    *,
    improved: bool,
    evals_without_gain: int,
    lr_scale: float,
    plateau_evals: int,
    plateau_factor: float,
    min_scale: float,
    early_stop_evals: int,
) -> tuple[int, float, bool, bool, str | None]:
    """Handle a val-IC eval: maybe drop LR, restore best, or (optionally) stop.

    Returns ``(evals_without_gain, lr_scale, stop, restore_best, log_message)``.
    """
    if improved:
        return 0, lr_scale, False, False, None
    n = evals_without_gain + 1
    if early_stop_evals > 0 and n >= early_stop_evals:
        return n, lr_scale, True, False, (
            f"early stop: no val IC gain in {n} evals"
        )
    if plateau_evals > 0 and n >= plateau_evals:
        factor = min(1.0, max(1e-6, float(plateau_factor)))
        floor = max(0.0, float(min_scale))
        new_scale = max(floor, lr_scale * factor)
        if new_scale < lr_scale:
            return 0, new_scale, False, True, (
                f"val IC plateau for {n} evals: lr scale "
                f"{lr_scale:.3g} -> {new_scale:.3g} (restoring best.pt, continuing)"
            )
        return 0, lr_scale, False, False, (
            f"val IC plateau for {n} evals: lr scale already at floor "
            f"({lr_scale:.3g}), continuing"
        )
    return n, lr_scale, False, False, None


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
    weighted_loss = 0.0
    total_weight = 0.0

    for i, (x, y, mask, scale) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, mask, scale = (t.to(device) for t in (x, y, mask, scale))
        with autocast_context(device, train_cfg.precision):
            mean, log_sigma = model(x)
        mean = mean.float()
        loss = masked_loss(mean, log_sigma.float(), y, mask, train_cfg)
        weight = float(mask.to(mean.dtype).sum())
        if weight > 0 and math.isfinite(float(loss)):
            weighted_loss += float(loss) * weight
            total_weight += weight
        sel = mask.bool()
        preds.append(mean[sel].cpu().numpy())
        targets.append(y[sel].cpu().numpy())
        scales.append(scale[sel].cpu().numpy())

    model.train()
    metrics = compute_metrics(
        np.concatenate(preds) if preds else np.empty(0),
        np.concatenate(targets) if targets else np.empty(0),
        np.concatenate(scales) if scales else np.empty(0),
        winsor=train_cfg.ic_winsor,
    )
    metrics["loss"] = weighted_loss / total_weight if total_weight > 0 else float("nan")
    return metrics


def _fmt(metrics: dict[str, float]) -> str:
    spearman = metrics.get("ic_spearman", float("nan"))
    raw = metrics.get("ic_raw", metrics.get("ic", float("nan")))
    return (
        f"loss={metrics['loss']:.5f} ic={metrics['ic']:+.4f} "
        f"spearman={spearman:+.4f} raw={raw:+.4f} "
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
    with keep_awake(log_fn=log_fn):
        return _train(
            data_cfg,
            model_cfg,
            train_cfg,
            device=device,
            log_fn=log_fn,
        )


def _train(
    data_cfg: DataConfig,
    model_cfg: ForecastModelConfig,
    train_cfg: ForecastTrainConfig,
    *,
    device: torch.device | None = None,
    log_fn: Any | None = print,
) -> dict[str, Any]:
    device = device or select_device()
    set_seed(train_cfg.seed)
    validate_loss_head(model_cfg, train_cfg)

    bundle = build_datasets(data_cfg, log_fn=log_fn)
    datasets = bundle["datasets"]
    model_cfg.n_features = len(bundle["feature_names"])

    model = ReturnForecaster(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if log_fn:
        log_fn(
            f"device={device} params={n_params:,} features={model_cfg.n_features} "
            f"seq_len={data_cfg.seq_len} horizon={data_cfg.horizon} "
            f"linear_skip={model_cfg.linear_skip} "
            f"dynamic_weights={model_cfg.dynamic_weights} "
            f"loss={train_cfg.loss} ic_loss_weight={train_cfg.ic_loss_weight} "
            f"heteroscedastic={model_cfg.heteroscedastic}"
        )
        autocast_context(device, train_cfg.precision, log_fn=log_fn)

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
    require_nonempty_loader(train_loader, "train")
    if train_cfg.max_steps is None:
        total_steps = steps_per_epoch * train_cfg.epochs
    else:
        total_steps = train_cfg.max_steps
    if total_steps < 1:
        raise RuntimeError("total_steps is 0; check epochs, max_steps, and dataset size")
    optimizer = build_optimizer(
        model, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    use_scaler = train_cfg.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    ckpt_dir = anchor_to_repo(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if log_fn:
        log_fn(f"checkpoints -> {ckpt_dir}")

    def save(path: Path, step: int, metrics: dict[str, float]) -> None:
        save_forecast_checkpoint(
            path,
            model=model,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            feature_names=bundle["feature_names"],
            feature_mean=bundle["feature_mean"],
            feature_std=bundle["feature_std"],
            symbols=bundle["meta"],
            step=step,
            metrics=metrics,
        )

    history: list[dict[str, Any]] = []
    best_ic = -float("inf")
    best_step = -1
    evals_without_gain = 0
    lr_scale = 1.0
    data_iter = cycle_loader(train_loader)
    running: list[float] = []
    t0 = time.perf_counter()
    model.train()

    if log_fn:
        log_fn(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    last_completed = 0
    interrupted = False
    try:
        for step in range(total_steps):
            lr = lr_at(step, total_steps, train_cfg) * lr_scale
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
            if not grads_finite(model):
                if log_fn:
                    log_fn(f"step {step + 1}: non-finite gradients, skipping optimizer step")
                scaler.update()
                continue
            grad_norm = clip_grad_norm_unique(model, train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach()))
            last_completed = step + 1

            if (step + 1) % train_cfg.log_interval == 0 or step == 0:
                elapsed = max(time.perf_counter() - t0, 1e-8)
                if log_fn:
                    msg = (
                        f"step {step + 1:6d}/{total_steps}  "
                        f"loss={np.mean(running[-train_cfg.log_interval:]):.5f}  "
                        f"grad={grad_norm:.3f}  lr={lr:.2e}  "
                        f"{(step + 1) / elapsed:.2f} it/s"
                    )
                    diag = model.collect_dynamic_diagnostics()
                    extra_diag = format_dynamic_diagnostics(diag)
                    if extra_diag:
                        msg += f"  {extra_diag}"
                    log_fn(msg)

            if (step + 1) % train_cfg.eval_interval == 0 or step + 1 == total_steps:
                val = evaluate(model, val_loader, device, train_cfg)
                val["step"] = float(step + 1)
                history.append(val)
                if log_fn:
                    log_fn(f"  eval step {step + 1}: {_fmt(val)}")
                save(ckpt_dir / "last.pt", step + 1, val)

                ic = val["ic"]
                improved = bool(np.isfinite(ic) and ic > best_ic)
                if improved:
                    best_ic, best_step = ic, step + 1
                    save(ckpt_dir / "best.pt", step + 1, val)
                    if log_fn:
                        log_fn(f"  new best val_ic={ic:+.4f} -> {ckpt_dir / 'best.pt'}")
                evals_without_gain, lr_scale, stop, restore_best, plateau_msg = (
                    decide_val_plateau(
                        improved=improved,
                        evals_without_gain=evals_without_gain,
                        lr_scale=lr_scale,
                        plateau_evals=train_cfg.lr_plateau_evals,
                        plateau_factor=train_cfg.lr_plateau_factor,
                        min_scale=train_cfg.lr_plateau_min_scale,
                        early_stop_evals=train_cfg.early_stop_evals,
                    )
                )
                if plateau_msg and log_fn:
                    log_fn(f"  {plateau_msg}")
                if restore_best:
                    best_path = ckpt_dir / "best.pt"
                    if best_path.exists():
                        load_forecast_checkpoint(
                            best_path, map_location=device, model=model
                        )
                if stop:
                    break
    except KeyboardInterrupt:
        interrupted = True
        if history:
            metrics = {
                k: float(v)
                for k, v in history[-1].items()
                if isinstance(v, (int, float))
            }
        else:
            metrics = {"ic": float("nan"), "n": 0.0}
        metrics["step"] = float(last_completed)
        save(ckpt_dir / "last.pt", last_completed, metrics)
        if log_fn:
            log_fn(
                f"keyboard interrupt at step {last_completed}/{total_steps}; "
                f"saved {ckpt_dir / 'last.pt'}"
                + (
                    f" (best.pt still step {best_step})"
                    if best_step >= 0
                    else " (no best.pt yet)"
                )
            )

    # Final test evaluation uses the best checkpoint when available.
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    if interrupted:
        test = {
            k: float(v)
            for k, v in (history[-1].items() if history else [])
            if isinstance(v, (int, float))
        }
        if log_fn:
            log_fn("TEST skipped (interrupted). Generate with best.pt if it exists, else last.pt.")
    else:
        if best_path.exists():
            load_forecast_checkpoint(best_path, map_location=device, model=model)
        elif last_path.exists():
            load_forecast_checkpoint(last_path, map_location=device, model=model)
            if log_fn:
                log_fn("warning: no best.pt saved; TEST uses last.pt weights")
        elif log_fn:
            log_fn("warning: no checkpoint saved; TEST uses final in-memory weights")
        test = evaluate(model, test_loader, device, train_cfg)
        if log_fn:
            if best_path.exists():
                log_fn(f"TEST (best step {best_step}): {_fmt(test)}")
            elif last_path.exists():
                log_fn(f"TEST (last checkpoint; no best.pt was saved): {_fmt(test)}")
            else:
                log_fn(f"TEST (in-memory weights; no checkpoint saved): {_fmt(test)}")

    summary = {
        "best_val_ic": best_ic,
        "best_step": best_step,
        "history": history,
        "test": test,
        "interrupted": interrupted,
        "last_step": last_completed,
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
    p = argparse.ArgumentParser(description="Train the equity return forecaster.")
    d, m, t = DataConfig(), ForecastModelConfig(), ForecastTrainConfig()

    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default=d.data_dir)
    g.add_argument(
        "--interval",
        default=d.interval,
        choices=("daily", "weekly", "monthly", "1min", "5min", "15min", "30min", "60min"),
        help="bar size; match the parquet interval you downloaded",
    )
    g.add_argument("--horizon", type=int, default=d.horizon, help="bars ahead (1 = next daily bar)")
    g.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="window length (default: 128 daily, 52 weekly, 256 for 1min)",
    )
    g.add_argument(
        "--stride",
        type=int,
        default=None,
        help="train window stride (default depends on --interval)",
    )
    g.add_argument(
        "--min-context",
        type=int,
        default=None,
        help="unsupervised prefix of each window (default depends on --interval)",
    )
    g.add_argument("--min-session-bars", type=int, default=d.min_session_bars)
    g.add_argument("--val-fraction", type=float, default=d.val_fraction)
    g.add_argument("--test-fraction", type=float, default=d.test_fraction)
    g.add_argument(
        "--allow-stale-horizon",
        action="store_true",
        help="label bars whose t+horizon slot was not a real print (not recommended)",
    )

    g = p.add_argument_group("model")
    g.add_argument("--d-model", type=int, default=m.d_model)
    g.add_argument("--n-layer", type=int, default=m.n_layer)
    g.add_argument("--d-state", type=int, default=m.d_state)
    g.add_argument("--expand", type=int, default=m.expand)
    g.add_argument("--dropout", type=float, default=m.dropout)
    g.add_argument("--dynamic-weights", action="store_true")
    g.add_argument("--dynamic-strength", type=float, default=m.dynamic_strength)
    g.add_argument(
        "--no-linear-skip",
        action="store_true",
        help="disable the features->mean skip (Mamba-only readout)",
    )
    g.add_argument(
        "--heteroscedastic",
        action="store_true",
        help="train a log-sigma head (with --loss gaussian, or Huber/MSE + aux weight)",
    )
    g.add_argument(
        "--no-heteroscedastic",
        action="store_true",
        help="force a single-output head even with --loss gaussian",
    )

    g = p.add_argument_group("optim")
    g.add_argument("--batch-size", type=int, default=t.batch_size)
    g.add_argument("--epochs", type=int, default=t.epochs)
    g.add_argument("--max-steps", type=int, default=t.max_steps)
    g.add_argument("--lr", type=float, default=t.lr)
    g.add_argument("--weight-decay", type=float, default=t.weight_decay)
    g.add_argument("--loss", choices=("huber", "mse", "gaussian"), default=t.loss)
    g.add_argument(
        "--ic-loss-weight",
        type=float,
        default=t.ic_loss_weight,
        help="weight on 1-Pearson mixed into the mean loss (0 disables)",
    )
    g.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=t.precision)
    g.add_argument("--seed", type=int, default=t.seed)
    g.add_argument("--eval-interval", type=int, default=t.eval_interval)
    g.add_argument("--log-interval", type=int, default=t.log_interval)
    g.add_argument("--num-workers", type=int, default=t.num_workers)
    g.add_argument("--checkpoint-dir", default=t.checkpoint_dir)
    g.add_argument(
        "--early-stop-evals",
        type=int,
        default=t.early_stop_evals,
        help="stop after N evals with no val-IC gain (0 = never; default)",
    )
    g.add_argument(
        "--lr-plateau-evals",
        type=int,
        default=t.lr_plateau_evals,
        help="after N evals with no val-IC gain, cut LR and restore best.pt (0 = off)",
    )
    g.add_argument(
        "--lr-plateau-factor",
        type=float,
        default=t.lr_plateau_factor,
        help="multiply the scheduled LR by this on a val-IC plateau",
    )
    g.add_argument("--cpu", action="store_true", help="force CPU")
    return p


def configs_from_cli(
    args: argparse.Namespace,
) -> tuple[DataConfig, ForecastModelConfig, ForecastTrainConfig]:
    """Map parsed CLI flags onto configs, filling interval-specific lookbacks."""
    if args.heteroscedastic and args.no_heteroscedastic:
        raise SystemExit("use only one of --heteroscedastic / --no-heteroscedastic")
    if args.loss == "gaussian":
        heteroscedastic = not args.no_heteroscedastic
    else:
        heteroscedastic = args.heteroscedastic

    preset = interval_data_kwargs(args.interval)
    ssm = interval_model_kwargs(args.interval)
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        interval=args.interval,
        horizon=args.horizon,
        seq_len=preset["seq_len"] if args.seq_len is None else args.seq_len,
        stride=preset["stride"] if args.stride is None else args.stride,
        min_context=(
            preset["min_context"] if args.min_context is None else args.min_context
        ),
        min_session_bars=args.min_session_bars,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        require_horizon_traded=not args.allow_stale_horizon,
        vol_halflife=preset["vol_halflife"],
        z_window=preset["z_window"],
        z_min_periods=preset["z_min_periods"],
        warmup_bars=preset["warmup_bars"],
        max_abs_log_return=preset["max_abs_log_return"],
        supervise_last=preset["supervise_last"],
    )
    model_cfg = ForecastModelConfig(
        n_features=len(FEATURE_NAMES),
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_state=args.d_state,
        expand=args.expand,
        dropout=args.dropout,
        heteroscedastic=heteroscedastic,
        linear_skip=not args.no_linear_skip,
        dt_min=ssm["dt_min"],
        dt_max=ssm["dt_max"],
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
        ic_loss_weight=args.ic_loss_weight,
        sigma_aux_weight=(
            0.0
            if (not heteroscedastic or args.loss == "gaussian")
            else 0.5
        ),
        precision=args.precision,
        seed=args.seed,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        early_stop_evals=args.early_stop_evals,
        lr_plateau_evals=args.lr_plateau_evals,
        lr_plateau_factor=args.lr_plateau_factor,
    )
    return data_cfg, model_cfg, train_cfg


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    data_cfg, model_cfg, train_cfg = configs_from_cli(args)
    train(
        data_cfg,
        model_cfg,
        train_cfg,
        device=torch.device("cpu") if args.cpu else None,
    )


if __name__ == "__main__":
    main()
