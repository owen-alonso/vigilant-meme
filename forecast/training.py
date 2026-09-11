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
from forecast.data import FEATURE_NAMES, build_datasets, collate_forecast, fit_ridge_readout
from forecast.model import ReturnForecaster
from mamba_lm.model import format_dynamic_diagnostics
from mamba_lm.paths import anchor_to_repo
from mamba_lm.reporting import clip_grad_norm_unique
from mamba_lm.training_utils import (
    autocast_context,
    build_optimizer,
    cycle_loader,
    dataloader_kwargs,
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


def _center_by_date(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    date_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract the labelled within-date mean. No-op without date breadth."""
    dates = _expand_date_ids(date_ids, mask)
    if dates is None:
        return pred, target
    sel = mask.bool()
    if int(sel.sum()) < 2:
        return pred, target
    pred_c = pred.clone()
    tgt_c = target.clone()
    for key in dates[sel].unique():
        m = sel & (dates == key)
        if int(m.sum()) < 2:
            continue
        pred_c[m] = pred[m] - pred[m].mean()
        tgt_c[m] = target[m] - target[m].mean()
    return pred_c, tgt_c


def masked_loss(
    mean: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cfg: ForecastTrainConfig,
    date_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Loss over labelled bars only. Returns a zero-grad-safe scalar."""
    weights = mask.to(mean.dtype)
    denom = weights.sum()
    if float(denom) <= 0:
        return mean.new_tensor(float("nan"))

    loc_mean, loc_target = mean, target
    if cfg.cs_center_loss:
        loc_mean, loc_target = _center_by_date(mean, target, mask, date_ids)

    if cfg.loss == "mse":
        per_bar = (loc_mean - loc_target).pow(2)
    elif cfg.loss == "huber":
        per_bar = F.huber_loss(
            loc_mean, loc_target, reduction="none", delta=cfg.huber_delta
        )
    elif cfg.loss == "gaussian":
        # Heteroscedastic NLL, constants dropped.
        inv_var = torch.exp(-2.0 * log_sigma)
        per_bar = 0.5 * (inv_var * (loc_mean - loc_target).pow(2)) + log_sigma
    else:
        raise ValueError(f"unknown loss {cfg.loss!r}")
    if cfg.loss != "gaussian" and cfg.sigma_aux_weight > 0:
        # Mean is detached so Huber/MSE still own the location; sigma learns
        # residual scale, which generate.py turns into a confidence score.
        resid_sq = (loc_mean.detach() - loc_target).pow(2)
        inv_var = torch.exp(-2.0 * log_sigma)
        aux = 0.5 * (inv_var * resid_sq) + log_sigma
        per_bar = per_bar + cfg.sigma_aux_weight * aux
    location = (per_bar * weights).sum() / denom
    total = cfg.location_loss_weight * location
    if cfg.ic_loss_weight > 0:
        ic_term = masked_correlation_loss(
            mean, target, mask, winsor=cfg.ic_winsor, date_ids=date_ids
        )
        if torch.isfinite(ic_term):
            total = total + cfg.ic_loss_weight * ic_term
    if cfg.sign_loss_weight > 0:
        sign_term = masked_sign_loss(
            mean, target, mask, min_abs=cfg.sign_min_abs
        )
        if torch.isfinite(sign_term):
            total = total + cfg.sign_loss_weight * sign_term
    if cfg.rank_loss_weight > 0:
        rank_term = masked_pairwise_rank_loss(mean, target, mask, date_ids=date_ids)
        if torch.isfinite(rank_term):
            total = total + cfg.rank_loss_weight * rank_term
    if cfg.pred_std_weight > 0:
        scale_term = masked_pred_std_loss(mean, target, mask)
        if torch.isfinite(scale_term):
            total = total + cfg.pred_std_weight * scale_term
    return total


def _expand_date_ids(
    date_ids: torch.Tensor | None, mask: torch.Tensor
) -> torch.Tensor | None:
    if date_ids is None:
        return None
    if date_ids.shape == mask.shape:
        return date_ids
    if date_ids.dim() == 1 and date_ids.size(0) == mask.size(0):
        return date_ids.unsqueeze(-1).expand_as(mask)
    return date_ids.reshape(mask.shape)


def _pearson_1d(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pc = pred - pred.mean()
    yc = y - y.mean()
    var_p = (pc * pc).sum()
    var_y = (yc * yc).sum()
    if float(var_p.detach()) <= 1e-8 or float(var_y.detach()) <= 1e-8:
        return pred.new_zeros(())
    return (pc * yc).sum() / torch.sqrt(var_p * var_y).clamp(min=1e-8)


def masked_correlation_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    winsor: float = 3.0,
    date_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """``1 - Pearson`` on labelled bars; within-date when ``date_ids`` has breadth."""
    cap = float(winsor)
    pred = mean.clamp(-cap, cap)
    y = target.clamp(-cap, cap)
    dates = _expand_date_ids(date_ids, mask)
    if dates is not None:
        sel = mask.bool()
        p = pred[sel]
        yy = y[sel]
        keys = dates[sel]
        rhos: list[torch.Tensor] = []
        for key in keys.unique():
            m = keys == key
            if int(m.sum()) < 3:
                continue
            rho = _pearson_1d(p[m], yy[m])
            if torch.isfinite(rho):
                rhos.append(rho)
        if rhos:
            return 1.0 - torch.stack(rhos).mean()
    weights = mask.to(dtype=mean.dtype).reshape(-1)
    pred_f = pred.reshape(-1)
    y_f = y.reshape(-1)
    wsum = weights.sum()
    if float(wsum.detach()) < 2:
        return mean.new_zeros(())
    mu_p = (pred_f * weights).sum() / wsum
    mu_y = (y_f * weights).sum() / wsum
    pc = (pred_f - mu_p) * weights
    yc = (y_f - mu_y) * weights
    cov = (pc * yc).sum()
    var_p = (pc * pc).sum()
    var_y = (yc * yc).sum()
    if float(var_p.detach()) <= 1e-8 or float(var_y.detach()) <= 1e-8:
        return mean.new_zeros(())
    rho = cov / torch.sqrt(var_p * var_y).clamp(min=1e-8)
    return 1.0 - rho


def masked_sign_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_abs: float = 0.25,
) -> torch.Tensor:
    """BCE on the sign of labelled moves larger than ``min_abs`` vol units."""
    weights = mask.to(dtype=mean.dtype) * (target.abs() >= float(min_abs)).to(
        dtype=mean.dtype
    )
    denom = weights.sum()
    if float(denom.detach()) <= 0:
        return mean.new_zeros(())
    labels = (target > 0).to(dtype=mean.dtype)
    # Sign BCE on raw logits wants |mean| -> inf. Divide by batch std so
    # this term only rotates predictions, matching Pearson.
    labelled = mask.bool()
    scale = mean.detach()[labelled].std(unbiased=False).clamp(min=1.0)
    per_bar = F.binary_cross_entropy_with_logits(
        mean / scale, labels, reduction="none"
    )
    return (per_bar * weights).sum() / denom


def _ranknet(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_zeros(())
    scale = pred.detach().std(unbiased=False).clamp(min=1.0)
    unit = pred / scale
    diff_p = unit.unsqueeze(0) - unit.unsqueeze(1)
    diff_y = y.unsqueeze(0) - y.unsqueeze(1)
    valid = diff_y.abs() > 1e-6
    if not bool(valid.any()):
        return pred.new_zeros(())
    return F.softplus(-diff_p * diff_y.sign())[valid].mean()


def masked_pairwise_rank_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_points: int = 256,
    date_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """RankNet: labelled pairs should keep the target order (within date if possible)."""
    dates = _expand_date_ids(date_ids, mask)
    sel = mask.bool()
    pred = mean[sel]
    y = target[sel]
    if dates is not None:
        keys = dates[sel]
        parts: list[torch.Tensor] = []
        for key in keys.unique():
            m = keys == key
            if int(m.sum()) < 2:
                continue
            parts.append(_ranknet(pred[m], y[m]))
        if parts:
            return torch.stack(parts).mean()
    n = int(pred.numel())
    if n < 2:
        return mean.new_zeros(())
    if n > max_points:
        idx = torch.randperm(n, device=pred.device)[:max_points]
        pred = pred[idx]
        y = y[idx]
    return _ranknet(pred, y)


def masked_pred_std_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Keep predicted vol close to labelled target vol."""
    sel = mask.bool()
    if int(sel.sum()) < 2:
        return mean.new_zeros(())
    pred = mean[sel]
    y = target[sel]
    return (pred.std(unbiased=False) - y.std(unbiased=False).detach()).pow(2)


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
    return {
        "ic": ic_raw,
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


def mean_cs_ic(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> float:
    """Mean Pearson IC across names on each date (the industry CS IC)."""
    return float(mean_cs_stats(pred, target, dates, min_names=min_names)["cs_ic"])


def mean_cs_stats(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> dict[str, float]:
    """Mean CS Pearson / Spearman, t-stat, and coverage over dates."""
    empty = {
        "cs_ic": float("nan"),
        "cs_ic_spearman": float("nan"),
        "cs_ic_tstat": float("nan"),
        "cs_n_dates": 0.0,
        "cs_mean_n": float("nan"),
    }
    if pred.size == 0 or dates.size != pred.size:
        return empty
    ics: list[float] = []
    spears: list[float] = []
    ns: list[float] = []
    for key in np.unique(dates):
        sel = dates == key
        n = int(sel.sum())
        if n < min_names:
            continue
        rho = _pearson(pred[sel], target[sel])
        sp = _spearman(pred[sel], target[sel])
        if np.isfinite(rho):
            ics.append(rho)
            spears.append(sp if np.isfinite(sp) else float("nan"))
            ns.append(float(n))
    if not ics:
        return empty
    arr = np.asarray(ics, dtype=np.float64)
    tstat = float("nan")
    if arr.size > 2:
        s = float(arr.std(ddof=1))
        if s > 1e-12:
            tstat = float(arr.mean() / (s / math.sqrt(arr.size)))
    spear = np.asarray(spears, dtype=np.float64)
    spear = spear[np.isfinite(spear)]
    return {
        "cs_ic": float(arr.mean()),
        "cs_ic_spearman": float(spear.mean()) if spear.size else float("nan"),
        "cs_ic_tstat": tstat,
        "cs_n_dates": float(arr.size),
        "cs_mean_n": float(np.mean(ns)),
    }


def selection_score(metrics: dict[str, float]) -> float:
    """Checkpoint / early-stop score: mean CS IC when it exists, else last-bar Pearson."""
    cs = metrics.get("cs_ic", float("nan"))
    if np.isfinite(cs):
        return float(cs)
    return float(metrics.get("ic", float("nan")))


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
    plateau_now = (
        plateau_evals > 0
        and n >= plateau_evals
        and n % plateau_evals == 0
    )
    if plateau_now:
        factor = min(1.0, max(1e-6, float(plateau_factor)))
        floor = max(0.0, float(min_scale))
        new_scale = max(floor, lr_scale * factor)
        if new_scale < lr_scale:
            return n, new_scale, False, True, (
                f"val IC plateau for {n} evals: lr scale "
                f"{lr_scale:.3g} -> {new_scale:.3g} (restoring best.pt, continuing)"
            )
        return n, lr_scale, False, False, (
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
    *,
    cs_min_names: int = 3,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    date_list: list[np.ndarray] = []
    weighted_loss = 0.0
    total_weight = 0.0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, mask, scale = batch[0], batch[1], batch[2], batch[3]
        date_ids = batch[4] if len(batch) > 4 else None
        x, y, mask, scale = (
            t.to(device, non_blocking=True) for t in (x, y, mask, scale)
        )
        if date_ids is not None:
            date_ids = date_ids.to(device, non_blocking=True)
        with autocast_context(device, train_cfg.precision):
            mean, log_sigma = model(x)
        mean = mean.float()
        loss = masked_loss(
            mean, log_sigma.float(), y, mask, train_cfg, date_ids=date_ids
        )
        weight = float(mask.to(mean.dtype).sum())
        if weight > 0 and math.isfinite(float(loss)):
            weighted_loss += float(loss) * weight
            total_weight += weight
        sel = mask.bool()
        preds.append(mean[sel].cpu().numpy())
        targets.append(y[sel].cpu().numpy())
        scales.append(scale[sel].cpu().numpy())
        if date_ids is not None:
            d = date_ids
            if d.dim() == 1 and d.size(0) == mask.size(0):
                d = d.unsqueeze(-1).expand_as(mask)
            date_list.append(d[sel].cpu().numpy())

    model.train()
    pred_np = np.concatenate(preds) if preds else np.empty(0)
    tgt_np = np.concatenate(targets) if targets else np.empty(0)
    scale_np = np.concatenate(scales) if scales else np.empty(0)
    metrics = compute_metrics(pred_np, tgt_np, scale_np, winsor=train_cfg.ic_winsor)
    if date_list:
        dates_np = np.concatenate(date_list)
        metrics.update(
            mean_cs_stats(pred_np, tgt_np, dates_np, min_names=int(cs_min_names))
        )
    metrics["select"] = selection_score(metrics)
    metrics["loss"] = weighted_loss / total_weight if total_weight > 0 else float("nan")
    return metrics


def _tag_skip_lr_mult(
    optimizer: torch.optim.Optimizer,
    model: ReturnForecaster,
    mult: float,
) -> None:
    """Split param groups so the linear skip can use a lower LR after ridge."""
    skip_ids = {id(p) for p in model.skip.parameters()}
    new_groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        base = {k: v for k, v in group.items() if k != "params"}
        rest = [p for p in group["params"] if id(p) not in skip_ids]
        skip = [p for p in group["params"] if id(p) in skip_ids]
        if rest:
            new_groups.append({**base, "params": rest, "lr_mult": 1.0})
        if skip:
            new_groups.append({**base, "params": skip, "lr_mult": float(mult)})
    optimizer.param_groups = new_groups


def apply_ridge_skip(
    model: ReturnForecaster,
    bundle: dict[str, Any],
    train_cfg: ForecastTrainConfig,
    device: torch.device,
) -> float:
    """Copy a train-only ridge readout into ``model.skip``. Returns in-sample IC."""
    if (not model.config.linear_skip) or float(train_cfg.ridge_skip) <= 0:
        return float("nan")
    weights, bias, ic = fit_ridge_readout(
        bundle["train_symbols"],
        bundle["feature_mean"],
        bundle["feature_std"],
        ridge=float(train_cfg.ridge_skip),
        cs_demean=bool(train_cfg.ridge_cs_demean),
        min_names=int(bundle.get("cs_min_names", 8)),
    )
    with torch.no_grad():
        model.skip.weight.copy_(
            torch.from_numpy(weights).to(device=device, dtype=model.skip.weight.dtype).unsqueeze(0)
        )
        model.skip.bias.copy_(
            torch.tensor([bias], device=device, dtype=model.skip.bias.dtype)
        )
    if train_cfg.freeze_skip:
        model.skip.weight.requires_grad_(False)
        model.skip.bias.requires_grad_(False)
    return ic


def _fmt(metrics: dict[str, float]) -> str:
    spearman = metrics.get("ic_spearman", float("nan"))
    raw = metrics.get("ic_raw", metrics.get("ic", float("nan")))
    cs_sp = metrics.get("cs_ic_spearman", float("nan"))
    cs_t = metrics.get("cs_ic_tstat", float("nan"))
    cs_n = metrics.get("cs_n_dates", float("nan"))
    return (
        f"loss={metrics['loss']:.5f} ic={metrics['ic']:+.4f} "
        f"spearman={spearman:+.4f} raw={raw:+.4f} "
        f"cs_ic={metrics.get('cs_ic', float('nan')):+.4f} "
        f"cs_sp={cs_sp:+.4f} cs_t={cs_t:+.2f} cs_dates={int(cs_n) if np.isfinite(cs_n) else 0} "
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
            f"ridge_skip={train_cfg.ridge_skip} "
            f"heteroscedastic={model_cfg.heteroscedastic}"
        )
        autocast_context(device, train_cfg.precision, log_fn=log_fn)

    skip_ic = apply_ridge_skip(model, bundle, train_cfg, device)
    if log_fn and np.isfinite(skip_ic):
        kind = "CS" if train_cfg.ridge_cs_demean else "pooled"
        log_fn(f"ridge skip in-sample {kind} IC={skip_ic:+.4f} (train labelled bars)")
    if log_fn and not bundle.get("cross_section"):
        log_fn(
            "NOTE: cross-section dataset is off "
            f"(need >= {data_cfg.cross_section_min_names} trading names). "
            "Pooled last-bar Pearson is not the CS residual estimand."
        )

    cs_eval = {"cs_min_names": int(data_cfg.cross_section_min_names)}

    loader_kwargs = {
        **dataloader_kwargs(device, train_cfg.num_workers),
        "collate_fn": collate_forecast,
    }
    drop_last = (
        (not train_cfg.skip_only)
        and len(datasets["train"]) >= 2 * max(1, train_cfg.batch_size)
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=train_cfg.batch_size,
        shuffle=not train_cfg.skip_only,
        drop_last=drop_last,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        datasets["train"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    steps_per_epoch = max(1, len(train_loader))
    if train_cfg.max_steps is None:
        total_steps = steps_per_epoch * train_cfg.epochs
    else:
        total_steps = int(train_cfg.max_steps)

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

    t0 = time.perf_counter()
    if log_fn:
        log_fn(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")
    skip_only_train = evaluate(model, train_eval_loader, device, train_cfg, **cs_eval)
    skip_only_val = evaluate(model, val_loader, device, train_cfg, **cs_eval)
    skip_only_test = evaluate(model, test_loader, device, train_cfg, **cs_eval)
    if log_fn:
        log_fn(f"  skip-only train: {_fmt(skip_only_train)}")
        log_fn(f"  skip-only val: {_fmt(skip_only_val)}")
        log_fn(f"  skip-only test: {_fmt(skip_only_test)}")

    if train_cfg.skip_only:
        save(ckpt_dir / "best.pt", 0, skip_only_val)
        save(ckpt_dir / "last.pt", 0, skip_only_val)
        summary = {
            "best_val_ic": selection_score(skip_only_val),
            "best_val_cs_ic": skip_only_val.get("cs_ic", float("nan")),
            "best_step": 0,
            "history": [],
            "test": skip_only_test,
            "skip_only_train": skip_only_train,
            "skip_only_val": skip_only_val,
            "skip_only_test": skip_only_test,
            "interrupted": False,
            "last_step": 0,
            "n_params": n_params,
            "device": str(device),
            "elapsed_sec": time.perf_counter() - t0,
            "symbols": bundle["meta"],
            "cross_section": bundle.get("cross_section", False),
            "skip_only": True,
        }
        (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        if log_fn:
            log_fn(f"skip-only run wrote {ckpt_dir / 'best.pt'}")
        return summary

    if not train_cfg.skip_only:
        require_nonempty_loader(train_loader, "train")
    if total_steps < 1:
        raise RuntimeError("total_steps is 0; check epochs, max_steps, and dataset size")
    optimizer = build_optimizer(
        model, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    _tag_skip_lr_mult(optimizer, model, train_cfg.skip_lr_mult)
    use_scaler = train_cfg.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    history: list[dict[str, Any]] = []
    best_ic = -float("inf")
    best_step = -1
    best_val_metrics: dict[str, float] = {}
    evals_without_gain = 0
    lr_scale = 1.0
    data_iter = cycle_loader(train_loader)
    running: list[float] = []
    model.train()

    last_completed = 0
    interrupted = False
    try:
        for step in range(total_steps):
            lr = lr_at(step, total_steps, train_cfg) * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = lr * float(group.get("lr_mult", 1.0))

            batch = next(data_iter)
            x, y, mask = batch[0], batch[1], batch[2]
            date_ids = batch[4] if len(batch) > 4 else None
            # non_blocking pairs with pin_memory: the H2D copy overlaps the
            # optimizer bookkeeping instead of stalling the step.
            x, y, mask = (
                t.to(device, non_blocking=True) for t in (x, y, mask)
            )
            if date_ids is not None:
                date_ids = date_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, train_cfg.precision):
                mean, log_sigma = model(x)
            loss = masked_loss(
                mean.float(), log_sigma.float(), y, mask, train_cfg, date_ids=date_ids
            )
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
                val = evaluate(model, val_loader, device, train_cfg, **cs_eval)
                val["step"] = float(step + 1)
                history.append(val)
                if log_fn:
                    log_fn(f"  eval step {step + 1}: {_fmt(val)}")
                    train_snap = evaluate(
                        model,
                        train_loader,
                        device,
                        train_cfg,
                        max_batches=8,
                        **cs_eval,
                    )
                    log_fn(f"  train sample: {_fmt(train_snap)}")
                save(ckpt_dir / "last.pt", step + 1, val)

                ic = selection_score(val)
                improved = bool(np.isfinite(ic) and ic > best_ic)
                if improved:
                    best_ic, best_step = ic, step + 1
                    best_val_metrics = {
                        k: float(v)
                        for k, v in val.items()
                        if isinstance(v, (int, float))
                    }
                    save(ckpt_dir / "best.pt", step + 1, val)
                    if log_fn:
                        label = "val_cs_ic" if np.isfinite(val.get("cs_ic", float("nan"))) else "val_ic"
                        log_fn(f"  new best {label}={ic:+.4f} -> {ckpt_dir / 'best.pt'}")
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
        test = evaluate(model, test_loader, device, train_cfg, **cs_eval)
        if log_fn:
            if best_path.exists():
                log_fn(f"TEST (best step {best_step}): {_fmt(test)}")
            elif last_path.exists():
                log_fn(f"TEST (last checkpoint; no best.pt was saved): {_fmt(test)}")
            else:
                log_fn(f"TEST (in-memory weights; no checkpoint saved): {_fmt(test)}")
            log_fn(f"TEST skip-only (ridge, frozen): {_fmt(skip_only_test)}")

    summary = {
        "best_val_ic": best_ic,
        "best_val_cs_ic": best_val_metrics.get("cs_ic", float("nan")),
        "best_step": best_step,
        "history": history,
        "test": test,
        "skip_only_train": skip_only_train,
        "skip_only_val": skip_only_val,
        "skip_only_test": skip_only_test,
        "interrupted": interrupted,
        "last_step": last_completed,
        "n_params": n_params,
        "device": str(device),
        "elapsed_sec": time.perf_counter() - t0,
        "symbols": bundle["meta"],
        "cross_section": bundle.get("cross_section", False),
        "skip_only": False,
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
    g.add_argument(
        "--allow-mixed-prices",
        action="store_true",
        help="do not fail when weekly lag-1 autocorr looks like mixed adjusted/raw closes",
    )
    g.add_argument(
        "--no-residual-target",
        action="store_true",
        help="predict raw next-bar return instead of trailing-beta residual vs SPY",
    )
    g.add_argument(
        "--no-eval-last-bar",
        action="store_true",
        help="score every labelled bar instead of unique last-bar dates",
    )
    g.add_argument(
        "--no-global-split",
        action="store_true",
        help="split each symbol on its own session count (legacy)",
    )
    g.add_argument(
        "--universe",
        default="",
        choices=("", "liquid"),
        help="restrict parquets to the train-era-locked liquid list (+ SPY)",
    )
    g.add_argument(
        "--no-cs-zscore",
        action="store_true",
        help="leave cs_* features at 0 (ablation)",
    )
    g.add_argument(
        "--cs-min-names",
        type=int,
        default=None,
        help="min names per date for CS batches / CS IC (default: 30)",
    )
    g.add_argument(
        "--no-sector-residual",
        action="store_true",
        help="residualize vs SPY only (skip mapped sector ETFs)",
    )
    g.add_argument(
        "--no-equities-only",
        action="store_true",
        help="keep index/sector/macro ETFs in the trading book",
    )
    g.add_argument(
        "--train-from",
        default=d.train_from,
        help="drop train labels before this date (YYYY-MM-DD); val/test cuts unchanged",
    )
    g.add_argument(
        "--no-train-from",
        action="store_true",
        help="use every train-session label (no 1999 floor)",
    )

    g = p.add_argument_group("model")
    g.add_argument("--d-model", type=int, default=None)
    g.add_argument("--n-layer", type=int, default=None)
    g.add_argument("--d-state", type=int, default=None)
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
    g.add_argument(
        "--location-loss-weight",
        type=float,
        default=t.location_loss_weight,
        help="weight on Huber/MSE (scale). Keep below --ic-loss-weight",
    )
    g.add_argument(
        "--sign-loss-weight",
        type=float,
        default=t.sign_loss_weight,
        help="weight on sign BCE for moves larger than sign_min_abs",
    )
    g.add_argument(
        "--rank-loss-weight",
        type=float,
        default=t.rank_loss_weight,
        help="weight on pairwise RankNet over labelled bars in the batch",
    )
    g.add_argument(
        "--ridge-skip",
        type=float,
        default=t.ridge_skip,
        help="ridge lambda for the linear skip init (0 = Xavier, no closed-form fit)",
    )
    g.add_argument(
        "--no-freeze-skip",
        action="store_true",
        help="let AdamW keep updating the ridge skip (default: freeze after init)",
    )
    g.add_argument(
        "--no-ridge-cs-demean",
        action="store_true",
        help="fit pooled time-series ridge instead of date-demeaned CS ridge",
    )
    g.add_argument(
        "--no-cs-center",
        action="store_true",
        help="do not subtract within-date means in the location loss",
    )
    g.add_argument(
        "--skip-only",
        action="store_true",
        help="fit the ridge skip, log last-bar train/val/test IC, write best.pt, exit",
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
        help="stop after N evals with no val-IC gain (0 = never; default 24)",
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
    m = ForecastModelConfig()
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
        eval_last_bar=not args.no_eval_last_bar,
        global_calendar_split=not args.no_global_split,
        residual_target=not args.no_residual_target,
        allow_mixed_prices=args.allow_mixed_prices,
        universe=args.universe,
        cs_zscore=not args.no_cs_zscore,
        cross_section_min_names=(
            DataConfig().cross_section_min_names
            if args.cs_min_names is None
            else args.cs_min_names
        ),
        sector_residual=not args.no_sector_residual,
        equities_only=not args.no_equities_only,
        train_from="" if args.no_train_from else str(args.train_from or ""),
    )
    model_cfg = ForecastModelConfig(
        n_features=len(FEATURE_NAMES),
        d_model=ssm.get("d_model", m.d_model) if args.d_model is None else args.d_model,
        n_layer=ssm.get("n_layer", m.n_layer) if args.n_layer is None else args.n_layer,
        d_state=ssm.get("d_state", m.d_state) if args.d_state is None else args.d_state,
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
        location_loss_weight=args.location_loss_weight,
        sign_loss_weight=args.sign_loss_weight,
        rank_loss_weight=args.rank_loss_weight,
        ridge_skip=args.ridge_skip,
        freeze_skip=not args.no_freeze_skip,
        ridge_cs_demean=not args.no_ridge_cs_demean,
        cs_center_loss=not args.no_cs_center,
        skip_only=args.skip_only,
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
