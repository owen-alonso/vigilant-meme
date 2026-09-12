"""VAL-gated Dynamic A residual on the overnight skip.

Default overnight accuracy is still the ridge CS skip (no Mamba). This
module is the path that actually trains the Dynamic A hypernet and scores
last-bar residual blends for overnight-up / book-aligned sleeves.

Trace
-----
``ForecastModelConfig.dynamic_weights`` / ``dynamic_A``
  -> ``MambaConfig`` -> ``MambaBlock.controller`` (small MLP / hypernet)
  -> ``scale = 1 + strength * tanh(delta_A)``
  -> ``A_t = A_base * scale`` (token/bar-dependent timescales)

``ReturnForecaster`` reuses ``MambaLayer``. When Dynamic A is on, the
forecast mixer ``out_proj`` and head are small-Xavier (not zeros) so the
controller gets gradients on step 0. The LM path is unchanged.

Promotion is locked-VAL only. TEST is report-only. Live q20 is unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from forecast.dynamic_health import (
    controller_is_present,
    controller_param_grad_norm,
    format_dynamic_health_ascii,
    interpret_scale_reports,
)

BLEND_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_COVER = 0.05
VAL_SLEEVE_LIFT_PP = 0.50
VAL_POOLED_LIFT_PP = 1.00
TARGET_UP_PCT = 60.0
TINY_D_MODEL = 32
TINY_N_LAYER = 1
TINY_D_STATE = 8
DEFAULT_STEPS = 80
VAL_CS_LIFT = 0.005


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def apply_residual_pred(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Replace residual ``pred`` and implied overnight gap. y / r_on stay."""
    out = df.copy()
    p = np.asarray(pred, dtype=np.float64)
    if p.shape[0] != len(out):
        raise ValueError(f"pred length {p.shape[0]} != frame {len(out)}")
    out["pred"] = p
    scale = out["scale"].to_numpy(dtype=np.float64)
    out["pred_r"] = p * scale
    close = out["close"].to_numpy(dtype=np.float64)
    pr = out["pred_r"].to_numpy(dtype=np.float64)
    out["implied_open"] = close * np.exp(pr)
    if "r_on" in out.columns and "y" in out.columns:
        hedge = out["r_on"].to_numpy(dtype=np.float64) - (
            out["y"].to_numpy(dtype=np.float64) * scale
        )
        out["implied_open_given_hedge"] = close * np.exp(pr + hedge)
    return out


def blend_residual(skip: np.ndarray, encoder: np.ndarray, alpha: float) -> np.ndarray:
    """``skip + alpha * (encoder - skip)``. alpha=0 is skip; alpha=1 is encoder."""
    s = np.asarray(skip, dtype=np.float64)
    e = np.asarray(encoder, dtype=np.float64)
    a = float(alpha)
    return s + a * (e - s)


def decide_dynamic_a_cs_promote(
    *,
    dyn_val: float,
    skip_val: float,
    dyn_test: float = float("nan"),
    skip_test: float = float("nan"),
    val_lift: float = VAL_CS_LIFT,
) -> dict[str, Any]:
    """VAL-only CS-IC gate for ``cs_overnight --try-dynamic-a``. TEST never gates."""
    dv = _as_float(dyn_val)
    sv = _as_float(skip_val)
    dt = _as_float(dyn_test)
    st = _as_float(skip_test)
    lift = dv - sv if math.isfinite(dv) and math.isfinite(sv) else float("nan")
    test_lift = dt - st if math.isfinite(dt) and math.isfinite(st) else float("nan")
    promote = bool(math.isfinite(lift) and lift >= float(val_lift))
    if promote:
        reason = (
            f"PROMOTE Dynamic A encoder: VAL CS IC {dv:+.4f} vs skip {sv:+.4f} "
            f"(lift {lift:+.4f} >= {float(val_lift):+.3f}). TEST lift "
            f"{test_lift:+.4f} is report-only."
        )
    else:
        reason = (
            f"NO PROMOTE Dynamic A: VAL CS IC {dv:+.4f} vs skip {sv:+.4f} "
            f"(lift {lift:+.4f}; need +{float(val_lift):.3f}). TEST does not gate."
        )
    return {
        "promote": promote,
        "gated_on": "val",
        "val_lift": lift,
        "test_lift": test_lift,
        "dyn_val": dv,
        "skip_val": sv,
        "dyn_test": dt,
        "skip_test": st,
        "val_lift_floor": float(val_lift),
        "reason": reason,
    }


def decide_dynamic_a_overnight_promote(
    *,
    val_on: Mapping[str, Any],
    val_off: Mapping[str, Any],
    val_skip: Mapping[str, Any],
    min_cover: float = MIN_COVER,
    sleeve_lift_pp: float = VAL_SLEEVE_LIFT_PP,
    pooled_lift_pp: float = VAL_POOLED_LIFT_PP,
) -> dict[str, Any]:
    """VAL-only overnight-up gate. TEST never enters.

    Promote the Dynamic A blend if locked VAL book-aligned sleeve overnight-up
    beats the skip sleeve by ``sleeve_lift_pp`` with cover >= ``min_cover``,
    or pooled TS dir beats skip by ``pooled_lift_pp``. The off ablation is
    reported so a skip-only residual encoder cannot be sold as the hypernet.
    """
    def _sleeve(d: Mapping[str, Any]) -> float:
        return _as_float(d.get("sleeve_up_pct"), _as_float(d.get("val_sleeve_up")))

    def _cover(d: Mapping[str, Any]) -> float:
        return _as_float(d.get("sleeve_coverage"), _as_float(d.get("val_sleeve_cover")))

    def _dir(d: Mapping[str, Any]) -> float:
        return _as_float(d.get("dir_pct"), _as_float(d.get("val_dir")))

    on_sl = _sleeve(val_on or {})
    off_sl = _sleeve(val_off or {})
    skip_sl = _sleeve(val_skip or {})
    on_cov = _cover(val_on or {})
    on_dir = _dir(val_on or {})
    skip_dir = _dir(val_skip or {})
    sleeve_lift = (
        on_sl - skip_sl if math.isfinite(on_sl) and math.isfinite(skip_sl) else float("nan")
    )
    pooled_lift = (
        on_dir - skip_dir if math.isfinite(on_dir) and math.isfinite(skip_dir) else float("nan")
    )
    vs_off = (
        on_sl - off_sl if math.isfinite(on_sl) and math.isfinite(off_sl) else float("nan")
    )
    cover_ok = bool(math.isfinite(on_cov) and on_cov >= float(min_cover))
    sleeve_ok = bool(
        cover_ok and math.isfinite(sleeve_lift) and sleeve_lift >= float(sleeve_lift_pp)
    )
    pooled_ok = bool(math.isfinite(pooled_lift) and pooled_lift >= float(pooled_lift_pp))
    beats_off = bool(math.isfinite(vs_off) and vs_off >= 0.0)
    promote = bool(sleeve_ok or pooled_ok)
    reached_60 = bool(math.isfinite(on_sl) and on_sl >= TARGET_UP_PCT and cover_ok)
    if promote and sleeve_ok:
        reason = (
            f"PROMOTE Dynamic A overnight blend: VAL sleeve up {on_sl:.2f}% vs "
            f"skip {skip_sl:.2f}% (lift {sleeve_lift:+.2f} pp, cover "
            f"{100.0 * on_cov:.1f}% >= {100.0 * float(min_cover):.0f}%). "
            "TEST report-only. Live q20 unchanged."
        )
    elif promote and pooled_ok:
        reason = (
            f"PROMOTE Dynamic A overnight blend: VAL pooled dir {on_dir:.2f}% vs "
            f"skip {skip_dir:.2f}% (lift {pooled_lift:+.2f} pp). "
            "TEST report-only. Live q20 unchanged."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE Dynamic A: VAL sleeve cover {100.0 * on_cov:.1f}% "
            f"< {100.0 * float(min_cover):.0f}% (do not shrink to a handful of names)."
        )
    else:
        reason = (
            f"NO PROMOTE Dynamic A: VAL sleeve {on_sl:.2f}% vs skip {skip_sl:.2f}% "
            f"(lift {sleeve_lift:+.2f} pp, need +{float(sleeve_lift_pp):.2f})  "
            f"pooled dir lift {pooled_lift:+.2f} pp (need +{float(pooled_lift_pp):.2f}). "
            "TEST does not gate."
        )
    return {
        "promote_dynamic_a": promote,
        "gated_on": "val",
        "reason": reason,
        "sleeve_ok": sleeve_ok,
        "pooled_ok": pooled_ok,
        "cover_ok": cover_ok,
        "beats_off_ablation": beats_off,
        "reached_60": reached_60,
        "val_sleeve_up": on_sl,
        "val_skip_sleeve_up": skip_sl,
        "val_off_sleeve_up": off_sl,
        "val_sleeve_lift_pp": sleeve_lift,
        "val_vs_off_pp": vs_off,
        "val_pooled_dir": on_dir,
        "val_skip_dir": skip_dir,
        "val_pooled_lift_pp": pooled_lift,
        "val_cover": on_cov,
        "min_cover": float(min_cover),
        "sleeve_lift_pp": float(sleeve_lift_pp),
        "pooled_lift_pp": float(pooled_lift_pp),
        "target_up_pct": TARGET_UP_PCT,
    }


def format_dynamic_a_block(payload: Mapping[str, Any]) -> str:
    """Human report for the VAL-gated Dynamic A overnight path. ASCII only."""
    blob = dict(payload.get("dynamic_a") or payload or {})
    if not blob:
        return ""
    promo = dict(blob.get("promotion") or {})
    on = dict(blob.get("on") or {})
    off = dict(blob.get("off") or {})
    skip = dict(blob.get("skip") or {})
    yes = bool(promo.get("promote_dynamic_a"))
    health_i = dict(on.get("health_init") or {})
    health_f = dict(on.get("health_final") or {})
    lines = [
        f"PROMOTE DYNAMIC A OVERNIGHT? {'YES' if yes else 'NO'}",
        "  Tiny frozen-skip Mamba (d_model=32 n_layer=1) with Dynamic A on vs off. "
        "Last-bar residual blend = skip + alpha*(encoder-skip). "
        "VAL gates; TEST report-only. Live q20 unchanged. "
        "Cover floor 5% -- no handful-of-names cheat.",
        f"  chosen alpha={_as_float(blob.get('alpha')):.2f}  "
        f"steps={int(_as_float(blob.get('max_steps'), 0.0))}  "
        f"aligned={int(_as_float(blob.get('n_aligned_val'), 0.0))} VAL rows",
        f"  health init  {format_dynamic_health_ascii(health_i)}",
        f"  health final {format_dynamic_health_ascii(health_f)}",
        f"  controller present={bool(on.get('controller_present'))}  "
        f"step0_grad={_as_float(on.get('controller_grad_step0')):.3e}  "
        f"final_grad={_as_float(on.get('controller_grad_final')):.3e}",
        f"  VAL skip   sleeve {_as_float(skip.get('val_sleeve_up')):.2f}%  "
        f"cover {100.0 * _as_float(skip.get('val_sleeve_cover')):.1f}%  "
        f"dir {_as_float(skip.get('val_dir')):.2f}%  "
        f"CS IC {_as_float(skip.get('val_cs_ic')):+.4f}",
        f"  VAL off    sleeve {_as_float(off.get('val_sleeve_up')):.2f}%  "
        f"cover {100.0 * _as_float(off.get('val_sleeve_cover')):.1f}%  "
        f"dir {_as_float(off.get('val_dir')):.2f}%  "
        f"(ablation: same recipe, dynamic_weights=0)",
        f"  VAL on     sleeve {_as_float(on.get('val_sleeve_up')):.2f}%  "
        f"cover {100.0 * _as_float(on.get('val_sleeve_cover')):.1f}%  "
        f"dir {_as_float(on.get('val_dir')):.2f}%  "
        f"CS IC {_as_float(on.get('val_cs_ic')):+.4f}",
        f"  VAL lift vs skip sleeve {_as_float(promo.get('val_sleeve_lift_pp')):+.2f} pp  "
        f"vs off {_as_float(promo.get('val_vs_off_pp')):+.2f} pp  "
        f"pooled dir {_as_float(promo.get('val_pooled_lift_pp')):+.2f} pp  "
        f"reached_60={bool(promo.get('reached_60'))}",
        f"  TEST on    sleeve {_as_float(on.get('test_sleeve_up')):.2f}%  "
        f"cover {100.0 * _as_float(on.get('test_sleeve_cover')):.1f}%  "
        f"dir {_as_float(on.get('test_dir')):.2f}%  "
        f"(report-only; do not retarget)",
        f"  TEST skip  sleeve {_as_float(skip.get('test_sleeve_up')):.2f}%  "
        f"dir {_as_float(skip.get('test_dir')):.2f}%",
        f"  {promo.get('reason') or 'no decision'}",
    ]
    return "\n".join(lines)


def _tiny_model_cfg(
    n_features: int,
    *,
    dynamic_weights: bool,
    strength: float = 0.1,
) -> Any:
    from forecast.config import ForecastModelConfig

    return ForecastModelConfig(
        n_features=int(n_features),
        d_model=TINY_D_MODEL,
        n_layer=TINY_N_LAYER,
        d_state=TINY_D_STATE,
        expand=2,
        dropout=0.0,
        linear_skip=True,
        dynamic_weights=bool(dynamic_weights),
        dynamic_A=True,
        dynamic_strength=float(strength),
    )


def _tiny_train_cfg(max_steps: int, ckpt_dir: str) -> Any:
    from forecast.config import ForecastTrainConfig

    steps = max(1, int(max_steps))
    return ForecastTrainConfig(
        batch_size=8,
        epochs=1,
        max_steps=steps,
        lr=1e-3,
        checkpoint_dir=str(ckpt_dir),
        skip_only=False,
        ridge_skip=10.0,
        freeze_skip=True,
        ridge_cs_demean=True,
        ridge_rank_target=True,
        ridge_feat_winsor=3.0,
        ridge_features="no_long_ts",
        precision="fp32",
        early_stop_evals=8,
        eval_interval=max(4, steps // 2),
        log_interval=max(1, steps // 4),
        ic_loss_weight=2.0,
        rank_loss_weight=1.0,
        eval_train_split=False,
        seed=42,
    )


def _predict_last_bars(model: Any, loader: Any, symbols: list[Any], device: Any) -> pd.DataFrame:
    import torch

    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x, _y, _mask, _scale, dates, syms = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[5],
            )
            x = x.to(device)
            mean, _log_sigma = model(x)
            pred = mean[:, -1].detach().float().cpu().numpy()
            d = dates.detach().cpu().numpy().reshape(-1)
            s = syms.detach().cpu().numpy().reshape(-1)
            n = min(pred.shape[0], d.shape[0], s.shape[0])
            for i in range(n):
                si = int(s[i])
                if si < 0 or si >= len(symbols):
                    continue
                rows.append(
                    {
                        "symbol": str(symbols[si].symbol),
                        "date": int(d[i]),
                        "enc_pred": float(pred[i]),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "enc_pred"])
    return pd.DataFrame(rows)


def align_encoder_pred(
    frame: pd.DataFrame,
    enc: pd.DataFrame,
    *,
    fill: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Map encoder last-bar residual onto an accuracy frame. Fill misses with skip."""
    skip = (
        frame["pred"].to_numpy(dtype=np.float64)
        if fill is None
        else np.asarray(fill, dtype=np.float64)
    )
    if frame.empty:
        return skip, 0
    if enc is None or enc.empty:
        return skip.copy(), 0
    key = frame[["symbol", "date"]].merge(enc, on=["symbol", "date"], how="left")
    pred = key["enc_pred"].to_numpy(dtype=np.float64)
    missing = ~np.isfinite(pred)
    n_hit = int((~missing).sum())
    pred[missing] = skip[missing]
    return pred, n_hit


def _probe_health(model: Any, loader: Any, device: Any, *, after_training: bool) -> dict[str, Any]:
    import torch

    cfg = getattr(model, "config", None)
    strength = float(getattr(cfg, "dynamic_strength", 0.1) or 0.1)
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return interpret_scale_reports(
            [],
            strength=strength,
            after_training=after_training,
        )
    x = batch[0].to(device)
    with torch.no_grad():
        model(x)
    return interpret_scale_reports(
        model.collect_dynamic_diagnostics(),
        strength=strength,
        after_training=after_training,
    )


def train_tiny_overnight_encoder(
    data_dir: str,
    universe: str,
    *,
    dynamic_weights: bool,
    max_steps: int = DEFAULT_STEPS,
    ckpt_dir: str | Path | None = None,
    device: Any | None = None,
    log_fn: Any | None = None,
    strength: float = 0.1,
) -> dict[str, Any]:
    """Train a tiny frozen-skip overnight encoder. Returns last-bar preds + health."""
    import torch
    from torch.utils.data import DataLoader

    from forecast.accuracy import overnight_skip_data_config
    from forecast.data import FEATURE_NAMES, build_datasets, collate_forecast
    from forecast.model import ReturnForecaster
    from forecast.training import apply_ridge_skip, evaluate, masked_loss
    from mamba_lm.training_utils import (
        autocast_context,
        dataloader_kwargs,
        grads_finite,
        select_device,
        set_seed,
    )

    device = device or select_device()
    data_cfg = overnight_skip_data_config(data_dir, universe)
    ckpt = Path(ckpt_dir or "/tmp/overnight_dynamic_a")
    ckpt.mkdir(parents=True, exist_ok=True)
    train_cfg = _tiny_train_cfg(max_steps, str(ckpt))
    set_seed(train_cfg.seed)
    bundle = build_datasets(data_cfg, log_fn=None)
    n_features = len(bundle.get("feature_names") or FEATURE_NAMES)
    model_cfg = _tiny_model_cfg(
        n_features, dynamic_weights=dynamic_weights, strength=strength
    )
    model = ReturnForecaster(model_cfg).to(device)
    apply_ridge_skip(model, bundle, train_cfg, device)

    datasets = bundle["datasets"]
    loader_kwargs = {
        **dataloader_kwargs(device, 0),
        "collate_fn": collate_forecast,
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_size=train_cfg.batch_size,
        shuffle=True,
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
    health_init = _probe_health(model, val_loader, device, after_training=False)
    cs_eval = {"cs_min_names": int(data_cfg.cross_section_min_names)}
    skip_val = evaluate(model, val_loader, device, train_cfg, **cs_eval)
    skip_test = evaluate(model, test_loader, device, train_cfg, **cs_eval)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(train_cfg.lr), weight_decay=0.01)
    model.train()
    step0_grad = float("nan")
    last_grad = float("nan")
    best_ic = float("-inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    steps_done = 0
    it = iter(train_loader)

    def _next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(train_loader)
            return next(it)

    total = max(1, int(max_steps))
    for step in range(total):
        batch = _next_batch()
        x, y, mask, _scale = batch[0], batch[1], batch[2], batch[3]
        date_ids = batch[4] if len(batch) > 4 else None
        x, y, mask = (t.to(device) for t in (x, y, mask))
        if date_ids is not None:
            date_ids = date_ids.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_cfg.precision):
            mean, log_sigma = model(x)
            loss = masked_loss(
                mean.float(), log_sigma.float(), y, mask, train_cfg, date_ids=date_ids
            )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if not grads_finite(model):
            optimizer.zero_grad(set_to_none=True)
            continue
        gnorm = controller_param_grad_norm(model)
        if step == 0:
            step0_grad = gnorm
        last_grad = gnorm
        optimizer.step()
        steps_done = step + 1
        if (step + 1) % int(train_cfg.eval_interval) == 0 or step + 1 == total:
            val = evaluate(model, val_loader, device, train_cfg, **cs_eval)
            ic = float(val.get("cs_ic", val.get("ic", float("nan"))))
            if math.isfinite(ic) and ic > best_ic:
                best_ic = ic
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if log_fn:
                log_fn(
                    f"  dynamic_a {'on' if dynamic_weights else 'off'} "
                    f"step {step + 1}/{total} val_cs_ic={ic:+.4f} "
                    f"{format_dynamic_health_ascii(model.collect_dynamic_health(controller_grad_norm=gnorm))}"
                )
        elif log_fn and ((step + 1) % int(train_cfg.log_interval) == 0 or step == 0):
            log_fn(
                f"  dynamic_a {'on' if dynamic_weights else 'off'} "
                f"step {step + 1}/{total} loss={float(loss.detach()):.5f} "
                f"{format_dynamic_health_ascii(model.collect_dynamic_health(controller_grad_norm=gnorm))}"
            )

    model.load_state_dict(best_state)
    model.to(device)
    health_final = _probe_health(model, val_loader, device, after_training=True)
    health_final["controller_grad_norm"] = last_grad
    val_metrics = evaluate(model, val_loader, device, train_cfg, **cs_eval)
    test_metrics = evaluate(model, test_loader, device, train_cfg, **cs_eval)
    train_eval_loader = DataLoader(
        datasets["train"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    pred_train = _predict_last_bars(
        model, train_eval_loader, bundle["train_symbols"], device
    )
    pred_val = _predict_last_bars(model, val_loader, bundle["val_symbols"], device)
    pred_test = _predict_last_bars(model, test_loader, bundle["test_symbols"], device)
    return {
        "dynamic_weights": bool(dynamic_weights),
        "controller_present": controller_is_present(model),
        "health_init": health_init,
        "health_final": health_final,
        "controller_grad_step0": step0_grad,
        "controller_grad_final": last_grad,
        "best_val_cs_ic": float(val_metrics.get("cs_ic", best_ic)),
        "test_cs_ic": float(test_metrics.get("cs_ic", float("nan"))),
        "skip_only_val_cs_ic": float(skip_val.get("cs_ic", float("nan"))),
        "skip_only_test_cs_ic": float(skip_test.get("cs_ic", float("nan"))),
        "steps": steps_done,
        "device": str(device),
        "preds": {"train": pred_train, "val": pred_val, "test": pred_test},
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def _score_split_frames(
    frames: Mapping[str, pd.DataFrame],
    pred_by_split: Mapping[str, np.ndarray],
    *,
    min_names: int,
    e_q: float,
    e_tau: float,
) -> dict[str, Any]:
    from forecast.accuracy import (
        score_book_aligned_sleeve,
        score_eval_frame,
        slim_accuracy,
    )

    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        df = apply_residual_pred(frames[split], pred_by_split[split])
        slim = slim_accuracy(score_eval_frame(df, min_names=min_names))
        sleeve = score_book_aligned_sleeve(
            df, q=float(e_q), abs_tau=float(e_tau), min_names=min_names
        )
        out[split] = {
            **slim,
            "sleeve_up_pct": _as_float(sleeve.get("up_pct")),
            "sleeve_coverage": _as_float(sleeve.get("coverage")),
            "sleeve_excess_pp": _as_float(sleeve.get("excess_pp")),
            "sleeve_n": _as_float(sleeve.get("n"), 0.0),
            "sleeve": sleeve,
        }
    return out


def evaluate_overnight_dynamic_a(
    frames: Mapping[str, pd.DataFrame],
    data_dir: str,
    universe: str,
    *,
    min_names: int,
    book_aligned_fit: Mapping[str, Any] | None = None,
    max_steps: int = DEFAULT_STEPS,
    ckpt_dir: str | Path | None = None,
    log_fn: Any | None = None,
    strength: float = 0.1,
) -> dict[str, Any]:
    """Train Dynamic A on vs off, blend with skip, VAL-gate overnight-up."""
    from forecast.accuracy import fit_book_aligned_on_train

    chosen = dict((book_aligned_fit or {}).get("chosen") or {})
    e_q = _as_float(chosen.get("q"), 0.80)
    e_tau = _as_float(chosen.get("abs_tau"), 0.0)
    if log_fn:
        log_fn(
            "Dynamic A overnight residual: tiny encoder on vs off "
            f"(steps={int(max_steps)}, d_model={TINY_D_MODEL}, n_layer={TINY_N_LAYER})"
        )
    base = Path(ckpt_dir or "/tmp/overnight_dynamic_a")
    on = train_tiny_overnight_encoder(
        data_dir,
        universe,
        dynamic_weights=True,
        max_steps=max_steps,
        ckpt_dir=base / "on",
        log_fn=log_fn,
        strength=strength,
    )
    off = train_tiny_overnight_encoder(
        data_dir,
        universe,
        dynamic_weights=False,
        max_steps=max_steps,
        ckpt_dir=base / "off",
        log_fn=log_fn,
        strength=strength,
    )

    skip_pred = {
        split: frames[split]["pred"].to_numpy(dtype=np.float64) for split in frames
    }
    aligned = {}
    n_hits = {}
    for tag, blob in (("on", on), ("off", off)):
        aligned[tag] = {}
        n_hits[tag] = {}
        for split in ("train", "val", "test"):
            pred, n_hit = align_encoder_pred(
                frames[split], blob["preds"][split], fill=skip_pred[split]
            )
            aligned[tag][split] = pred
            n_hits[tag][split] = n_hit

    skip_scores = _score_split_frames(
        frames, skip_pred, min_names=min_names, e_q=e_q, e_tau=e_tau
    )
    # VAL-only alpha pick on the ON encoder (TRAIN fit of E q is reused).
    best_alpha = 1.0
    best_key = (-1e18, -1e18)
    alpha_rows: list[dict[str, Any]] = []
    for alpha in BLEND_ALPHAS:
        blended = {
            split: blend_residual(skip_pred[split], aligned["on"][split], alpha)
            for split in ("train", "val", "test")
        }
        # Re-fit E sleeve on TRAIN blended ranks; score VAL.
        tr_df = apply_residual_pred(frames["train"], blended["train"])
        refit = fit_book_aligned_on_train(tr_df, min_names=min_names)
        q = _as_float((refit.get("chosen") or {}).get("q"), e_q)
        tau = _as_float((refit.get("chosen") or {}).get("abs_tau"), e_tau)
        scored = _score_split_frames(
            frames, blended, min_names=min_names, e_q=q, e_tau=tau
        )
        va = scored["val"]
        cover = _as_float(va.get("sleeve_coverage"))
        if math.isfinite(cover) and cover < MIN_COVER:
            alpha_rows.append(
                {
                    "alpha": float(alpha),
                    "val_sleeve_up": _as_float(va.get("sleeve_up_pct")),
                    "val_cover": cover,
                    "rejected_cover": True,
                    "q": q,
                    "abs_tau": tau,
                }
            )
            continue
        key = (
            _as_float(va.get("sleeve_up_pct"), -1e18),
            _as_float(va.get("dir_pct"), -1e18),
        )
        row = {
            "alpha": float(alpha),
            "val_sleeve_up": _as_float(va.get("sleeve_up_pct")),
            "val_cover": cover,
            "val_dir": _as_float(va.get("dir_pct")),
            "rejected_cover": False,
            "q": q,
            "abs_tau": tau,
            "scored": scored,
        }
        alpha_rows.append(row)
        if key > best_key:
            best_key = key
            best_alpha = float(alpha)

    chosen_row = next((r for r in alpha_rows if r.get("alpha") == best_alpha), None)
    if chosen_row and chosen_row.get("scored"):
        on_scores = chosen_row["scored"]
        use_q = _as_float(chosen_row.get("q"), e_q)
        use_tau = _as_float(chosen_row.get("abs_tau"), e_tau)
    else:
        blended = {
            split: blend_residual(skip_pred[split], aligned["on"][split], 1.0)
            for split in ("train", "val", "test")
        }
        on_scores = _score_split_frames(
            frames, blended, min_names=min_names, e_q=e_q, e_tau=e_tau
        )
        use_q, use_tau = e_q, e_tau
        best_alpha = 1.0

    off_blended = {
        split: blend_residual(skip_pred[split], aligned["off"][split], best_alpha)
        for split in ("train", "val", "test")
    }
    off_scores = _score_split_frames(
        frames, off_blended, min_names=min_names, e_q=use_q, e_tau=use_tau
    )

    def _pack(scores: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        packed = {
            "val_sleeve_up": _as_float((scores.get("val") or {}).get("sleeve_up_pct")),
            "val_sleeve_cover": _as_float((scores.get("val") or {}).get("sleeve_coverage")),
            "val_dir": _as_float((scores.get("val") or {}).get("dir_pct")),
            "val_cs_ic": _as_float((scores.get("val") or {}).get("cs_ic")),
            "val_excess_pp": _as_float((scores.get("val") or {}).get("excess_pp")),
            "test_sleeve_up": _as_float((scores.get("test") or {}).get("sleeve_up_pct")),
            "test_sleeve_cover": _as_float((scores.get("test") or {}).get("sleeve_coverage")),
            "test_dir": _as_float((scores.get("test") or {}).get("dir_pct")),
            "test_cs_ic": _as_float((scores.get("test") or {}).get("cs_ic")),
            "train_sleeve_up": _as_float((scores.get("train") or {}).get("sleeve_up_pct")),
        }
        if extra:
            packed.update(extra)
        return packed

    on_pack = _pack(
        on_scores,
        {
            "health_init": on.get("health_init"),
            "health_final": on.get("health_final"),
            "controller_present": on.get("controller_present"),
            "controller_grad_step0": on.get("controller_grad_step0"),
            "controller_grad_final": on.get("controller_grad_final"),
            "best_val_cs_ic": on.get("best_val_cs_ic"),
            "encoder_test_cs_ic": on.get("test_cs_ic"),
        },
    )
    off_pack = _pack(
        off_scores,
        {
            "controller_present": off.get("controller_present"),
            "best_val_cs_ic": off.get("best_val_cs_ic"),
        },
    )
    skip_pack = _pack(skip_scores)
    promotion = decide_dynamic_a_overnight_promote(
        val_on=on_pack,
        val_off=off_pack,
        val_skip=skip_pack,
    )
    if log_fn:
        log_fn(format_dynamic_a_block({"dynamic_a": {
            "promotion": promotion,
            "on": on_pack,
            "off": off_pack,
            "skip": skip_pack,
            "alpha": best_alpha,
            "max_steps": max_steps,
            "n_aligned_val": n_hits["on"].get("val", 0),
        }}))
    return {
        "alpha": float(best_alpha),
        "q": float(use_q),
        "abs_tau": float(use_tau),
        "max_steps": int(max_steps),
        "n_aligned": n_hits,
        "n_aligned_val": int(n_hits["on"].get("val", 0)),
        "alpha_rows": [
            {k: v for k, v in row.items() if k != "scored"} for row in alpha_rows
        ],
        "on": on_pack,
        "off": off_pack,
        "skip": skip_pack,
        "promotion": promotion,
        "note": (
            "VAL-gated Dynamic A last-bar residual blend on the overnight skip. "
            "Ablation is the same tiny encoder with dynamic_weights off. "
            "TEST is report-only. Live q20 unchanged. Cover floor 5%."
        ),
    }
