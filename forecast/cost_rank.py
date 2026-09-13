"""(2b) Cost-aware ranking / IR-proxy aligned to live_locate after costs.

Huber ranks residual *magnitude*. The overnight book selects a q20 long/short
sleeve and pays live_locate costs (auction, thin-name multiplier, vol impact,
borrow, locate haircut). This module:

1. Subtracts a causal name-level live_locate drag from residual ``y`` so
   RankNet / rank-target ridge order names by *net* contribution.
2. Adds a differentiable IR proxy of a soft q20 LS book after the same costs.

Does **not** change default live_locate knobs. VAL-gate remains liquid
``live_locate`` unlevered net IR **+5.58** / max DD **−0.95**. TEST is
report-only. Enable with ``--cost-rank-loss``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from forecast.data import FEATURE_NAMES
from forecast.overnight import (
    LIVE_LOCATE_BUNDLE,
    LS_LIVE_HAIRCUT,
    LS_LIVE_Q,
    LS_LIVE_SHORT,
    row_cs_thin_mask,
)

# Book gate (liquid VAL, report-only on TEST). Do not retarget from TEST.
BASELINE_IR = 5.58
BASELINE_DD = -0.95
IR_LIFT = 0.05
DD_TOL = 0.05

TURN_COL = FEATURE_NAMES.index("turnover_z")
VOL_COL = FEATURE_NAMES.index("vol_level")

# live_locate name-level drag. Hedge overlay is date-flat and does not rank.
_RT_BPS = float(LIVE_LOCATE_BUNDLE["round_trip_bps"])
_MOC_BPS = float(LIVE_LOCATE_BUNDLE["moc_bps"])
_MOO_BPS = float(LIVE_LOCATE_BUNDLE["moo_bps"])
_BORROW_BPS = float(LIVE_LOCATE_BUNDLE["borrow_bps"])
_IMPACT_K = float(LIVE_LOCATE_BUNDLE["impact_vol_k"])
_THIN_MULT = float(LIVE_LOCATE_BUNDLE["thin_mult"])
_THIN_PCTILE = float(LIVE_LOCATE_BUNDLE["thin_pctile"])
_LOCATE_PCTILE = float(LIVE_LOCATE_BUNDLE["locate_pctile"])


def _as_1d(value: Any, n: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(n, default, dtype=np.float64)
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size == n:
        return out
    if out.size == 1:
        return np.full(n, float(out[0]), dtype=np.float64)
    n_use = min(int(out.size), n)
    pad = np.full(n, default, dtype=np.float64)
    pad[:n_use] = out[:n_use]
    return pad


def _dates_or_one(dates: np.ndarray | None, n: int) -> np.ndarray:
    if dates is None:
        return np.zeros(n, dtype=np.int64)
    out = np.asarray(dates, dtype=np.int64).reshape(-1)
    if out.size == n:
        return out
    return np.zeros(n, dtype=np.int64)


def thin_mask_for_dates(
    turnover_z: np.ndarray,
    dates: np.ndarray | None,
    *,
    pctile: float = _THIN_PCTILE,
) -> np.ndarray:
    """Within-date HTB / thin proxy. ``turnover_z`` is known at close t."""
    z = np.asarray(turnover_z, dtype=np.float64).reshape(-1)
    keys = _dates_or_one(dates, z.size)
    out = np.zeros(z.size, dtype=bool)
    if float(pctile) <= 0:
        return out
    for key in np.unique(keys):
        sel = keys == key
        if int(sel.sum()) < 3:
            continue
        row = row_cs_thin_mask(z[sel], float(pctile)).reshape(-1)
        out[sel] = row
    return out


def name_live_locate_cost(
    turnover_z: np.ndarray | None,
    vol_level: np.ndarray | None,
    dates: np.ndarray | None = None,
    *,
    short: np.ndarray | None = None,
    n: int | None = None,
) -> np.ndarray:
    """Per-name live_locate drag in **return** units (fraction of NAV).

    Assumes unit notional on the selected side. Thin names pay ``thin_mult``
    on MOC+MOO. High ``vol_level`` pays impact. Shorts add borrow. Locate
    haircut is applied in the IR proxy weights, not here (a selected HTB
    short still pays; the book just holds less of it).
    """
    if n is None:
        for src in (turnover_z, vol_level, short, dates):
            if src is not None:
                n = int(np.asarray(src).reshape(-1).size)
                break
        n = int(n or 0)
    n = int(n)
    turn = _as_1d(turnover_z, n, 0.0)
    vol = _as_1d(vol_level, n, 0.0)
    thin = thin_mask_for_dates(turn, dates, pctile=_THIN_PCTILE)
    thin_scale = np.where(thin, _THIN_MULT, 1.0)
    auction = (_MOC_BPS + _MOO_BPS) * thin_scale
    impact = _IMPACT_K * np.clip(vol, 0.0, None)
    cost_bps = _RT_BPS + auction + impact
    if short is not None:
        cost_bps = cost_bps + _BORROW_BPS * np.asarray(short, dtype=np.float64).reshape(-1)
    return cost_bps.astype(np.float64) * 1e-4


def cost_adjusted_residual(
    y: np.ndarray,
    scale: np.ndarray | None,
    turnover_z: np.ndarray | None = None,
    vol_level: np.ndarray | None = None,
    dates: np.ndarray | None = None,
) -> np.ndarray:
    """Residual units after shrinking by side-aware live_locate drag.

    ``y_net = y - sign(y) * cost_side / scale``. Expensive longs are pulled
    down; expensive shorts are pulled toward zero (less attractive to short).
    Longs pay RT+auction+impact; shorts add borrow. Date-flat hedge is omitted.
    """
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(target.size)
    sig = np.clip(_as_1d(scale, n, 1.0), 1e-8, None)
    short = target < 0.0
    cost = name_live_locate_cost(turnover_z, vol_level, dates, short=short, n=n)
    return target - np.sign(target) * cost / sig


def labelled_cost_aux(
    symbols: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``scale, turnover_z, vol_level`` aligned with ``labelled_rows``."""
    scales: list[np.ndarray] = []
    turns: list[np.ndarray] = []
    vols: list[np.ndarray] = []
    for sym in symbols:
        if not bool(getattr(sym, "valid", np.zeros(0, dtype=bool)).any()):
            continue
        feat = np.asarray(sym.features, dtype=np.float64)[np.asarray(sym.valid, dtype=bool)]
        scales.append(np.asarray(sym.scale, dtype=np.float64)[np.asarray(sym.valid, dtype=bool)])
        if feat.ndim != 2 or feat.shape[1] <= max(TURN_COL, VOL_COL):
            n = int(feat.shape[0]) if feat.ndim >= 1 else 0
            turns.append(np.zeros(n, dtype=np.float64))
            vols.append(np.zeros(n, dtype=np.float64))
            continue
        turns.append(feat[:, TURN_COL])
        vols.append(feat[:, VOL_COL])
    if not scales:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.concatenate(scales, axis=0),
        np.concatenate(turns, axis=0),
        np.concatenate(vols, axis=0),
    )


def apply_cost_adjusted_target(
    y: np.ndarray,
    symbols: Sequence[Any],
) -> np.ndarray:
    """Skip-fit helper: cost-adjust ``labelled_rows`` targets."""
    scale, turn, vol = labelled_cost_aux(symbols)
    dates = None
    rows_d: list[np.ndarray] = []
    for sym in symbols:
        if not bool(getattr(sym, "valid", np.zeros(0, dtype=bool)).any()):
            continue
        if getattr(sym, "dates", None) is None:
            rows_d.append(np.full(int(sym.valid.sum()), -1, dtype=np.int64))
        else:
            rows_d.append(np.asarray(sym.dates, dtype=np.int64)[np.asarray(sym.valid, dtype=bool)])
    if rows_d:
        dates = np.concatenate(rows_d, axis=0)
    n = int(np.asarray(y).reshape(-1).size)
    if scale.size != n:
        n_use = min(int(scale.size), n)
        y_use = np.asarray(y, dtype=np.float64).reshape(-1)[:n_use]
        return cost_adjusted_residual(
            y_use, scale[:n_use], turn[:n_use], vol[:n_use],
            dates=None if dates is None else dates[:n_use],
        )
    return cost_adjusted_residual(y, scale, turn, vol, dates=dates)


def decide_book_gate(
    val_ir: float,
    val_dd: float,
    *,
    baseline_ir: float = BASELINE_IR,
    baseline_dd: float = BASELINE_DD,
    ir_lift: float = IR_LIFT,
    dd_tol: float = DD_TOL,
) -> dict[str, Any]:
    """VAL-only live_locate IR/DD vs the parked overnight book. TEST never enters."""
    ir = float(val_ir)
    dd = float(val_dd)
    ir_delta = ir - float(baseline_ir)
    dd_delta = dd - float(baseline_dd)
    ir_ok = np.isfinite(ir) and ir_delta + 1e-12 >= float(ir_lift)
    dd_ok = np.isfinite(dd) and dd_delta + 1e-12 >= -float(dd_tol)
    promote = bool(ir_ok and dd_ok)
    if promote:
        reason = (
            f"PROMOTE book: VAL live_locate unlev net IR {ir:+.3f} vs "
            f"{baseline_ir:+.2f} (delta {ir_delta:+.3f} >= {ir_lift:.2f}) "
            f"and max DD {dd:+.3f} vs {baseline_dd:+.2f} "
            f"(delta {dd_delta:+.3f} >= -{dd_tol:.2f})."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE book: VAL live_locate IR {ir:+.3f} vs "
            f"{baseline_ir:+.2f} (delta {ir_delta:+.3f} < {ir_lift:.2f}). "
            "Keep q20/h0.5/s0.50. TEST report-only."
        )
    else:
        reason = (
            f"NO PROMOTE book: VAL IR lift {ir_delta:+.3f} clears {ir_lift:.2f} "
            f"but max DD {dd:+.3f} is worse than {baseline_dd:+.2f} by more "
            f"than {dd_tol:.2f}. Keep q20/h0.5/s0.50. TEST report-only."
        )
    return {
        "promote": promote,
        "val_ir": ir,
        "val_dd": dd,
        "baseline_ir": float(baseline_ir),
        "baseline_dd": float(baseline_dd),
        "ir_delta": ir_delta,
        "dd_delta": dd_delta,
        "ir_lift": float(ir_lift),
        "dd_tol": float(dd_tol),
        "reason": reason,
        "test_report_only": True,
        "keep_knobs": "q20/h0.5/s0.50",
    }


def book_gate_payload() -> dict[str, Any]:
    return {
        "metric": "live_locate_unlevered_net_ir",
        "baseline_ir": BASELINE_IR,
        "baseline_dd": BASELINE_DD,
        "ir_lift": IR_LIFT,
        "dd_tol": DD_TOL,
        "knobs": {
            "quantile": LS_LIVE_Q,
            "locate_haircut": LS_LIVE_HAIRCUT,
            "max_short_gross": LS_LIVE_SHORT,
        },
        "role": "VAL gate; TEST report-only; do not retarget TEST",
    }


def format_cost_rank_block(cfg: Mapping[str, Any] | Any) -> str:
    get = cfg.get if isinstance(cfg, Mapping) else lambda k, d=None: getattr(cfg, k, d)
    replace = bool(get("cost_rank_replace_huber", False))
    return "\n".join(
        [
            "Cost-aware ranking / IR-proxy (2b) - live_locate after costs",
            f"  RankNet on net residual  weight={float(get('cost_rank_weight', 0.0) or 0.0):.3f}",
            f"  IR proxy (soft q20 LS)   weight={float(get('ir_proxy_weight', 0.0) or 0.0):.3f}",
            f"  Huber                    {'replaced' if replace else 'augmented'}",
            f"  VAL book gate            live_locate IR vs {BASELINE_IR:+.2f} / "
            f"DD vs {BASELINE_DD:+.2f} (lift {IR_LIFT:.2f}, DD tol {DD_TOL:.2f})",
            "  Knobs stay q20/h0.5/s0.50 unless that VAL gate clears. TEST report-only.",
        ]
    )


# --------------------------------------------------------------------------
# Torch losses
# --------------------------------------------------------------------------


def _expand_like(ids: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
    if ids is None:
        return None
    if ids.shape == mask.shape:
        return ids
    if ids.dim() == 1 and ids.size(0) == mask.size(0):
        return ids.unsqueeze(-1).expand_as(mask)
    return ids.reshape(mask.shape)


def _last_bar_or_masked(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Flatten ``tensor`` the same way as ``mean[mask]``."""
    if tensor.shape[:2] == mask.shape[:2] or tensor.shape == mask.shape:
        return tensor[mask.bool()]
    if tensor.dim() >= 2 and tensor.size(0) == mask.size(0):
        last = tensor[:, -1] if tensor.dim() > 1 else tensor
        row_sel = mask.bool().any(dim=-1) if mask.dim() > 1 else mask.bool()
        return last[row_sel]
    return tensor.reshape(-1)[mask.bool().reshape(-1)[: tensor.numel()]]


def denormalize_last_features(
    features: torch.Tensor,
    mask: torch.Tensor,
    feature_mean: np.ndarray | torch.Tensor | None,
    feature_std: np.ndarray | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(turnover_z, vol_level)`` on labelled bars, raw units."""
    last = features
    if features.dim() == 3:
        last = features
    raw = last
    if feature_mean is not None and feature_std is not None:
        mean = torch.as_tensor(feature_mean, device=features.device, dtype=features.dtype)
        std = torch.as_tensor(feature_std, device=features.device, dtype=features.dtype)
        std = torch.clamp(std, min=1e-8)
        if raw.dim() == 3:
            raw = raw * std.view(1, 1, -1) + mean.view(1, 1, -1)
        elif raw.dim() == 2:
            raw = raw * std.view(1, -1) + mean.view(1, -1)
    if raw.dim() == 3:
        take = raw[mask.bool()]
    elif raw.dim() == 2 and raw.size(0) == mask.size(0):
        take = raw[mask.bool().any(dim=-1) if mask.dim() > 1 else mask.bool()]
    else:
        take = raw.reshape(-1, raw.shape[-1])
    n_feat = int(take.shape[-1]) if take.numel() else 0
    zeros = features.new_zeros((take.shape[0],))
    if n_feat <= max(TURN_COL, VOL_COL):
        return zeros, zeros
    return take[:, TURN_COL], take[:, VOL_COL]


def _torch_thin_mask(
    turnover_z: torch.Tensor,
    dates: torch.Tensor | None,
    *,
    pctile: float,
) -> torch.Tensor:
    z = turnover_z.reshape(-1)
    out = torch.zeros_like(z, dtype=torch.bool)
    p = float(pctile)
    if p <= 0 or int(z.numel()) < 3:
        return out
    if dates is None:
        keys = z.new_zeros(z.shape, dtype=torch.long)
    else:
        keys = dates.reshape(-1).to(dtype=torch.long)
        if keys.numel() != z.numel():
            keys = z.new_zeros(z.shape, dtype=torch.long)
    for key in keys.unique():
        sel = keys == key
        row = z[sel]
        finite = torch.isfinite(row)
        if int(finite.sum()) < 3:
            continue
        vals = row[finite]
        if float((vals.max() - vals.min()).detach()) < 1e-12:
            continue
        # quantile via kthvalue on detached scores so the mask is a feature.
        k = max(1, min(int(vals.numel()), int(round((vals.numel() - 1) * p)) + 1))
        cut = torch.kthvalue(vals.detach(), k).values
        out[sel] = finite & (row < cut)
    return out


def _torch_name_cost(
    turnover_z: torch.Tensor,
    vol_level: torch.Tensor,
    dates: torch.Tensor | None,
    *,
    short: torch.Tensor,
) -> torch.Tensor:
    thin = _torch_thin_mask(turnover_z, dates, pctile=_THIN_PCTILE)
    thin_scale = torch.where(thin, turnover_z.new_tensor(_THIN_MULT), turnover_z.new_tensor(1.0))
    auction = (_MOC_BPS + _MOO_BPS) * thin_scale
    impact = _IMPACT_K * torch.clamp(vol_level.reshape(-1), min=0.0)
    cost_bps = _RT_BPS + auction + impact + _BORROW_BPS * short.to(dtype=turnover_z.dtype).reshape(-1)
    return cost_bps * 1e-4


def torch_cost_adjusted_residual(
    y: torch.Tensor,
    scale: torch.Tensor | None,
    turnover_z: torch.Tensor | None,
    vol_level: torch.Tensor | None,
    dates: torch.Tensor | None,
) -> torch.Tensor:
    target = y.reshape(-1)
    n = int(target.numel())
    if scale is None:
        sig = target.new_ones(n)
    else:
        sig = torch.clamp(scale.reshape(-1)[:n], min=1e-8)
        if sig.numel() != n:
            sig = target.new_ones(n)
    zeros = target.new_zeros(n)
    turn = zeros if turnover_z is None else turnover_z.reshape(-1)
    vol = zeros if vol_level is None else vol_level.reshape(-1)
    if turn.numel() != n:
        turn = zeros
    if vol.numel() != n:
        vol = zeros
    short = target < 0
    cost = _torch_name_cost(turn, vol, dates, short=short)
    return target - torch.sign(target) * cost / sig


def _ranknet(pred: torch.Tensor, y: torch.Tensor, pair_w: torch.Tensor | None = None) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_zeros(())
    scale = pred.detach().std(unbiased=False).clamp(min=1.0)
    unit = pred / scale
    diff_p = unit.unsqueeze(0) - unit.unsqueeze(1)
    diff_y = y.unsqueeze(0) - y.unsqueeze(1)
    valid = diff_y.abs() > 1e-6
    if not bool(valid.any()):
        return pred.new_zeros(())
    per = F.softplus(-diff_p * diff_y.sign())
    if pair_w is None:
        return per[valid].mean()
    w = pair_w.to(dtype=per.dtype)
    w = torch.where(valid, w, w.new_zeros(()))
    denom = w.sum().clamp(min=1e-8)
    return (per * w).sum() / denom


def _tail_pair_weights(y: torch.Tensor, q: float = LS_LIVE_Q) -> torch.Tensor:
    """Upweight pairs that touch the live_locate q-tails of the *target*."""
    n = int(y.numel())
    if n < 4 or q <= 0:
        return y.new_ones((n, n))
    k = max(1, int(round(n * float(q))))
    order = torch.argsort(y)
    tail = torch.zeros(n, dtype=torch.bool, device=y.device)
    tail[order[:k]] = True
    tail[order[-k:]] = True
    w = 1.0 + tail.float().unsqueeze(0) + tail.float().unsqueeze(1)
    return w


def masked_cost_rank_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    date_ids: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    turnover_z: torch.Tensor | None = None,
    vol_level: torch.Tensor | None = None,
    max_points: int = 256,
) -> torch.Tensor:
    """Within-date RankNet on cost-adjusted residual, tail-weighted for q20."""
    dates = _expand_like(date_ids, mask)
    sel = mask.bool()
    pred = mean[sel]
    y = target[sel]
    sc = None if scale is None else _last_bar_or_masked(scale, mask)
    dsel = None if dates is None else dates[sel]
    y_net = torch_cost_adjusted_residual(y, sc, turnover_z, vol_level, dsel)
    if dsel is not None:
        parts: list[torch.Tensor] = []
        for key in dsel.unique():
            m = dsel == key
            if int(m.sum()) < 2:
                continue
            parts.append(_ranknet(pred[m], y_net[m], _tail_pair_weights(y_net[m])))
        if parts:
            return torch.stack(parts).mean()
    n = int(pred.numel())
    if n < 2:
        return mean.new_zeros(())
    if n > max_points:
        idx = torch.randperm(n, device=pred.device)[:max_points]
        pred = pred[idx]
        y_net = y_net[idx]
    return _ranknet(pred, y_net, _tail_pair_weights(y_net))


def _soft_ls_weights(
    pred: torch.Tensor,
    turnover_z: torch.Tensor,
    dates: torch.Tensor | None,
    *,
    q: float = LS_LIVE_Q,
    haircut: float = LS_LIVE_HAIRCUT,
    max_short: float = LS_LIVE_SHORT,
) -> torch.Tensor:
    """Soft q-tail LS weights (long +0.5 / short -0.5 NAV before haircut)."""
    n = int(pred.numel())
    if n < 2:
        return pred.new_zeros(n)
    tau = pred.detach().std(unbiased=False).clamp(min=1e-3)
    # Detached quantile so the cut tracks live_locate selection, grads flow
    # through the sigmoid membership.
    hi = torch.quantile(pred.detach(), 1.0 - float(q))
    lo = torch.quantile(pred.detach(), float(q))
    long_m = torch.sigmoid((pred - hi) / tau)
    short_m = torch.sigmoid((lo - pred) / tau)
    thin = _torch_thin_mask(turnover_z, dates, pctile=_LOCATE_PCTILE)
    short_m = short_m * (1.0 - float(haircut) * thin.to(dtype=pred.dtype))
    long_sum = long_m.sum().clamp(min=1e-8)
    short_sum = short_m.sum().clamp(min=1e-8)
    long_w = 0.5 * long_m / long_sum
    short_w = float(max_short) * short_m / short_sum
    return long_w - short_w


def masked_ir_proxy_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    date_ids: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    turnover_z: torch.Tensor | None = None,
    vol_level: torch.Tensor | None = None,
) -> torch.Tensor:
    """``-mean(net) / std(net)`` of a soft live_locate book after costs.

    Per-date net is ``sum(w * (y * scale - name_cost))``. Weights are a
    sigmoid-q20 long/short with locate haircut on thin shorts. Maximizing
    this proxy is IR-aligned; the loss is the negative.
    """
    dates = _expand_like(date_ids, mask)
    sel = mask.bool()
    pred = mean[sel]
    y = target[sel]
    if int(pred.numel()) < 2:
        return mean.new_zeros(())
    sc = (
        _last_bar_or_masked(scale, mask).reshape(-1)
        if scale is not None
        else pred.new_ones(pred.shape)
    )
    if sc.numel() != pred.numel():
        sc = pred.new_ones(pred.shape)
    sc = torch.clamp(sc, min=1e-8)
    dsel = None if dates is None else dates[sel]
    zeros = pred.new_zeros(pred.shape)
    turn = zeros if turnover_z is None else turnover_z.reshape(-1)
    vol = zeros if vol_level is None else vol_level.reshape(-1)
    if turn.numel() != pred.numel():
        turn = zeros
    if vol.numel() != pred.numel():
        vol = zeros
    keys = dsel if dsel is not None else pred.new_zeros(pred.shape, dtype=torch.long)
    nets: list[torch.Tensor] = []
    for key in keys.unique():
        m = keys == key
        if int(m.sum()) < 3:
            continue
        p = pred[m]
        yy = y[m]
        ss = sc[m]
        tz = turn[m]
        vl = vol[m]
        dk = keys[m]
        w = _soft_ls_weights(p, tz, dk)
        short = w < 0
        cost = _torch_name_cost(tz, vl, dk, short=short)
        gross = (w * yy * ss).sum()
        drag = (w.abs() * cost).sum()
        nets.append(gross - drag)
    if not nets:
        # Single pooled date (or a thin CS). Still a return proxy.
        w = _soft_ls_weights(pred, turn, dsel)
        short = w < 0
        cost = _torch_name_cost(turn, vol, dsel, short=short)
        net = (w * y * sc).sum() - (w.abs() * cost).sum()
        return -net
    stacked = torch.stack(nets)
    if stacked.numel() < 2:
        return -stacked.mean()
    mu = stacked.mean()
    sd = stacked.std(unbiased=False).clamp(min=1e-4)
    return -(mu / sd)


def cost_rank_terms(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cfg: Any,
    *,
    date_ids: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    turnover_z: torch.Tensor | None = None,
    vol_level: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted RankNet + IR-proxy. Zero when both weights are 0."""
    rank_w = float(getattr(cfg, "cost_rank_weight", 0.0) or 0.0)
    ir_w = float(getattr(cfg, "ir_proxy_weight", 0.0) or 0.0)
    if rank_w <= 0 and ir_w <= 0:
        return mean.new_zeros(())
    if turnover_z is None or vol_level is None:
        if features is not None:
            turnover_z, vol_level = denormalize_last_features(
                features, mask, feature_mean, feature_std
            )
    total = mean.new_zeros(())
    if rank_w > 0:
        term = masked_cost_rank_loss(
            mean, target, mask,
            date_ids=date_ids, scale=scale,
            turnover_z=turnover_z, vol_level=vol_level,
        )
        if torch.isfinite(term):
            total = total + rank_w * term
    if ir_w > 0:
        term = masked_ir_proxy_loss(
            mean, target, mask,
            date_ids=date_ids, scale=scale,
            turnover_z=turnover_z, vol_level=vol_level,
        )
        if torch.isfinite(term):
            total = total + ir_w * term
    return total
