"""Last-bar quantile long-short book on a locked test window.

Usage:
    python -m forecast.backtest --checkpoint checkpoints/forecast_ridge/best.pt
    python -m forecast.backtest --checkpoint ... --quantile 0.2 --cost-bps 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from forecast.checkpoint import load_forecaster
from forecast.config import DataConfig
from forecast.overnight import (
    apply_locate_gate,
    apply_short_constraints,
    book_side_stats,
    capacity_note,
    holding_for_label,
    merge_cost_kwargs,
    overnight_cost_breakdown,
    overnight_one_way_turnover,
    resolve_cost_bundle,
    LS_HAIRCUT_EXPERIMENT,
)
from forecast.generate import (
    forecast_panel,
    load_forecast_panels,
    parse_symbols,
    resolve_data_files,
    symbol_from_path,
)
from forecast.training import _pearson, _spearman
from forecast.universe import is_equity_name
from mamba_lm.paths import anchor_to_repo, resolve_path


def test_start_from_state(state: dict[str, Any]) -> pd.Timestamp | None:
    ends = []
    for row in state.get("symbols") or []:
        raw = row.get("val_end")
        if raw:
            ends.append(pd.Timestamp(raw))
    if not ends:
        return None
    return max(ends)


def score_last_bars(
    model: torch.nn.Module,
    panels: dict[str, pd.DataFrame],
    state: dict[str, Any],
    device: torch.device,
    *,
    context: int,
    start: pd.Timestamp | None,
    batch_size: int = 64,
    exclude: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide pred_norm and realized residual (same units as the training label)."""
    skip = exclude or frozenset()
    pred_parts: list[pd.Series] = []
    y_parts: list[pd.Series] = []
    for symbol, panel in panels.items():
        if symbol in skip or len(panel) < context:
            continue
        if start is not None:
            stamps = pd.to_datetime(panel["datetime"])
            mask = stamps >= start
            idx = np.flatnonzero(mask.to_numpy())
            idx = idx[idx >= context - 1]
        else:
            idx = np.arange(context - 1, len(panel), dtype=np.int64)
        if idx.size == 0:
            continue
        scored = forecast_panel(
            model,
            panel,
            state,
            device,
            context=context,
            positions=idx.astype(np.int64),
            batch_size=batch_size,
            include_uncertainty=False,
        )
        when = pd.to_datetime(scored["datetime"]).dt.tz_localize(None).dt.normalize()
        pred_parts.append(
            pd.Series(scored["pred_norm"].to_numpy(dtype=np.float64), index=when, name=symbol)
        )
        if "realized_bps" in scored.columns and "scale" in scored.columns:
            realized = scored["realized_bps"].to_numpy(dtype=np.float64) / 1e4
            scale = scored["scale"].to_numpy(dtype=np.float64)
            y_norm = np.divide(
                realized, scale, out=np.full_like(realized, np.nan), where=scale > 0
            )
        else:
            y_norm = np.full(len(scored), np.nan)
        y_parts.append(pd.Series(y_norm, index=when, name=symbol))
    if not pred_parts:
        empty = pd.DataFrame()
        return empty, empty
    pred = pd.concat(pred_parts, axis=1).sort_index()
    realized = pd.concat(y_parts, axis=1).reindex_like(pred)
    pred = pred.groupby(level=0).last()
    realized = realized.groupby(level=0).last()
    return pred, realized


def cs_ic_by_date(
    pred: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    min_names: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for ts in pred.index:
        p = pred.loc[ts]
        y = realized.loc[ts]
        pair = pd.concat([p, y], axis=1, keys=["p", "y"]).dropna()
        if len(pair) < min_names:
            continue
        rows.append(
            {
                "datetime": ts,
                "n": float(len(pair)),
                "ic": _pearson(pair["p"].to_numpy(), pair["y"].to_numpy()),
                "ic_spearman": _spearman(pair["p"].to_numpy(), pair["y"].to_numpy()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["datetime", "n", "ic", "ic_spearman"])
    return pd.DataFrame(rows).set_index("datetime")


def quantile_weights(
    scores: pd.Series,
    *,
    quantile: float,
    long_only: bool = False,
) -> pd.Series:
    """Long top / short bottom (dollar-neutral) or long-only top quantile."""
    s = scores.dropna()
    w = pd.Series(0.0, index=s.index, dtype=np.float64)
    n = int(s.size)
    if n < 2:
        return w
    q = min(0.49, max(0.05, float(quantile)))
    k = max(1, int(math.floor(n * q)))
    order = s.sort_values()
    if long_only:
        if n < 5:
            w.loc[order.index[-1]] = 1.0
            return w
        long = order.index[-k:]
        w.loc[long] = 1.0 / k
        return w
    if n < 5:
        w.loc[order.index[0]] = -0.5
        w.loc[order.index[-1]] = 0.5
        return w
    short = order.index[:k]
    long = order.index[-k:]
    w.loc[short] = -0.5 / k
    w.loc[long] = 0.5 / k
    return w


def rank_weights(scores: pd.Series, *, long_only: bool = False) -> pd.Series:
    """Dollar-neutral weights from centered CS rank (softer than 20% tails)."""
    s = scores.dropna()
    w = pd.Series(0.0, index=s.index, dtype=np.float64)
    n = int(s.size)
    if n < 2:
        return w
    r = s.rank(method="average")
    z = r - r.mean()
    if long_only:
        z = z.clip(lower=0.0)
        total = float(z.sum())
        if total <= 1e-12:
            return w
        w.loc[z.index] = z / total
        return w
    denom = float(z.abs().sum())
    if denom <= 1e-12:
        return w
    w.loc[z.index] = z / denom
    return w


def panel_overnight_r_wide(
    panels: dict[str, pd.DataFrame],
    like: pd.DataFrame,
) -> pd.DataFrame:
    """``r_on = log(open_{t+1}) - log(close_t)`` aligned to a pred frame.

    Next open is a *label*, never a feature. Used only to score sleeve
    overnight direction after the residual skip has picked names.
    """
    parts: list[pd.Series] = []
    for symbol in like.columns:
        panel = panels.get(str(symbol))
        if panel is None or "open" not in panel.columns or "close" not in panel.columns:
            continue
        close = panel["close"].to_numpy(dtype=np.float64)
        opn = panel["open"].to_numpy(dtype=np.float64)
        r_on = np.full(close.shape[0], np.nan, dtype=np.float64)
        if close.size > 1:
            nxt = opn[1:]
            c = close[:-1]
            ok = (c > 0) & (nxt > 0) & np.isfinite(c) & np.isfinite(nxt)
            r_on[:-1] = np.where(ok, np.log(nxt) - np.log(np.clip(c, 1e-12, None)), np.nan)
        when = pd.to_datetime(panel["datetime"])
        if getattr(when.dt, "tz", None) is not None:
            when = when.dt.tz_convert("America/New_York").dt.tz_localize(None)
        when = when.dt.normalize()
        parts.append(pd.Series(r_on, index=when, name=symbol))
    if not parts:
        return pd.DataFrame(np.nan, index=like.index, columns=like.columns)
    out = pd.concat(parts, axis=1).sort_index()
    out = out.groupby(level=0).last()
    return out.reindex(index=like.index, columns=like.columns)


def sleeve_direction_from_weights(
    weights: pd.DataFrame,
    overnight_r: pd.DataFrame,
) -> dict[str, float]:
    """Overnight up/down hit rates on the *held* long and short legs."""
    w = np.asarray(weights, dtype=np.float64)
    r = overnight_r.reindex(index=weights.index, columns=weights.columns).to_numpy(
        dtype=np.float64
    )
    long_m = (w > 1e-12) & np.isfinite(r) & (r != 0.0)
    short_m = (w < -1e-12) & np.isfinite(r) & (r != 0.0)
    moved = np.isfinite(r) & (r != 0.0)
    uncond_up = float((r[moved] > 0).mean()) if int(moved.sum()) else float("nan")
    uncond_down = float((r[moved] < 0).mean()) if int(moved.sum()) else float("nan")
    long_up = float((r[long_m] > 0).mean()) if int(long_m.sum()) else float("nan")
    short_down = float((r[short_m] < 0).mean()) if int(short_m.sum()) else float("nan")
    return {
        "uncond_up_pct": float(100.0 * uncond_up) if np.isfinite(uncond_up) else float("nan"),
        "uncond_down_pct": (
            float(100.0 * uncond_down) if np.isfinite(uncond_down) else float("nan")
        ),
        "long_n": float(int(long_m.sum())),
        "long_up_pct": float(100.0 * long_up) if np.isfinite(long_up) else float("nan"),
        "long_excess_pp": (
            float(100.0 * (long_up - uncond_up))
            if np.isfinite(long_up) and np.isfinite(uncond_up)
            else float("nan")
        ),
        "short_n": float(int(short_m.sum())),
        "short_down_pct": (
            float(100.0 * short_down) if np.isfinite(short_down) else float("nan")
        ),
        "short_excess_pp": (
            float(100.0 * (short_down - uncond_down))
            if np.isfinite(short_down) and np.isfinite(uncond_down)
            else float("nan")
        ),
    }


def panel_feature_wide(
    panels: dict[str, pd.DataFrame],
    column: str,
    like: pd.DataFrame,
) -> pd.DataFrame:
    """Align a panel column onto a pred/weight frame (date x symbol)."""
    parts: list[pd.Series] = []
    for symbol in like.columns:
        panel = panels.get(str(symbol))
        if panel is None or column not in panel.columns:
            continue
        when = pd.to_datetime(panel["datetime"])
        if getattr(when.dt, "tz", None) is not None:
            when = when.dt.tz_convert("America/New_York").dt.tz_localize(None)
        when = when.dt.normalize()
        parts.append(
            pd.Series(panel[column].to_numpy(dtype=np.float64), index=when, name=symbol)
        )
    if not parts:
        return pd.DataFrame(np.nan, index=like.index, columns=like.columns)
    out = pd.concat(parts, axis=1).sort_index()
    out = out.groupby(level=0).last()
    return out.reindex(index=like.index, columns=like.columns)


def date_weights(
    scores: pd.Series,
    *,
    weighting: str = "quantile",
    quantile: float = 0.2,
    long_only: bool = False,
) -> pd.Series:
    mode = str(weighting or "quantile").lower()
    if mode == "rank":
        return rank_weights(scores, long_only=long_only)
    return quantile_weights(scores, quantile=quantile, long_only=long_only)


def _renorm_row(w: np.ndarray, *, long_only: bool) -> np.ndarray:
    out = np.asarray(w, dtype=np.float64).copy()
    if long_only:
        out = np.clip(out, 0.0, None)
        s = float(out.sum())
        if s > 1e-12:
            return out / s
        return np.zeros_like(out)
    long = np.clip(out, 0.0, None)
    short = np.clip(out, None, 0.0)
    ls = float(long.sum())
    ss = float(-short.sum())
    if ls > 1e-12:
        long *= 0.5 / ls
    else:
        long[:] = 0.0
    if ss > 1e-12:
        short *= 0.5 / ss
    else:
        short[:] = 0.0
    return long + short


def smooth_weights(
    w_panel: pd.DataFrame,
    *,
    hold_halflife: float,
    long_only: bool = False,
) -> pd.DataFrame:
    """Causal EWMA of target weights, then renormalize each date."""
    hl = float(hold_halflife)
    if hl <= 0 or len(w_panel) < 2:
        return w_panel
    alpha = 1.0 - math.exp(math.log(0.5) / hl)
    arr = w_panel.to_numpy(dtype=np.float64, copy=True)
    smoothed = np.empty_like(arr)
    smoothed[0] = _renorm_row(arr[0], long_only=long_only)
    for i in range(1, len(arr)):
        blended = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
        smoothed[i] = _renorm_row(blended, long_only=long_only)
    return pd.DataFrame(smoothed, index=w_panel.index, columns=w_panel.columns)


def _annualized_ir(x: pd.Series, periods_per_year: float) -> float:
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=0))
    if sd <= 1e-12:
        return float("nan")
    return float(x.mean() / sd * math.sqrt(periods_per_year))


def _max_dd(pnl: pd.Series) -> float:
    if pnl.empty:
        return float("nan")
    equity = pnl.cumsum()
    return float((equity - equity.cummax()).min())


def book_pnl(
    pred: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    quantile: float = 0.2,
    round_trip_bps: float = 10.0,
    vol_target: float = 0.15,
    periods_per_year: float = 252.0,
    long_only: bool = False,
    min_names: int = 8,
    weighting: str = "quantile",
    hold_halflife: float = 1.0,
    causal_vol: bool = True,
    lever_cap: float = 3.0,
    min_vol_days: int = 21,
    holding: str = "close",
    open_auction_bps: float = 0.0,
    borrow_bps: float = 0.0,
    hedge_cost_bps: float = 0.0,
    moc_bps: float = 0.0,
    moo_bps: float = 0.0,
    session_exit_bps: float = 0.0,
    impact_vol_k: float = 0.0,
    thin_mult: float = 1.0,
    thin_pctile: float = 0.0,
    locate_pctile: float = 0.0,
    locate_haircut: float = 1.0,
    locate_frac: float = 1.0,
    max_short_gross: float = 0.5,
    ex_post_gap_k: float = 0.0,
    turnover_z: pd.DataFrame | None = None,
    vol_level: pd.DataFrame | None = None,
    overnight_r: pd.DataFrame | None = None,
    adv_floor_pctile: float = 0.0,
) -> dict[str, Any]:
    """Cost-aware long-short with optional rank weights, hold smoothing, causal vol.

    Full-sample ``vol / lever`` is look-ahead on the *path* (IR of a constant
    scale is invariant). Default ``causal_vol`` uses expanding std of prior
    unlevered gross only. ``vol_target`` is annualized; 1.0 is a 100% vol book
    and will print catastrophic max DD even when IR is ~1.

    ``holding='overnight'`` matches the overnight gap label: flatten every
    open (no session EWMA). Costs charge a full enter+exit each night, plus
    optional MOC/MOO auction, thin-name multiplier, vol impact, borrow, and
    residual-hedge overlay. ``open_fill`` is MOC plus a continuous open+N exit.
    ``ex_post_gap_k`` uses |realized| and is a sensitivity, not the default.
    """
    if long_only:
        borrow_bps = 0.0
        locate_pctile = 0.0
    hold_mode = str(holding or "close").strip().lower()
    if hold_mode in ("on", "gap", "close_open"):
        hold_mode = "overnight"
    if hold_mode in ("fill", "open_n", "open+n"):
        hold_mode = "open_fill"
    flatten = hold_mode in ("overnight", "open_fill")
    if flatten and float(hold_halflife) > 0:
        # Session carry would mix open→close into an overnight / open+N book.
        hold_halflife = 0.0
    dates = pred.index.intersection(realized.index)
    raw_w: list[pd.Series] = []
    gross: list[float] = []
    kept: list[Any] = []
    for ts in dates:
        pair_all = pd.concat(
            [pred.loc[ts], realized.loc[ts]], axis=1, keys=["p", "r"]
        ).dropna()
        if (
            turnover_z is not None
            and float(adv_floor_pctile) > 0
            and ts in turnover_z.index
        ):
            tzrow = turnover_z.loc[ts].reindex(pair_all.index)
            finite = tzrow.notna() & np.isfinite(tzrow)
            if int(finite.sum()) >= 5:
                cut = float(np.nanpercentile(tzrow[finite], 100.0 * float(adv_floor_pctile)))
                pair_all = pair_all.loc[finite & (tzrow >= cut)]
        if len(pair_all) < int(min_names):
            continue
        w = date_weights(
            pair_all["p"],
            weighting=weighting,
            quantile=quantile,
            long_only=long_only,
        )
        r = pair_all["r"].reindex(w.index)
        pair = pd.concat([w, r], axis=1, keys=["w", "r"]).dropna()
        if len(pair) < 2:
            continue
        pnl = float((pair["w"] * pair["r"]).sum())
        if not np.isfinite(pnl):
            continue
        raw_w.append(w.rename(ts))
        gross.append(pnl)
        kept.append(ts)
    if len(gross) < 5:
        return {
            "n_dates": float(len(gross)),
            "gross_ir": float("nan"),
            "net_ir": float("nan"),
            "unlevered_gross_ir": float("nan"),
            "unlevered_net_ir": float("nan"),
            "hit_rate": float("nan"),
            "max_dd": float("nan"),
            "unlevered_max_dd": float("nan"),
            "mean_cs_ic": float("nan"),
            "mean_cs_ic_spearman": float("nan"),
        }
    w_panel = pd.concat(raw_w, axis=1).T.fillna(0.0)
    w_panel.index = pd.to_datetime(w_panel.index)
    w_panel = smooth_weights(
        w_panel, hold_halflife=hold_halflife, long_only=long_only
    )
    tz_kept = None
    vol_kept = None
    if turnover_z is not None:
        tz_kept = turnover_z.reindex(index=w_panel.index, columns=w_panel.columns)
    if vol_level is not None:
        vol_kept = vol_level.reindex(index=w_panel.index, columns=w_panel.columns)
    n_blocked = np.zeros(len(w_panel), dtype=np.float64)
    if not long_only:
        gated, n_blocked = apply_short_constraints(
            w_panel.to_numpy(dtype=np.float64),
            None if tz_kept is None else tz_kept.to_numpy(dtype=np.float64),
            locate_pctile=float(locate_pctile),
            locate_haircut=float(locate_haircut),
            locate_frac=float(locate_frac),
            max_short_gross=float(max_short_gross),
            long_only=False,
        )
        w_panel = pd.DataFrame(gated, index=w_panel.index, columns=w_panel.columns)
    realized_kept = realized.reindex(index=w_panel.index, columns=w_panel.columns)
    gross_s = (w_panel * realized_kept).sum(axis=1, skipna=True).astype(np.float64)
    w_arr = w_panel.to_numpy(dtype=np.float64)
    tz_arr = None if tz_kept is None else tz_kept.to_numpy(dtype=np.float64)
    vol_arr = None if vol_kept is None else vol_kept.to_numpy(dtype=np.float64)
    gap_arr = None
    if float(ex_post_gap_k):
        gap_arr = realized_kept.abs().to_numpy(dtype=np.float64)
    cost_kwargs = dict(
        round_trip_bps=round_trip_bps,
        open_auction_bps=open_auction_bps,
        borrow_bps=borrow_bps,
        hedge_cost_bps=hedge_cost_bps,
        moc_bps=moc_bps,
        moo_bps=moo_bps,
        session_exit_bps=session_exit_bps,
        turnover_z=tz_arr,
        vol_level=vol_arr,
        realized_abs=gap_arr,
        impact_vol_k=impact_vol_k,
        thin_mult=thin_mult,
        thin_pctile=thin_pctile,
        ex_post_gap_k=ex_post_gap_k,
    )
    if flatten:
        turnover = pd.Series(overnight_one_way_turnover(w_arr), index=w_panel.index)
        parts = overnight_cost_breakdown(weights=w_arr, **cost_kwargs)
        cost_unlev = pd.Series(parts["total"], index=w_panel.index)
        cost_parts = {k: float(np.nanmean(v)) for k, v in parts.items()}
    else:
        prev = w_panel.shift(1).fillna(0.0)
        turnover = 0.5 * (w_panel - prev).abs().sum(axis=1)
        cost_unlev = (float(round_trip_bps) * 1e-4) * turnover
        parts = None
        if any(
            float(x)
            for x in (
                open_auction_bps,
                borrow_bps,
                hedge_cost_bps,
                moc_bps,
                moo_bps,
                session_exit_bps,
                impact_vol_k,
                ex_post_gap_k,
            )
        ):
            extra_kw = dict(cost_kwargs)
            extra_kw["round_trip_bps"] = 0.0
            parts = overnight_cost_breakdown(weights=w_arr, **extra_kw)
            cost_unlev = cost_unlev + parts["total"]
        cost_parts = {k: float(np.nanmean(v)) for k, v in parts.items()} if parts else {}
        cost_parts["total"] = float(cost_unlev.mean()) if len(cost_unlev) else float("nan")
        cost_parts["round_trip"] = float(
            ((float(round_trip_bps) * 1e-4) * turnover).mean()
        ) if len(turnover) else float("nan")
    net_unlev = gross_s - cost_unlev

    ppy = float(periods_per_year)
    if causal_vol and vol_target > 0:
        past = gross_s.shift(1)
        exp_std = past.expanding(min_periods=max(2, int(min_vol_days))).std(ddof=0)
        prior = float(vol_target) / math.sqrt(ppy)
        exp_std = exp_std.fillna(prior)
        lever_s = float(vol_target) / (exp_std * math.sqrt(ppy))
        if lever_cap > 0:
            lever_s = lever_s.clip(upper=float(lever_cap))
        lever_s = lever_s.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        mean_lever = float(lever_s.mean())
    elif vol_target > 0:
        vol = float(gross_s.std(ddof=0))
        lever = 1.0
        if vol > 1e-12:
            lever = float(vol_target) / (vol * math.sqrt(ppy))
            if lever_cap > 0:
                lever = min(lever, float(lever_cap))
        lever_s = pd.Series(lever, index=gross_s.index, dtype=np.float64)
        mean_lever = float(lever)
    else:
        lever_s = pd.Series(1.0, index=gross_s.index, dtype=np.float64)
        mean_lever = 1.0

    cost = cost_unlev * lever_s
    net_s = lever_s * gross_s - cost
    ics = cs_ic_by_date(pred.loc[w_panel.index], realized.loc[w_panel.index])
    sides = book_side_stats(w_arr)
    sleeve: dict[str, float] = {}
    if overnight_r is not None and not w_panel.empty:
        sleeve = sleeve_direction_from_weights(w_panel, overnight_r)
    mean_cost_unlev = float(cost_unlev.mean()) if len(cost_unlev) else float("nan")
    note = capacity_note(
        long_only=bool(long_only),
        mean_long_nav=float(sides["mean_long_nav"]),
        mean_short_nav=float(sides["mean_short_nav"]),
        mean_turnover=float(turnover.mean()) if len(turnover) else float("nan"),
        locate_pctile=float(locate_pctile),
    )
    return {
        "n_dates": float(len(net_s)),
        "n_names": float(pred.shape[1]),
        "lever": mean_lever,
        "mean_lever": mean_lever,
        "max_lever": float(lever_s.max()) if len(lever_s) else float("nan"),
        "gross_ir": _annualized_ir(lever_s * gross_s, ppy),
        "net_ir": _annualized_ir(net_s, ppy),
        "unlevered_gross_ir": _annualized_ir(gross_s, ppy),
        "unlevered_net_ir": _annualized_ir(net_unlev, ppy),
        "hit_rate": float((net_s > 0).mean()),
        "max_dd": _max_dd(net_s),
        "unlevered_max_dd": _max_dd(net_unlev),
        "mean_turnover": float(turnover.mean()),
        "mean_cost": float(cost.mean()),
        "mean_cost_unlev": mean_cost_unlev,
        "mean_cost_bp": (
            float(cost.mean()) * 1e4 if np.isfinite(float(cost.mean())) else float("nan")
        ),
        "mean_cost_unlev_bp": (
            mean_cost_unlev * 1e4 if np.isfinite(mean_cost_unlev) else float("nan")
        ),
        "round_trip_bps": float(round_trip_bps),
        "quantile": float(quantile),
        "vol_target": float(vol_target),
        "weighting": str(weighting),
        "hold_halflife": float(hold_halflife),
        "causal_vol": bool(causal_vol),
        "lever_cap": float(lever_cap),
        "long_only": bool(long_only),
        "min_names": float(min_names),
        "holding": hold_mode,
        "open_auction_bps": float(open_auction_bps),
        "borrow_bps": float(borrow_bps),
        "hedge_cost_bps": float(hedge_cost_bps),
        "moc_bps": float(moc_bps),
        "moo_bps": float(moo_bps),
        "session_exit_bps": float(session_exit_bps),
        "impact_vol_k": float(impact_vol_k),
        "thin_mult": float(thin_mult),
        "thin_pctile": float(thin_pctile),
        "locate_pctile": float(locate_pctile),
        "locate_haircut": float(locate_haircut) if not long_only else 0.0,
        "locate_frac": float(locate_frac) if not long_only else 0.0,
        "max_short_gross": float(max_short_gross) if not long_only else 0.0,
        "adv_floor_pctile": float(adv_floor_pctile),
        "ex_post_gap_k": float(ex_post_gap_k),
        "mean_long_nav": sides["mean_long_nav"],
        "mean_short_nav": sides["mean_short_nav"],
        "mean_gross": sides["mean_gross"],
        "mean_n_long": sides["mean_n_long"],
        "mean_n_short": sides["mean_n_short"],
        "mean_shorts_blocked": float(np.mean(n_blocked)) if len(n_blocked) else 0.0,
        "sleeve": sleeve,
        "cost_parts": cost_parts,
        "capacity_note": note,
        "mean_cs_ic": float(ics["ic"].mean()) if len(ics) else float("nan"),
        "mean_cs_ic_spearman": (
            float(ics["ic_spearman"].mean()) if len(ics) else float("nan")
        ),
        "cs_ic_tstat": (
            float(ics["ic"].mean() / (ics["ic"].std(ddof=1) / math.sqrt(len(ics))))
            if len(ics) > 2 and float(ics["ic"].std(ddof=1) or 0) > 1e-12
            else float("nan")
        ),
        "net": net_s,
        "gross": lever_s * gross_s,
        "unlevered_net": net_unlev,
        "leverage": lever_s,
        "cs_ic": ics,
        "weights": w_panel,
    }


def format_report(stats: dict[str, Any], *, checkpoint: Path, test_start: Any) -> str:
    parts = stats.get("cost_parts") or {}
    lines = [
        "=" * 72,
        "  LAST-BAR CROSS-SECTIONAL BOOK",
        "=" * 72,
        f"  Checkpoint  {checkpoint}",
        f"  Test from   {test_start}",
        f"  Names       {int(stats.get('n_names', 0))}   dates {int(stats.get('n_dates', 0))}",
        f"  Weighting   {stats.get('weighting', 'quantile')}  "
        f"quantile {stats.get('quantile', float('nan')):.2f}  "
        f"hold_hl {stats.get('hold_halflife', 0):.1f}  "
        f"holding {stats.get('holding', 'close')}"
        f"{'  LONG-ONLY' if stats.get('long_only') else ''}",
        f"  round-trip  {stats.get('round_trip_bps', float('nan')):.1f} bp"
        f"  moc {stats.get('moc_bps', 0):.1f} bp"
        f"  moo {stats.get('moo_bps', 0):.1f} bp"
        f"  auction {stats.get('open_auction_bps', 0):.1f} bp"
        f"  sess-exit {stats.get('session_exit_bps', 0):.1f} bp",
        f"  borrow {stats.get('borrow_bps', 0):.1f} bp"
        f"  hedge {stats.get('hedge_cost_bps', 0):.1f} bp"
        f"  impact_k {stats.get('impact_vol_k', 0):.1f}"
        f"  thin x{stats.get('thin_mult', 1):.1f}@{100 * float(stats.get('thin_pctile', 0) or 0):.0f}%",
        f"  locate     bottom {100 * float(stats.get('locate_pctile', 0) or 0):.0f}% turnover "
        f"haircut={float(stats.get('locate_haircut', 1) or 0):.2f} "
        f"frac={float(stats.get('locate_frac', 1) or 0):.2f} "
        f"max_short={float(stats.get('max_short_gross', 0.5) or 0):.2f}"
        f"  (mean names/date {stats.get('mean_shorts_blocked', 0):.2f})",
        f"  book NAV   long {stats.get('mean_long_nav', float('nan')):.3f}  "
        f"short {stats.get('mean_short_nav', float('nan')):.3f}  "
        f"gross {stats.get('mean_gross', float('nan')):.3f}",
        f"  Lever       mean {stats.get('mean_lever', stats.get('lever', float('nan'))):.3f}  "
        f"max {stats.get('max_lever', float('nan')):.3f}  "
        f"(vol target {stats.get('vol_target', float('nan')):.2f} annual"
        f"{', causal' if stats.get('causal_vol') else ', FULL-SAMPLE'})",
        "",
        f"  mean CS IC (Pearson)   {stats.get('mean_cs_ic', float('nan')):+.4f}  "
        f"t={stats.get('cs_ic_tstat', float('nan')):.2f}",
        f"  mean CS IC (Spearman)  {stats.get('mean_cs_ic_spearman', float('nan')):+.4f}",
        f"  unlevered gross IR     {stats.get('unlevered_gross_ir', float('nan')):+.3f}",
        f"  unlevered net IR       {stats.get('unlevered_net_ir', float('nan')):+.3f}",
        f"  unlevered max DD       {stats.get('unlevered_max_dd', float('nan')):+.3f}",
        f"  levered gross IR       {stats.get('gross_ir', float('nan')):+.3f}",
        f"  levered net IR         {stats.get('net_ir', float('nan')):+.3f}",
        f"  levered max DD         {stats.get('max_dd', float('nan')):+.3f}",
        f"  hit rate               {stats.get('hit_rate', float('nan')):.3f}",
        f"  mean turnover (1-way)  {stats.get('mean_turnover', float('nan')):.3f}",
        f"  unlev cost (NAV)       {stats.get('mean_cost_unlev', float('nan')):.5f}"
        f"  ({float(stats.get('mean_cost_unlev_bp') or float('nan')):.1f} bp)"
        f"  (rt {parts.get('round_trip', float('nan')):.5f}"
        f"  moc {parts.get('moc', 0):.5f}"
        f"  moo {parts.get('moo', 0):.5f}"
        f"  borrow {parts.get('borrow', 0):.5f}"
        f"  hedge {parts.get('hedge', 0):.5f}"
        f"  impact {parts.get('impact', 0):.5f})",
    ]
    sleeve = stats.get("sleeve") or {}
    if sleeve:
        lines.append(
            f"  long sleeve overnight up-rate   {float(sleeve.get('long_up_pct') or float('nan')):.1f}%"
            f"  excess {float(sleeve.get('long_excess_pp') or float('nan')):+.2f} pp"
            f" vs {float(sleeve.get('uncond_up_pct') or float('nan')):.1f}% up-floor"
            f"  n={int(float(sleeve.get('long_n') or 0))}"
        )
        if not stats.get("long_only"):
            lines.append(
                f"  short sleeve overnight down-rate {float(sleeve.get('short_down_pct') or float('nan')):.1f}%"
                f"  excess {float(sleeve.get('short_excess_pp') or float('nan')):+.2f} pp"
                f" vs {float(sleeve.get('uncond_down_pct') or float('nan')):.1f}% down-floor"
                f"  n={int(float(sleeve.get('short_n') or 0))}"
            )
    if stats.get("capacity_note"):
        lines.append(f"  capacity   {stats['capacity_note']}")
    if float(stats.get("ex_post_gap_k") or 0) > 0:
        lines.append(
            "  NOTE: ex_post_gap_k uses |realized| — sensitivity, not a tradable default."
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quantile / rank long-short backtest of last-bar scores.")
    p.add_argument("--checkpoint", default="checkpoints/forecast/best.pt")
    p.add_argument("--data", default=None)
    p.add_argument("--symbols", default=None)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument(
        "--weighting",
        choices=("quantile", "rank"),
        default="quantile",
        help="20%% tails (default) or centered CS rank (softer, usually lower IR)",
    )
    p.add_argument(
        "--hold-halflife",
        type=float,
        default=None,
        help="EWMA half-life in days for target weights "
        "(default: 0 overnight / 1 close-to-close; 5 is too slow for 1-day CS)",
    )
    p.add_argument(
        "--holding",
        choices=("auto", "close", "overnight", "open_fill"),
        default="auto",
        help="close-to-close roll vs overnight flatten vs open+N fill "
        "(auto: checkpoint label_return)",
    )
    p.add_argument(
        "--cost-bundle",
        default="",
        help="named overnight cost pack: paper, live_flat, live, live_locate, "
        "live_long_only, harsh, ex_post_gap, fill_live. CLI flags override fields.",
    )
    p.add_argument(
        "--live-costs",
        action="store_true",
        help="shorthand for --cost-bundle live_locate (honest LS) or live_long_only "
        "with --long-only. Unconstrained shorts: --cost-bundle live.",
    )
    p.add_argument(
        "--open-auction-bps",
        type=float,
        default=None,
        help="legacy extra one-way cost on the open *exit half-notional* vs official print",
    )
    p.add_argument(
        "--moc-bps",
        type=float,
        default=None,
        help="extra one-way MOC slippage vs the official close print (full |w|)",
    )
    p.add_argument(
        "--moo-bps",
        type=float,
        default=None,
        help="extra one-way MOO slippage vs the official open print (full |w|)",
    )
    p.add_argument(
        "--session-exit-bps",
        type=float,
        default=None,
        help="open_fill continuous-session exit vs MOO (default 0 unless fill_live bundle)",
    )
    p.add_argument(
        "--borrow-bps",
        type=float,
        default=None,
        help="overnight borrow fee on short notional (forced to 0 with --long-only)",
    )
    p.add_argument(
        "--hedge-cost-bps",
        type=float,
        default=None,
        help="extra daily cost for auctioning the residual sector/SPY hedge (~1 NAV)",
    )
    p.add_argument(
        "--impact-vol-k",
        type=float,
        default=None,
        help="extra one-way bps * max(vol_level,0) * |w| (vol_level known at t)",
    )
    p.add_argument(
        "--thin-mult",
        type=float,
        default=None,
        help="multiply MOC+MOO on names in the bottom --thin-pctile of CS turnover_z",
    )
    p.add_argument(
        "--thin-pctile",
        type=float,
        default=None,
        help="CS turnover_z percentile treated as thin / HTB (0.3 = bottom 30%%)",
    )
    p.add_argument(
        "--adv-floor-pctile",
        type=float,
        default=0.0,
        help="drop names below this CS turnover_z percentile before forming weights "
        "(0.67 = top tercile liquid sleeve)",
    )
    p.add_argument(
        "--locate-adv-pctile",
        type=float,
        default=None,
        help="block shorts in the bottom CS turnover_z percentile (HTB proxy; 0.3 = bottom 30%%)",
    )
    p.add_argument(
        "--locate-haircut",
        type=float,
        default=None,
        help="1.0 skip HTB shorts (default); 0.5 haircut them to half size (causal turnover_z)",
    )
    p.add_argument(
        "--locate-frac",
        type=float,
        default=None,
        help="locatable short NAV as a fraction of the 50/50 short leg (1.0 = full 0.5 NAV)",
    )
    p.add_argument(
        "--max-short-gross",
        type=float,
        default=None,
        help="cap short NAV after locate (default 0.5 = dollar-neutral short leg)",
    )
    p.add_argument(
        "--ls-haircut-experiment",
        action="store_true",
        help="VAL-gated experiment (NOT default): locate_haircut=0.5 and "
        "max_short_gross=0.3 on live_locate. Ignored with --long-only. "
        "Default remains skip HTB shorts (haircut=1, short NAV 0.5).",
    )
    p.add_argument(
        "--ex-post-gap-k",
        type=float,
        default=None,
        help="sensitivity: extra cost k*|overnight move|*|w| (uses realized; not default)",
    )
    p.add_argument("--cost-bps", type=float, default=None, help="round-trip cost in basis points")
    p.add_argument(
        "--vol-target",
        type=float,
        default=0.15,
        help="annualized vol target for the levered path (1.0 is a 100%% vol book)",
    )
    p.add_argument(
        "--lever-cap",
        type=float,
        default=3.0,
        help="cap on causal leverage (0 disables)",
    )
    p.add_argument(
        "--full-sample-vol",
        action="store_true",
        help="look-ahead full-sample vol targeting (old behavior; IR scale-invariant)",
    )
    p.add_argument("--long-only", action="store_true", help="long the top quantile/ranks only (no short leg, no locate)")
    p.add_argument(
        "--compare-long-only",
        action="store_true",
        help="also print the long-only overnight book on the same scores (no locate). "
        "Default on for overnight --live-costs long-short.",
    )
    p.add_argument(
        "--no-compare-long-only",
        action="store_true",
        help="do not auto-print the long-only comparison next to a live LS book",
    )
    p.add_argument(
        "--min-names",
        type=int,
        default=None,
        help="skip dates with fewer names than this (default: checkpoint cross_section_min_names)",
    )
    p.add_argument(
        "--all-names",
        action="store_true",
        help="score every parquet in data/, not only names listed in the checkpoint",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--json", default=None, help="write scalar stats to this path")
    p.add_argument("--cs-csv", default=None, help="write per-date CS IC series")
    return p


def cost_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Named bundle first, then explicit CLI overrides. Defaults match paper 10 bp."""
    name = str(getattr(args, "cost_bundle", "") or "").strip()
    if bool(getattr(args, "live_costs", False)) and not name:
        name = "live_long_only" if bool(getattr(args, "long_only", False)) else "live_locate"
    if bool(getattr(args, "long_only", False)) and name == "live":
        name = "live_long_only"
    bundle = resolve_cost_bundle(name) if name else None
    overrides = {
        "round_trip_bps": getattr(args, "cost_bps", None),
        "open_auction_bps": getattr(args, "open_auction_bps", None),
        "borrow_bps": getattr(args, "borrow_bps", None),
        "hedge_cost_bps": getattr(args, "hedge_cost_bps", None),
        "moc_bps": getattr(args, "moc_bps", None),
        "moo_bps": getattr(args, "moo_bps", None),
        "session_exit_bps": getattr(args, "session_exit_bps", None),
        "impact_vol_k": getattr(args, "impact_vol_k", None),
        "thin_mult": getattr(args, "thin_mult", None),
        "thin_pctile": getattr(args, "thin_pctile", None),
        "locate_pctile": getattr(args, "locate_adv_pctile", None),
        "ex_post_gap_k": getattr(args, "ex_post_gap_k", None),
    }
    merged = merge_cost_kwargs(bundle, **overrides)
    if not name and args.cost_bps is None:
        merged["round_trip_bps"] = 10.0
    return merged


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2
    model, state = load_forecaster(ckpt_path, device)
    data_cfg = DataConfig.from_dict(state["data_config"])
    hold_mode = str(args.holding or "auto").lower()
    if hold_mode == "auto":
        hold_mode = holding_for_label(getattr(data_cfg, "label_return", "close"))
    hold_hl = args.hold_halflife
    if hold_hl is None:
        hold_hl = 0.0 if hold_mode in ("overnight", "open_fill") else 1.0
    if float(args.vol_target) >= 0.999:
        print(
            "NOTE: vol_target=1 is a 100% vol toy. Report unlevered IR + 15% causal-vol "
            "max DD; do not headline this levered path.",
            file=sys.stderr,
        )
    try:
        files = resolve_data_files(
            args.data, data_cfg.data_dir, parse_symbols(args.symbols), interval=data_cfg.interval
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.data:
        data_path = resolve_path(args.data)
        universe_dir = data_path if data_path.is_dir() else data_path.parent
    else:
        universe_dir = data_cfg.data_dir
    panels = load_forecast_panels(files, data_cfg, universe_dir=universe_dir)
    bench = str(data_cfg.benchmark_symbol or "").upper()
    trained = {
        str(row.get("symbol", "")).upper()
        for row in (state.get("symbols") or [])
        if row.get("symbol")
    }
    if trained and not args.symbols and not args.all_names:
        files = [
            f
            for f in files
            if symbol_from_path(f) in trained or symbol_from_path(f) == bench
        ]
    min_names = (
        int(args.min_names)
        if args.min_names is not None
        else int(data_cfg.cross_section_min_names)
    )
    start = test_start_from_state(state)
    skip = {bench} if bench else set()
    if bool(getattr(data_cfg, "equities_only", False)):
        skip.update(s for s in panels if not is_equity_name(s))
    pred, realized = score_last_bars(
        model,
        panels,
        state,
        device,
        context=data_cfg.seq_len,
        start=start,
        batch_size=args.batch_size,
        exclude=frozenset(skip),
    )
    costs = cost_kwargs_from_args(args)
    tz = panel_feature_wide(panels, "turnover_z", pred) if len(pred) else None
    vol = panel_feature_wide(panels, "vol_level", pred) if len(pred) else None
    on_r = (
        panel_overnight_r_wide(panels, pred)
        if len(pred) and hold_mode == "overnight"
        else None
    )
    ppy = 252.0 if data_cfg.is_daily() else (52.0 if data_cfg.interval == "weekly" else 12.0)
    haircut = float(args.locate_haircut) if args.locate_haircut is not None else 1.0
    short_cap = float(args.max_short_gross) if args.max_short_gross is not None else 0.5
    if bool(getattr(args, "ls_haircut_experiment", False)) and not args.long_only:
        if args.locate_haircut is None:
            haircut = float(LS_HAIRCUT_EXPERIMENT["locate_haircut"])
        if args.max_short_gross is None:
            short_cap = float(LS_HAIRCUT_EXPERIMENT["max_short_gross"])
        print(
            "NOTE: --ls-haircut-experiment is NOT the default live_locate skip "
            f"(haircut={haircut:.2f}, max_short={short_cap:.2f}).",
            file=sys.stderr,
        )
    book_kw = dict(
        quantile=args.quantile,
        round_trip_bps=float(costs["round_trip_bps"]),
        vol_target=args.vol_target,
        periods_per_year=ppy,
        min_names=min_names,
        weighting=args.weighting,
        hold_halflife=hold_hl,
        causal_vol=not args.full_sample_vol,
        lever_cap=args.lever_cap,
        holding=hold_mode,
        open_auction_bps=float(costs["open_auction_bps"]),
        borrow_bps=float(costs["borrow_bps"]),
        hedge_cost_bps=float(costs["hedge_cost_bps"]),
        moc_bps=float(costs["moc_bps"]),
        moo_bps=float(costs["moo_bps"]),
        session_exit_bps=float(costs["session_exit_bps"]),
        impact_vol_k=float(costs["impact_vol_k"]),
        thin_mult=float(costs["thin_mult"]),
        thin_pctile=float(costs["thin_pctile"]),
        locate_pctile=float(costs["locate_pctile"]),
        locate_haircut=haircut,
        locate_frac=float(args.locate_frac) if args.locate_frac is not None else 1.0,
        max_short_gross=short_cap,
        ex_post_gap_k=float(costs["ex_post_gap_k"]),
        turnover_z=tz,
        vol_level=vol,
        overnight_r=on_r,
        adv_floor_pctile=float(args.adv_floor_pctile or 0.0),
    )
    stats = book_pnl(pred, realized, long_only=args.long_only, **book_kw)
    print(format_report(stats, checkpoint=ckpt_path, test_start=start))
    bundle_name = str(costs.get("name") or getattr(args, "cost_bundle", "") or "").lower()
    auto_compare = (
        not args.long_only
        and not bool(args.no_compare_long_only)
        and hold_mode == "overnight"
        and (
            bool(args.live_costs)
            or bundle_name in ("live", "live_locate")
        )
    )
    compare_lo = bool(args.compare_long_only) or auto_compare
    if compare_lo and not args.long_only:
        lo = book_pnl(pred, realized, long_only=True, **book_kw)
        print("\n--- long-only (no locate) ---\n")
        print(format_report(lo, checkpoint=ckpt_path, test_start=start))
        stats["long_only_compare"] = {
            k: v for k, v in lo.items() if k not in {"net", "gross", "cs_ic", "weights", "leverage", "unlevered_net"}
        }
    if args.json:
        skip_keys = {"net", "gross", "cs_ic", "weights", "leverage", "unlevered_net"}
        out = {k: v for k, v in stats.items() if k not in skip_keys}
        path = anchor_to_repo(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {path}", file=sys.stderr)
    if args.cs_csv:
        ics = stats.get("cs_ic")
        if isinstance(ics, pd.DataFrame) and len(ics):
            path = anchor_to_repo(args.cs_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            ics.to_csv(path, index_label="datetime")
            print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
