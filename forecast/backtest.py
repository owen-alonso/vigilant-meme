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


WEEKDAY_MASK_CHOICES = ("always", "flat_friday", "weekend_only", "flat_monday")
_WEEKDAY_MASK_ALIASES = {
    "": "always",
    "off": "always",
    "none": "always",
    "on": "always",
    "skip_friday": "flat_friday",
    "no_friday": "flat_friday",
    "no_weekend": "flat_friday",
    "friday_only": "weekend_only",
    "weekend": "weekend_only",
    "skip_monday": "flat_monday",
    "no_monday": "flat_monday",
}


def normalize_weekday_mask(mask: str | None) -> str:
    """Canonical weekday trade mask. Decision uses weekday(t) known at close t."""
    key = str(mask or "always").strip().lower().replace("-", "_")
    out = _WEEKDAY_MASK_ALIASES.get(key, key)
    if out not in WEEKDAY_MASK_CHOICES:
        raise ValueError(
            f"unknown weekday_mask {mask!r}; expected one of {WEEKDAY_MASK_CHOICES}"
        )
    return out


def weekday_mask_is_flat(ts: Any, mask: str | None) -> bool:
    """True if the overnight book starting at close ``t`` should stay flat.

    Friday flatten skips Friday close → Monday open (weekend gap).
    Weekend-only trades that Friday gap and flats every other night.
    Monday flatten skips Monday close → Tuesday open. All use weekday(t)
    only — next open is never consulted.
    """
    kind = normalize_weekday_mask(mask)
    if kind == "always":
        return False
    wd = int(pd.Timestamp(ts).dayofweek)  # Mon=0 … Fri=4
    if kind == "flat_friday":
        return wd == 4
    if kind == "weekend_only":
        return wd != 4
    if kind == "flat_monday":
        return wd == 0
    return False


def cs_std_by_date(panel: pd.DataFrame, *, min_names: int = 3) -> pd.Series:
    """Per-date cross-sectional standard deviation. NaN if too few names."""
    if panel is None or panel.empty:
        return pd.Series(dtype=np.float64)
    p = panel.sort_index()
    vals: dict[Any, float] = {}
    need = max(2, int(min_names))
    for ts in p.index:
        row = p.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        x = np.asarray(row, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < need:
            vals[ts] = float("nan")
        else:
            vals[ts] = float(np.std(x, ddof=1))
    return pd.Series(vals)


def causal_cc_dispersion(
    close: pd.DataFrame,
    *,
    min_names: int = 3,
    window: int = 1,
) -> pd.Series:
    """CS std of close-to-close log returns known at close ``t``.

    Date ``t`` uses ``close_t`` and ``close_{t-1}`` only — never ``open_{t+1}``.
    ``window>1`` is a trailing mean of that same-day CS std over dates ``≤ t``.
    """
    if close is None or close.empty:
        return pd.Series(dtype=np.float64)
    c = close.sort_index().astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(c.where(c > 0.0)).diff()
    raw = cs_std_by_date(r, min_names=min_names)
    w = max(1, int(window))
    if w == 1:
        return raw
    need = max(3, w // 3)
    return raw.rolling(w, min_periods=need).mean()


def trailing_on_resid_dispersion(
    resid: pd.DataFrame,
    *,
    window: int = 20,
    min_names: int = 3,
) -> pd.Series:
    """Trailing CS std of overnight residual ``y``.

    ``y[s]`` realizes at ``open_{s+1}``, so the value at ``t`` uses only
    dates ``s < t``. Date ``t``'s own overnight never enters.
    """
    raw = cs_std_by_date(resid, min_names=min_names)
    if raw.empty:
        return raw
    w = max(1, int(window))
    need = max(3, w // 3)
    return raw.shift(1).rolling(w, min_periods=need).mean()


def causal_disp_series(
    kind: str,
    *,
    close: pd.DataFrame | None = None,
    resid: pd.DataFrame | None = None,
    window: int = 1,
    min_names: int = 3,
) -> pd.Series:
    """Dispatch a causal dispersion proxy. Next open is never a feature."""
    key = str(kind or "cc").strip().lower().replace("-", "_")
    if key in ("cc", "close", "ret_1", "cc_trail"):
        return causal_cc_dispersion(
            close if close is not None else pd.DataFrame(),
            min_names=min_names,
            window=window,
        )
    if key in ("on_trail", "on", "resid", "y", "overnight"):
        return trailing_on_resid_dispersion(
            resid if resid is not None else pd.DataFrame(),
            window=window,
            min_names=min_names,
        )
    raise ValueError(f"unknown disp kind {kind!r}; expected cc or on_trail")


def trailing_mean_cs_ic(
    pred: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    window: int,
    min_names: int = 3,
    kind: str = "pearson",
    min_obs: int | None = None,
) -> pd.Series:
    """Causal trailing mean CS IC known at close ``t``.

    Per-date IC on ``s`` uses the overnight label that realizes at the next
    open, so the value at ``s`` is *not* known at close ``s``. The series
    at ``t`` is the mean of ICs on the prior ``window`` sessions
    ``s < t`` only. Date ``t``'s own overnight realization never enters.
    Warmup (too few prior ICs) is NaN — caller should leave those dates on.
    """
    ics = cs_ic_by_date(pred, realized, min_names=min_names)
    if ics.empty:
        return pd.Series(dtype=np.float64)
    col = "ic_spearman" if str(kind).lower() in ("spearman", "rank") else "ic"
    if col not in ics.columns:
        col = "ic"
    prior = ics[col].astype(np.float64).sort_index()
    # shift(1): at t, the newest term is IC[t-1], known after open t / by close t.
    w = max(1, int(window))
    need = int(min_obs) if min_obs is not None else max(8, w // 3)
    return prior.shift(1).rolling(w, min_periods=need).mean()


def soft_ic_gross_scale(
    trail_ic: float,
    tau: float,
    *,
    s_max: float = 1.0,
) -> float:
    """``clip(trail_IC / τ, 0, s_max)``. NaN warmup or τ≤0 → full gross (1).

    ``s_max > 1`` allows a modest lift on strong-IC nights. Not a hard flatten.
    """
    t = float(tau)
    if not np.isfinite(t) or t <= 1e-12:
        return 1.0
    x = float(trail_ic)
    if not np.isfinite(x):
        return 1.0
    hi = float(s_max) if np.isfinite(s_max) and s_max > 0 else 1.0
    return float(np.clip(x / t, 0.0, hi))


def conviction_long_weights(
    scores: pd.Series,
    *,
    q: float,
    abs_tau: float = 0.0,
    min_names: int = 3,
    require_above_median: bool = False,
) -> pd.Series:
    """Equal-weight names with ``pred >= nanquantile(q)`` and optional ``|pred|`` floor.

    ``q=0.90`` is the top 10% (accuracy IDEA E). Matches ``cs_top_abs_mask``.
    ``require_above_median`` also drops names at or below the date CS median
    (IDEA I stack). Default off. Next open is never used.
    """
    s = pd.to_numeric(scores, errors="coerce")
    w = pd.Series(0.0, index=s.index, dtype=np.float64)
    finite = s.notna() & np.isfinite(s)
    if int(finite.sum()) < int(min_names):
        return w
    cut = float(np.nanquantile(s.to_numpy(dtype=np.float64), float(q)))
    keep = finite & (s >= cut)
    tau = float(abs_tau)
    if tau > 0.0:
        keep = keep & (s.abs() >= tau)
    if require_above_median:
        med = float(np.nanmedian(s.to_numpy(dtype=np.float64)))
        if np.isfinite(med):
            keep = keep & (s > med)
    n = int(keep.sum())
    if n <= 0:
        return w
    w.loc[keep] = 1.0 / float(n)
    return w


def ls_short_aligned_weights(
    scores: pd.Series,
    *,
    short_q: float,
    abs_tau: float = 0.0,
    long_q: float = 0.80,
    min_names: int = 3,
) -> pd.Series:
    """Dollar-neutral LS: long top ``long_q``, short bottom-q ∩ optional |pred| floor.

    ``short_q=0.20`` is the bottom 20%. ``long_q=0.80`` is the top 20%.
    Default off in ``book_pnl``. Next open is never used.
    """
    s = pd.to_numeric(scores, errors="coerce")
    w = pd.Series(0.0, index=s.index, dtype=np.float64)
    finite = s.notna() & np.isfinite(s)
    if int(finite.sum()) < int(min_names):
        return w
    arr = s.to_numpy(dtype=np.float64)
    long_cut = float(np.nanquantile(arr, float(long_q)))
    short_cut = float(np.nanquantile(arr, float(short_q)))
    long_keep = finite & (s >= long_cut)
    short_keep = finite & (s <= short_cut)
    tau = float(abs_tau)
    if tau > 0.0:
        short_keep = short_keep & (s.abs() >= tau)
    long_keep = long_keep & ~short_keep
    n_long = int(long_keep.sum())
    n_short = int(short_keep.sum())
    if n_long <= 0 or n_short <= 0:
        return w
    w.loc[long_keep] = 0.5 / float(n_long)
    w.loc[short_keep] = -0.5 / float(n_short)
    return w


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


def as_quantile_frac(value: float) -> float:
    """Accept 0.20 or 20 (percent). Non-positive / non-finite → 0 (off)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x) or x <= 0:
        return 0.0
    if x > 1.0:
        x = x / 100.0
    return float(x)


def sticky_long_step(
    scores: pd.Series,
    held: set[Any] | frozenset[Any] | None,
    *,
    q_enter: float,
    q_exit: float,
    min_names: int = 8,
) -> tuple[pd.Series, set[Any]]:
    """Equal-weight sticky longs: enter top ``q_enter``, hold while in top ``q_exit``.

    Ranks use only today's scores (known at close t). ``held`` is yesterday's
    active set. Empty set → flat that night (caller keeps the date in IR).
    """
    s = scores.dropna()
    w = pd.Series(0.0, index=s.index, dtype=np.float64)
    n = int(s.size)
    if n < max(2, int(min_names)):
        return w, set()
    qe = as_quantile_frac(q_enter)
    qx = as_quantile_frac(q_exit)
    if qe <= 0 or qx <= qe:
        return w, set()
    qe = min(0.49, max(0.05, qe))
    qx = min(0.90, max(qe + 1e-9, qx))
    k_enter = max(1, int(math.floor(n * qe)))
    k_exit = max(k_enter + 1, int(math.floor(n * qx)))
    order = s.sort_values()
    enter = set(order.index[-k_enter:])
    keep_band = set(order.index[-k_exit:])
    prev = set(held or ()) & set(s.index)
    new_held = (prev & keep_band) | enter
    if not new_held:
        return w, set()
    wt = 1.0 / float(len(new_held))
    for name in new_held:
        w.loc[name] = wt
    return w, new_held


def last_sticky_held(
    pred: pd.DataFrame,
    *,
    q_enter: float,
    q_exit: float,
    min_names: int,
) -> set[Any]:
    """Walk scores date-by-date (causal). Used to warm VAL/TEST from prior splits."""
    held: set[Any] = set()
    if pred is None or pred.empty:
        return held
    for ts in pred.index:
        row = pred.loc[ts]
        if int(row.dropna().size) < int(min_names):
            continue
        _w, held = sticky_long_step(
            row, held, q_enter=q_enter, q_exit=q_exit, min_names=min_names
        )
    return held


def name_membership_churn(weights: pd.DataFrame) -> dict[str, float]:
    """Night-to-night long-set Hamming fraction (not flatten one-way cost turn)."""
    empty = {
        "mean_name_churn": float("nan"),
        "mean_n_held": float("nan"),
        "sticky_coverage": float("nan"),
    }
    if weights is None or weights.empty:
        return empty
    held = weights.gt(1e-12)
    prev = held.shift(1)
    prev = prev.where(prev.notna(), False).astype(bool)
    delta = (held.astype("int8") - prev.astype("int8")).abs().sum(axis=1)
    n_univ = float(max(1, int(weights.shape[1])))
    n_held = held.sum(axis=1).astype(np.float64)
    invested = n_held > 0
    return {
        "mean_name_churn": float((delta.astype(np.float64) / n_univ).mean()),
        "mean_n_held": float(n_held.mean()) if len(n_held) else float("nan"),
        "sticky_coverage": float(invested.mean()) if len(invested) else float("nan"),
    }


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


def resize_long_only(
    weights: pd.Series,
    scores: pd.Series,
    *,
    vol: pd.Series | None = None,
    long_size: str = "equal",
    conf_pctile: float = 0.0,
    conf_abs: float = 0.0,
) -> pd.Series:
    """Causal long-only resize known at close t. Does not use labels.

    ``conf_pctile`` drops longs whose ``|pred|`` is below that CS percentile
    of ``|pred|`` on the same date. ``conf_abs`` drops longs below a TRAIN
    ``|pred|`` floor (0=off). ``long_size`` reweights remaining longs:
    equal, ``abs_pred``, or ``inv_vol`` (missing vol → date median).
    """
    out = weights.astype(np.float64).copy()
    long = out > 1e-12
    if not bool(long.any()):
        return out
    mag = scores.reindex(out.index).abs()
    tau = float(conf_abs)
    if tau > 0.0:
        keep_abs = long & mag.notna() & np.isfinite(mag) & (mag >= tau)
        out.loc[long & ~keep_abs] = 0.0
        long = out > 1e-12
        if not bool(long.any()):
            return out * 0.0
    p = float(np.clip(conf_pctile, 0.0, 0.95))
    if p > 0:
        finite = mag.notna() & np.isfinite(mag)
        if int(finite.sum()) >= 3:
            cut = float(np.nanpercentile(mag[finite], 100.0 * p))
            keep = long & finite & (mag >= cut)
            out.loc[long & ~keep] = 0.0
            long = out > 1e-12
    if not bool(long.any()):
        return out * 0.0
    mode = str(long_size or "equal").strip().lower()
    if mode == "abs_pred":
        mag = scores.reindex(out.index).abs()
        mag = mag.where(long & np.isfinite(mag), 0.0)
        tot = float(mag.sum())
        if tot > 1e-12:
            return mag / tot
    elif mode == "inv_vol" and vol is not None:
        v = pd.to_numeric(vol.reindex(out.index), errors="coerce")
        finite = v.notna() & np.isfinite(v) & (v > 1e-8)
        if bool(finite.any()):
            fill = float(np.nanmedian(v[finite].to_numpy()))
            inv = 1.0 / v.where(finite, fill)
            inv = inv.where(long, 0.0)
            tot = float(inv.sum())
            if tot > 1e-12:
                return inv / tot
    n = int(long.sum())
    out.loc[long] = 1.0 / n
    out.loc[~long] = 0.0
    return out


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
    long_size: str = "equal",
    conf_pctile: float = 0.0,
    conf_abs: float = 0.0,
    conviction_q: float = 0.0,
    stack_long_half: bool = False,
    short_conviction_q: float = 0.0,
    short_conf_abs: float = 0.0,
    ic_gate_window: int = 0,
    ic_gate_tau: float = 0.0,
    ic_gate_kind: str = "pearson",
    ic_gate_trail: pd.Series | None = None,
    ic_scale_window: int = 0,
    ic_scale_tau: float = 0.0,
    ic_scale_trail: pd.Series | None = None,
    ic_scale_smax: float = 1.0,
    weekday_mask: str = "always",
    disp_gate_trail: pd.Series | None = None,
    disp_gate_tau: float = float("nan"),
    disp_gate_kind: str = "",
    disp_gate_window: int = 0,
    close_px: pd.DataFrame | None = None,
    sticky_q_enter: float = 0.0,
    sticky_q_exit: float = 0.0,
    sticky_held0: set[Any] | frozenset[Any] | None = None,
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
    trail_ic = None
    gate_w = int(ic_gate_window or 0)
    if ic_gate_trail is not None:
        trail_ic = ic_gate_trail
        gate_w = gate_w if gate_w > 0 else max(1, int(getattr(ic_gate_trail, "name", 0) or 1))
    elif gate_w > 0:
        trail_ic = trailing_mean_cs_ic(
            pred,
            realized,
            window=gate_w,
            min_names=int(min_names),
            kind=str(ic_gate_kind or "pearson"),
        )
    raw_w: list[pd.Series] = []
    gross: list[float] = []
    kept: list[Any] = []
    n_gated = 0
    n_gate_dates = 0
    scale_w = int(ic_scale_window or 0)
    scale_tau = float(ic_scale_tau or 0.0)
    scale_smax = float(ic_scale_smax) if np.isfinite(float(ic_scale_smax or 0)) else 1.0
    if scale_smax <= 0:
        scale_smax = 1.0
    trail_scale = ic_scale_trail
    if trail_scale is None and scale_w > 0 and scale_tau > 1e-12:
        trail_scale = trailing_mean_cs_ic(
            pred,
            realized,
            window=scale_w,
            min_names=int(min_names),
            kind=str(ic_gate_kind or "pearson"),
        )
    scale_on = trail_scale is not None and scale_tau > 1e-12
    scale_vals: list[float] = []
    n_scale_partial = 0
    n_scale_boost = 0
    n_scale_flat = 0
    n_scale_dates = 0
    wd_kind = normalize_weekday_mask(weekday_mask)
    n_wd_flat = 0
    n_wd_dates = 0
    disp_trail = disp_gate_trail
    disp_tau = float(disp_gate_tau) if disp_gate_tau is not None else float("nan")
    disp_kind = str(disp_gate_kind or "").strip().lower()
    disp_w = int(disp_gate_window or 0)
    n_disp_flat = 0
    n_disp_dates = 0
    if disp_trail is None and disp_kind and np.isfinite(disp_tau):
        if disp_kind in ("on_trail", "on", "resid", "y", "overnight"):
            disp_trail = trailing_on_resid_dispersion(
                realized, window=max(1, disp_w or 20), min_names=int(min_names)
            )
        elif close_px is not None and not close_px.empty:
            disp_trail = causal_cc_dispersion(
                close_px, min_names=int(min_names), window=max(1, disp_w or 1)
            )
    disp_on = disp_trail is not None and np.isfinite(disp_tau)
    sticky_qe = as_quantile_frac(sticky_q_enter)
    sticky_qx = as_quantile_frac(sticky_q_exit)
    sticky_on = bool(long_only and sticky_qe > 0 and sticky_qx > sticky_qe)
    sticky_held: set[Any] = set(sticky_held0 or ()) if sticky_on else set()
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
        if sticky_on:
            w, sticky_held = sticky_long_step(
                pair_all["p"],
                sticky_held,
                q_enter=sticky_qe,
                q_exit=sticky_qx,
                min_names=int(min_names),
            )
        elif long_only and float(conviction_q or 0.0) > 0.0:
            w = conviction_long_weights(
                pair_all["p"],
                q=float(conviction_q),
                abs_tau=float(conf_abs or 0.0),
                min_names=int(min_names),
                require_above_median=bool(stack_long_half),
            )
        elif (not long_only) and float(short_conviction_q or 0.0) > 0.0:
            q_lo = float(quantile)
            long_q = (1.0 - q_lo) if q_lo <= 0.5 else q_lo
            w = ls_short_aligned_weights(
                pair_all["p"],
                short_q=float(short_conviction_q),
                abs_tau=float(short_conf_abs or 0.0),
                long_q=float(long_q),
                min_names=int(min_names),
            )
        else:
            w = date_weights(
                pair_all["p"],
                weighting=weighting,
                quantile=quantile,
                long_only=long_only,
            )
        if long_only and float(conviction_q or 0.0) <= 0.0 and (
            str(long_size or "equal").lower() != "equal"
            or float(conf_pctile) > 0
            or float(conf_abs or 0.0) > 0
        ):
            vol_row = None
            if vol_level is not None and ts in vol_level.index:
                vol_row = vol_level.loc[ts].reindex(w.index)
            w = resize_long_only(
                w,
                pair_all["p"].reindex(w.index),
                vol=vol_row,
                long_size=str(long_size or "equal"),
                conf_pctile=float(conf_pctile),
                conf_abs=float(conf_abs or 0.0),
            )
        if trail_ic is not None:
            n_gate_dates += 1
            tval = (
                float(trail_ic.loc[ts])
                if ts in trail_ic.index
                else float("nan")
            )
            # NaN warmup: leave the date on (same as skip-IC shrink).
            if np.isfinite(tval) and tval < float(ic_gate_tau):
                w = w * 0.0
                n_gated += 1
        if scale_on:
            n_scale_dates += 1
            sval = (
                float(trail_scale.loc[ts])
                if ts in trail_scale.index
                else float("nan")
            )
            s = soft_ic_gross_scale(sval, scale_tau, s_max=scale_smax)
            scale_vals.append(s)
            if abs(s - 1.0) > 1e-12:
                n_scale_partial += 1
            if s > 1.0 + 1e-12:
                n_scale_boost += 1
            if s <= 1e-12:
                n_scale_flat += 1
            w = w * s
        if wd_kind != "always":
            n_wd_dates += 1
            if weekday_mask_is_flat(ts, wd_kind):
                w = w * 0.0
                n_wd_flat += 1
        if disp_on:
            n_disp_dates += 1
            dval = (
                float(disp_trail.loc[ts])
                if ts in disp_trail.index
                else float("nan")
            )
            # NaN warmup: leave the date on.
            if np.isfinite(dval) and dval >= disp_tau:
                w = w * 0.0
                n_disp_flat += 1
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
        "long_size": str(long_size or "equal"),
        "conf_pctile": float(conf_pctile),
        "conf_abs": float(conf_abs or 0.0),
        "conviction_q": float(conviction_q or 0.0),
        "stack_long_half": bool(stack_long_half),
        "short_conviction_q": float(short_conviction_q or 0.0),
        "short_conf_abs": float(short_conf_abs or 0.0),
        "ic_gate_window": float(gate_w),
        "ic_gate_tau": float(ic_gate_tau) if gate_w > 0 else 0.0,
        "ic_gate_kind": str(ic_gate_kind or "pearson") if gate_w > 0 else "",
        "ic_gate_n_flat": float(n_gated),
        "ic_gate_n_dates": float(n_gate_dates),
        "ic_gate_coverage": (
            float(1.0 - n_gated / n_gate_dates) if n_gate_dates else float("nan")
        ),
        "ic_scale_window": float(scale_w if scale_on else 0),
        "ic_scale_tau": float(scale_tau if scale_on else 0.0),
        "ic_scale_smax": float(scale_smax if scale_on else 1.0),
        "mean_ic_scale": (
            float(np.mean(scale_vals)) if scale_vals else float("nan")
        ),
        "ic_scale_n_partial": float(n_scale_partial),
        "ic_scale_n_boost": float(n_scale_boost),
        "ic_scale_n_flat": float(n_scale_flat),
        "ic_scale_n_dates": float(n_scale_dates),
        "ic_scale_coverage": (
            float(1.0 - n_scale_flat / n_scale_dates) if n_scale_dates else float("nan")
        ),
        "weekday_mask": wd_kind,
        "weekday_n_flat": float(n_wd_flat),
        "weekday_n_dates": float(n_wd_dates),
        "weekday_coverage": (
            float(1.0 - n_wd_flat / n_wd_dates)
            if n_wd_dates
            else (1.0 if wd_kind == "always" else float("nan"))
        ),
        "disp_gate_kind": disp_kind if disp_on else "",
        "disp_gate_window": float(disp_w if disp_on else 0),
        "disp_gate_tau": float(disp_tau) if disp_on else 0.0,
        "disp_gate_n_flat": float(n_disp_flat),
        "disp_gate_n_dates": float(n_disp_dates),
        "disp_gate_coverage": (
            float(1.0 - n_disp_flat / n_disp_dates) if n_disp_dates else float("nan")
        ),
        **name_membership_churn(w_panel),
        "sticky_q_enter": float(sticky_qe if sticky_on else 0.0),
        "sticky_q_exit": float(sticky_qx if sticky_on else 0.0),
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
        f"{'  LONG-ONLY' if stats.get('long_only') else ''}"
        + (
            f"  ic_gate W={int(stats.get('ic_gate_window') or 0)} "
            f"τ={float(stats.get('ic_gate_tau') or 0):+.3f} "
            f"cover {100 * float(stats.get('ic_gate_coverage') or float('nan')):.0f}%"
            if float(stats.get("ic_gate_window") or 0) > 0
            else ""
        )
        + (
            f"  ic_scale W={int(stats.get('ic_scale_window') or 0)} "
            f"τ={float(stats.get('ic_scale_tau') or 0):+.3f} "
            f"s_max={float(stats.get('ic_scale_smax') or 1):.2f} "
            f"mean_s {float(stats.get('mean_ic_scale') or float('nan')):.2f}"
            if float(stats.get("ic_scale_window") or 0) > 0
            else ""
        )
        + (
            f"  weekday_mask={stats.get('weekday_mask')} "
            f"cover {100 * float(stats.get('weekday_coverage') or float('nan')):.0f}%"
            if str(stats.get("weekday_mask") or "always") not in ("", "always")
            else ""
        )
        + (
            f"  disp_gate {stats.get('disp_gate_kind')} "
            f"W={int(stats.get('disp_gate_window') or 0)} "
            f"τ={float(stats.get('disp_gate_tau') or 0):.4f} "
            f"cover {100 * float(stats.get('disp_gate_coverage') or float('nan')):.0f}%"
            if str(stats.get("disp_gate_kind") or "")
            else ""
        )
        + (
            f"  sticky enter={100 * float(stats.get('sticky_q_enter') or 0):.0f}% "
            f"exit={100 * float(stats.get('sticky_q_exit') or 0):.0f}% "
            f"name_churn {float(stats.get('mean_name_churn') or float('nan')):.3f} "
            f"cover {100 * float(stats.get('sticky_coverage') or float('nan')):.0f}%"
            if float(stats.get("sticky_q_enter") or 0) > 0
            else ""
        ),
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
        "--long-size",
        choices=("equal", "abs_pred", "inv_vol"),
        default="equal",
        help="long-only resize among the selected sleeve (equal default; "
        "abs_pred / inv_vol are causal). Ignored unless --long-only.",
    )
    p.add_argument(
        "--conf-pctile",
        type=float,
        default=0.0,
        help="long-only: drop sleeve names with |pred| below this CS percentile "
        "(0=off, 0.5=date-median conviction). Causal; ignored unless --long-only.",
    )
    p.add_argument(
        "--conviction-q",
        type=float,
        default=0.0,
        help="long-only accuracy-style sleeve: keep pred >= nanquantile(q) "
        "(0=off; 0.90 = top 10%%). Optional live path; default stays q20. "
        "Ignored unless --long-only.",
    )
    p.add_argument(
        "--conf-abs",
        type=float,
        default=0.0,
        help="long-only: drop names with |pred| below this TRAIN floor (0=off). "
        "Use with --conviction-q for IDEA E's sleeve. Default off.",
    )
    p.add_argument(
        "--ic-gate-window",
        type=int,
        default=0,
        help="causal trailing overnight CS-IC sessions used at close t "
        "(0=off). Decision uses only dates < t; t's own overnight is never in.",
    )
    p.add_argument(
        "--ic-gate-tau",
        type=float,
        default=0.0,
        help="trade the overnight book only if trailing mean CS IC >= this "
        "(flat otherwise). Used with --ic-gate-window.",
    )
    p.add_argument(
        "--ic-scale-window",
        type=int,
        default=0,
        help="soft trailing CS-IC gross scale window (0=off). Causal dates < t. "
        "Different from --ic-gate-window (binary flatten). Not the live default.",
    )
    p.add_argument(
        "--ic-scale-tau",
        type=float,
        default=0.0,
        help="s_t = clip(trail_IC / tau, 0, s_max). tau>0 required. Used with "
        "--ic-scale-window. Full gross when trail >= tau; flat when trail <= 0.",
    )
    p.add_argument(
        "--ic-scale-smax",
        type=float,
        default=1.25,
        help="cap on soft IC scale (1.0 = no lift; 1.25 = modest strong-IC lift). "
        "Used with --ic-scale-window. Not the live default.",
    )
    p.add_argument(
        "--weekday-mask",
        default="always",
        choices=list(WEEKDAY_MASK_CHOICES),
        help="causal calendar mask using weekday(t) at close t: always | "
        "flat_friday (no weekend gap) | weekend_only (Friday overnight only) | "
        "flat_monday (no Mon close→Tue open). Default always (off).",
    )
    p.add_argument(
        "--disp-gate-kind",
        default="",
        choices=("", "cc", "on_trail"),
        help="causal CS-dispersion stress gate: cc = same-day close-to-close "
        "CS std (dates ≤ t); on_trail = trailing overnight residual CS std "
        "(dates < t only). Empty = off.",
    )
    p.add_argument(
        "--disp-gate-window",
        type=int,
        default=1,
        help="trailing sessions for --disp-gate-kind (1 = same-day cc). "
        "on_trail always excludes date t.",
    )
    p.add_argument(
        "--disp-gate-tau",
        type=float,
        default=float("nan"),
        help="flat overnight when causal dispersion >= this (NaN = off).",
    )
    p.add_argument(
        "--sticky-q-enter",
        type=float,
        default=0.0,
        help="long-only hysteresis: enter when rank is in the top this quantile "
        "(0=off, 0.15 or 15 = top 15%%). TRAIN/VAL gated in overnight_shorting; "
        "not the live default.",
    )
    p.add_argument(
        "--sticky-q-exit",
        type=float,
        default=0.0,
        help="long-only hysteresis: keep a prior long until rank falls below "
        "this quantile (must exceed --sticky-q-enter). 0=off.",
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
        long_size=str(getattr(args, "long_size", "equal") or "equal"),
        conf_pctile=float(getattr(args, "conf_pctile", 0.0) or 0.0),
        conf_abs=float(getattr(args, "conf_abs", 0.0) or 0.0),
        conviction_q=float(getattr(args, "conviction_q", 0.0) or 0.0),
        ic_gate_window=int(getattr(args, "ic_gate_window", 0) or 0),
        ic_gate_tau=float(getattr(args, "ic_gate_tau", 0.0) or 0.0),
        ic_scale_window=int(getattr(args, "ic_scale_window", 0) or 0),
        ic_scale_tau=float(getattr(args, "ic_scale_tau", 0.0) or 0.0),
        ic_scale_smax=float(getattr(args, "ic_scale_smax", 1.25) or 1.25),
        weekday_mask=str(getattr(args, "weekday_mask", "always") or "always"),
        disp_gate_kind=str(getattr(args, "disp_gate_kind", "") or ""),
        disp_gate_window=int(getattr(args, "disp_gate_window", 0) or 0),
        disp_gate_tau=float(getattr(args, "disp_gate_tau", float("nan"))),
        close_px=panel_feature_wide(panels, "close", pred) if len(pred) else None,
        sticky_q_enter=float(getattr(args, "sticky_q_enter", 0.0) or 0.0),
        sticky_q_exit=float(getattr(args, "sticky_q_exit", 0.0) or 0.0),
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
