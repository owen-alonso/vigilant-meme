"""Causal accuracy levers for the overnight residual book.

Protocol knobs, not a bigger model. Promote only on locked val
(≥0.003 CS IC lift and do not kill val-2017). Default overnight ``y``
is unchanged unless a separate estimand wins that gate. Live long-only
is first-class. Trailing CS-IC shrink of *scores* already lost on
close-to-close val; leverage/sizing shrink is the remaining overlay
because it does not retarget Pearson IC.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from forecast.ridge import trailing_skip_ic_stats, univariate_cs_ics
from forecast.universe import is_equity_name


def sign_consistency_weights(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
    min_abs: float = 0.003,
) -> np.ndarray:
    """Per-column weight in ``[0, 1]`` from train-year sign agreement.

    Fraction of train years whose univariate CS IC matches the pooled train
    sign, among years with ``|IC| >= min_abs``. Soft shrink, not a hard drop:
    year-flipping OHLC is downweighted instead of deleted (hard year-stable
    masks lost locked val). Fit on train only. Columns with no finite year
    ICs keep weight 1.
    """
    from forecast.ridge import year_feature_ics

    f = int(x.shape[1]) if x.ndim == 2 else 0
    if f == 0 or x.shape[0] == 0:
        return np.ones(f, dtype=np.float64)
    pooled = univariate_cs_ics(x, y, dates, min_names=min_names)
    by_year = year_feature_ics(x, y, dates, min_names=min_names)
    if not by_year:
        return np.ones(f, dtype=np.float64)
    stacked = np.stack(list(by_year.values()), axis=0)
    out = np.ones(f, dtype=np.float64)
    for j in range(f):
        col = stacked[:, j]
        finite = col[np.isfinite(col) & (np.abs(col) >= float(min_abs))]
        if finite.size < 2:
            continue
        sign = float(pooled[j]) if np.isfinite(pooled[j]) else float(np.sign(finite.mean()))
        if abs(sign) < 1e-12:
            sign = float(np.sign(finite.mean()))
        if abs(sign) < 1e-12:
            continue
        out[j] = float((np.sign(finite) == np.sign(sign)).mean())
    return out


def bake_col_scale(weights: np.ndarray, col_scale: np.ndarray | None) -> np.ndarray:
    """Fold train-only column scales into skip weights so inference is ``x @ w``."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if col_scale is None:
        return w.astype(np.float32)
    s = np.asarray(col_scale, dtype=np.float64).reshape(-1)
    if s.size != w.size:
        raise ValueError(f"col_scale size {s.size} != weights {w.size}")
    return (w * s).astype(np.float32)


def long_only_cs_target(
    y: np.ndarray,
    *,
    quantile: float = 0.2,
) -> np.ndarray:
    """Within-date long-sleeve target: ``max(rank_pctile - (1-q), 0)``.

    Aligns the skip with a long-only book (top quantile) instead of fitting
    LS ranks and dropping shorts at backtest. Bottom names are zeros, not
    negative ranks. Caller should still date-demean if the ridge does.
    """
    yd = np.asarray(y, dtype=np.float64).copy()
    n = int(yd.size)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    q = min(0.49, max(0.05, float(quantile)))
    order = np.argsort(yd, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    pct = (ranks + 0.5) / n
    return np.maximum(pct - (1.0 - q), 0.0)


def within_date_zscore(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Causal-safe CS z-score of a readout (uses that date's scores only)."""
    out = np.asarray(values, dtype=np.float64).copy()
    d = np.asarray(dates, dtype=np.int64)
    for key in np.unique(d):
        sel = d == key
        row = out[sel]
        finite = np.isfinite(row)
        if int(finite.sum()) < 2:
            out[sel] = 0.0
            continue
        mu = float(row[finite].mean())
        sd = float(row[finite].std())
        if sd < 1e-8:
            out[sel] = 0.0
            continue
        z = np.zeros(int(sel.sum()), dtype=np.float64)
        z[finite] = (row[finite] - mu) / sd
        out[sel] = z
    return out


def blend_readouts(
    preds: Sequence[np.ndarray],
    dates: np.ndarray,
    *,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Equal-weight (or weighted) blend of within-date z-scored readouts."""
    if not preds:
        raise ValueError("need at least one readout")
    zs = [within_date_zscore(p, dates) for p in preds]
    if weights is None:
        w = np.ones(len(zs), dtype=np.float64) / len(zs)
    else:
        w = np.asarray(list(weights), dtype=np.float64)
        w = w / max(float(w.sum()), 1e-12)
    out = np.zeros_like(zs[0])
    for wi, z in zip(w, zs):
        out = out + float(wi) * z
    return out


def sleeve_row_mask(
    turnover_z: np.ndarray,
    dates: np.ndarray,
    *,
    floor: float,
) -> np.ndarray:
    """True for names at/above the within-date ``turnover_z`` percentile."""
    z = np.asarray(turnover_z, dtype=np.float64)
    d = np.asarray(dates, dtype=np.int64)
    keep = np.zeros(z.shape[0], dtype=bool)
    p = float(floor)
    if p <= 0:
        return np.ones(z.shape[0], dtype=bool)
    for key in np.unique(d):
        sel = d == key
        row = z[sel]
        finite = np.isfinite(row)
        if int(finite.sum()) < 5:
            keep[sel] = finite
            continue
        cut = float(np.nanpercentile(row[finite], 100.0 * p))
        local = np.zeros(int(sel.sum()), dtype=bool)
        local[finite] = row[finite] >= cut
        keep[sel] = local
    return keep


def dollar_adv_series(panel: pd.DataFrame) -> pd.Series:
    """``close * volume`` proxy. Not vendor ADV; split-adjusted parquet.

    A later corporate-action-aware field should replace this with unadjusted
    dollar volume or a vendor ADV tape. Do not treat this as share-count truth.
    """
    close = pd.to_numeric(panel["close"], errors="coerce").astype(np.float64)
    vol = pd.to_numeric(panel["volume"], errors="coerce").astype(np.float64)
    return close * vol


def train_era_liquid_names(
    panels: dict[str, pd.DataFrame],
    trade_names: Sequence[str],
    train_end: Any,
    *,
    pctile: float,
    min_usd: float = 0.0,
    train_from: Any | None = None,
) -> list[str]:
    """Keep names whose *train-era* median dollar ADV is at/above the CS percentile.

    Membership is locked from sessions ``< train_end`` (and ``>= train_from``).
    Val/test cannot resurrect a name that was illiquid in train.
    """
    names = [str(s) for s in trade_names]
    if float(pctile) <= 0 and float(min_usd) <= 0:
        return names
    end = pd.Timestamp(train_end)
    start = pd.Timestamp(train_from) if train_from is not None else None
    adv: dict[str, float] = {}
    for sym in names:
        panel = panels.get(sym)
        if panel is None or "session" not in panel.columns:
            continue
        sess = pd.to_datetime(panel["session"])
        mask = sess < end
        if start is not None:
            mask = mask & (sess >= start)
        if not bool(mask.any()):
            continue
        dollar = dollar_adv_series(panel.loc[mask])
        med = float(dollar.replace([np.inf, -np.inf], np.nan).median())
        if np.isfinite(med):
            adv[sym] = med
    if not adv:
        return names
    values = np.asarray(list(adv.values()), dtype=np.float64)
    cut = 0.0
    if float(pctile) > 0:
        cut = float(np.nanpercentile(values, 100.0 * float(pctile)))
    cut = max(cut, float(min_usd))
    kept = [s for s in names if adv.get(s, 0.0) >= cut]
    return kept if kept else names


def apply_adv_usd_valid_mask(
    panel: pd.DataFrame,
    *,
    min_usd: float,
) -> np.ndarray:
    """Per-bar mask: dollar ADV proxy at ``t`` (known at close) meets the floor."""
    if float(min_usd) <= 0:
        return np.ones(len(panel), dtype=bool)
    dollar = dollar_adv_series(panel).to_numpy(dtype=np.float64)
    return np.isfinite(dollar) & (dollar >= float(min_usd))


def trailing_leverage_scale(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    lookback_days: int,
    train_ic: float,
    min_names: int = 8,
    min_obs: int = 20,
    dead_t: float = 1.0,
    floor: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-date leverage multiplier from causal trailing CS IC.

    Dates with trailing t-stat below ``dead_t`` (or non-positive IC) shrink
    toward ``floor``. Does **not** change within-date Pearson IC when applied
    as a date-level scale on weights/leverage. Labels on date ``t`` are never
    in the trailing window.
    """
    keys, mu, tt, _n = trailing_skip_ic_stats(
        pred,
        target,
        dates,
        lookback_days=int(lookback_days),
        min_names=min_names,
        min_obs=min_obs,
    )
    dates_i = np.asarray(dates, dtype=np.int64)
    factor = np.ones(dates_i.shape[0], dtype=np.float64)
    if keys.size == 0:
        return keys, factor
    loc = np.searchsorted(keys, dates_i)
    loc = np.clip(loc, 0, keys.size - 1)
    match = keys[loc] == dates_i
    mu_r = np.full(dates_i.shape[0], np.nan, dtype=np.float64)
    tt_r = np.full(dates_i.shape[0], np.nan, dtype=np.float64)
    mu_r[match] = mu[loc[match]]
    tt_r[match] = tt[loc[match]]
    train = float(train_ic) if np.isfinite(train_ic) and abs(float(train_ic)) > 1e-8 else 1.0
    have = np.isfinite(mu_r)
    scale = np.clip(mu_r / train, 0.0, 1.0)
    dead = have & ((~np.isfinite(tt_r)) | (tt_r < float(dead_t)) | (mu_r <= 0.0))
    scale = np.where(dead, float(floor), np.where(have, scale, 1.0))
    scale = np.clip(scale, float(floor), 1.0)
    factor[match] = scale[match]
    return keys, factor


def cap_name_risk(
    weights: np.ndarray,
    vol_level: np.ndarray | None = None,
    *,
    max_weight: float = 0.0,
    vol_k: float = 0.0,
    long_only: bool = False,
) -> np.ndarray:
    """Causal overnight gap-risk overlay on a single date's weights.

    ``max_weight`` caps ``|w_i|``. ``vol_k`` downweights names with positive
    ``vol_level`` (known at t): ``w /= 1 + vol_k * relu(vol_level)``.
    Caller must renormalize the row.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    squeeze = w.ndim == 1
    if squeeze:
        w = w.reshape(1, -1)
    if float(max_weight) > 0:
        cap = float(max_weight)
        w = np.clip(w, 0.0 if long_only else -cap, cap)
    if float(vol_k) > 0 and vol_level is not None:
        vol = np.asarray(vol_level, dtype=np.float64)
        if vol.ndim == 1:
            vol = vol.reshape(1, -1)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        scale = 1.0 + float(vol_k) * np.clip(vol, 0.0, None)
        w = w / np.where(scale > 1e-8, scale, 1.0)
    return w[0] if squeeze else w


def dead_ic_blend_weights(
    trailing_t: np.ndarray,
    *,
    dead_t: float = 1.0,
    dead_mix: float = 0.75,
) -> np.ndarray:
    """Weight on the conservative readout when trailing CS IC is dead.

    ``0`` = keep the primary skip; ``dead_mix`` when ``t-stat < dead_t``.
    """
    tt = np.asarray(trailing_t, dtype=np.float64)
    mix = np.zeros(tt.shape[0], dtype=np.float64)
    dead = (~np.isfinite(tt)) | (tt < float(dead_t))
    mix[dead] = float(dead_mix)
    return mix


def filter_trade_names_equity(names: Sequence[str]) -> list[str]:
    return [s for s in names if is_equity_name(str(s))]
