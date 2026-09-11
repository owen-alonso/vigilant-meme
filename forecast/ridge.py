"""Closed-form CS ridge and causal walk-forward refits.

Frozen ridge (one ``w`` fit on train) is the generate.py skip. Walk-forward
refits ``w`` on labels whose horizon is strictly before the score date, so
ancient pre-GFC structure can age out without peeking at today's residual.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from forecast.data import (
    CROSS_SECTION_FEATURES,
    CS_PRODUCT_FEATURES,
    FEATURE_NAMES,
    VOL_FEATURES,
    SymbolArrays,
)

CALENDAR_FEATURES = frozenset(
    {"traded", "staleness", "new_session", "tod_sin", "tod_cos", "tod_frac", "dow_frac"}
)
OHLC_FEATURES = frozenset({"range_hl", "body_co", "close_loc", "wick_up", "wick_dn"})
LONG_TS_FEATURES = frozenset({"ret_60", "ret_390", "vol_level"})
VOL_FEATURE_SET = frozenset(VOL_FEATURES)
CS_PRODUCT_SET = frozenset(CS_PRODUCT_FEATURES)
# Dot-com + GFC inside the 1999–2009 train window. Used only when drop_crashes=True.
CRASH_WINDOWS = (
    ("2000-03-01", "2002-10-31"),
    ("2007-07-01", "2009-03-31"),
)


def _ymd_to_days(ymd: str) -> int:
    return int((np.datetime64(ymd) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))


def dates_to_year(dates: np.ndarray) -> np.ndarray:
    """Calendar year for day-since-epoch keys."""
    cal = np.datetime64("1970-01-01") + np.asarray(dates, dtype=np.int64).astype("timedelta64[D]")
    return cal.astype("datetime64[Y]").astype(int) + 1970


def crash_date_set(windows: Sequence[tuple[str, str]] = CRASH_WINDOWS) -> set[int]:
    """Inclusive calendar-day keys for crash windows (days since epoch)."""
    out: set[int] = set()
    for lo, hi in windows:
        a, b = _ymd_to_days(lo), _ymd_to_days(hi)
        out.update(range(a, b + 1))
    return out


def feature_mask(mode: str) -> np.ndarray:
    """Boolean mask over ``FEATURE_NAMES``. ``all`` keeps every column."""
    names = list(FEATURE_NAMES)
    n = len(names)
    raw = (mode or "all").strip().lower()
    mask = np.ones(n, dtype=bool)
    if raw in ("", "all"):
        return mask
    if raw in ("cs", "cs_only"):
        keep = set(CROSS_SECTION_FEATURES)
        return np.array([nm in keep for nm in names], dtype=bool)
    if raw in ("no_calendar", "no_tod"):
        return np.array([nm not in CALENDAR_FEATURES for nm in names], dtype=bool)
    if raw in ("no_long_ts", "no_ts_long"):
        drop = LONG_TS_FEATURES | CALENDAR_FEATURES
        return np.array([nm not in drop for nm in names], dtype=bool)
    if raw in ("no_ohlc",):
        drop = OHLC_FEATURES | CALENDAR_FEATURES
        return np.array([nm not in drop for nm in names], dtype=bool)
    if raw in ("core", "core_cs"):
        drop = LONG_TS_FEATURES | OHLC_FEATURES | CALENDAR_FEATURES
        return np.array([nm not in drop for nm in names], dtype=bool)
    if raw in ("no_vol_products", "no_vol_cs_products"):
        # Promoted no_long_ts, plus the year-ablation unstable groups.
        drop = LONG_TS_FEATURES | CALENDAR_FEATURES | VOL_FEATURE_SET | CS_PRODUCT_SET
        return np.array([nm not in drop for nm in names], dtype=bool)
    raise ValueError(f"unknown feature mask {mode!r}")


def labelled_rows(
    symbols: Sequence[SymbolArrays],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized last-bar rows: ``x [N, F]``, ``y [N]``, ``dates [N]`` (days since epoch)."""
    mean = np.asarray(feature_mean, dtype=np.float64)
    std = np.asarray(feature_std, dtype=np.float64)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    rows_d: list[np.ndarray] = []
    for sym in symbols:
        if not bool(sym.valid.any()):
            continue
        raw = sym.features[sym.valid].astype(np.float64, copy=False)
        rows_x.append((raw - mean) / std)
        rows_y.append(sym.target[sym.valid].astype(np.float64, copy=False))
        if sym.dates is None:
            rows_d.append(np.full(int(sym.valid.sum()), -1, dtype=np.int64))
        else:
            rows_d.append(sym.dates[sym.valid].astype(np.int64, copy=False))
    if not rows_x:
        f = int(mean.shape[0])
        return (
            np.zeros((0, f), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(rows_x, axis=0),
        np.concatenate(rows_y, axis=0),
        np.concatenate(rows_d, axis=0),
    )


def _winsor_1d(values: np.ndarray, k: float) -> np.ndarray:
    if k <= 0 or values.size < 3:
        return values
    s = float(values.std())
    if s < 1e-8:
        return values
    mu = float(values.mean())
    return np.clip(values, mu - k * s, mu + k * s)


def _prepare_cs_design(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int,
    cs_demean: bool,
    cs_zscore: bool,
    rank_target: bool,
    feature_mask_bool: np.ndarray | None,
    date_halflife: float,
    exclude_dates: set[int] | None = None,
    y_winsor: float = 0.0,
    feat_winsor: float = 0.0,
    drop_disp_q: float = 0.0,
    year_balance: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Within-date design for ridge. Returns ``x, y, sample_weight, used_dates`` or None."""
    if not cs_demean and not cs_zscore and not rank_target:
        keep = np.ones(x.shape[0], dtype=bool)
        if exclude_dates:
            keep &= ~np.isin(dates.astype(np.int64), list(exclude_dates))
        x = x[keep].copy()
        y = y[keep].copy()
        dates_k = dates[keep]
        if feature_mask_bool is not None:
            mask = np.asarray(feature_mask_bool, dtype=bool)
            x[:, ~mask] = 0.0
        w = np.ones(x.shape[0], dtype=np.float64)
        if date_halflife > 0 and dates_k.size:
            d_max = int(dates_k.max())
            w = 0.5 ** (np.maximum(0, d_max - dates_k.astype(np.int64)) / float(date_halflife))
        return x, y, w, dates_k.astype(np.int64)
    mask = (
        np.ones(x.shape[1], dtype=bool)
        if feature_mask_bool is None
        else np.asarray(feature_mask_bool, dtype=bool)
    )
    x = np.asarray(x, dtype=np.float64).copy()
    x[:, ~mask] = 0.0
    y = np.asarray(y, dtype=np.float64).copy()
    weights: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    used_keys: list[int] = []
    d_max = int(dates.max()) if dates.size else 0
    hl = float(date_halflife)
    blocked = exclude_dates or set()
    disp: dict[int, float] = {}
    if drop_disp_q > 0:
        for key in np.unique(dates):
            sel = dates == key
            if int(sel.sum()) < int(min_names):
                continue
            disp[int(key)] = float(np.std(y[sel]))
        if disp:
            cutoff = float(np.quantile(np.asarray(list(disp.values())), 1.0 - drop_disp_q))
            blocked = set(blocked) | {k for k, v in disp.items() if v > cutoff}
    have = False
    for key in np.unique(dates):
        if int(key) in blocked:
            continue
        sel = dates == key
        n = int(sel.sum())
        if n < int(min_names):
            continue
        xd = x[sel].copy()
        yd = y[sel].copy()
        if y_winsor > 0 and not rank_target:
            yd = _winsor_1d(yd, y_winsor)
        if feat_winsor > 0:
            for j in range(xd.shape[1]):
                xd[:, j] = _winsor_1d(xd[:, j], feat_winsor)
        if rank_target:
            order = np.argsort(yd, kind="mergesort")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            if n > 1:
                yd = (ranks - ranks.mean()) / max(ranks.std(), 1e-8)
            else:
                yd = ranks - ranks.mean()
        if cs_demean or cs_zscore or rank_target:
            xd = xd - xd.mean(axis=0, keepdims=True)
            if not rank_target:
                yd = yd - yd.mean()
        if cs_zscore:
            std = xd.std(axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            xd = xd / std
        w = np.ones(n, dtype=np.float64)
        if hl > 0:
            w *= 0.5 ** (max(0, d_max - int(key)) / hl)
        xs.append(xd)
        ys.append(yd)
        weights.append(w)
        used_keys.append(int(key))
        have = True
    if not have:
        return None
    if year_balance and used_keys:
        years = dates_to_year(np.asarray(used_keys, dtype=np.int64))
        year_sum: dict[int, float] = {}
        for i, year in enumerate(years):
            year_sum[int(year)] = year_sum.get(int(year), 0.0) + float(weights[i].sum())
        for i, year in enumerate(years):
            denom = year_sum[int(year)]
            if denom > 0:
                weights[i] = weights[i] / denom
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(weights),
        np.asarray(used_keys, dtype=np.int64),
    )


def _solve_weighted_ridge(
    xd: np.ndarray,
    yd: np.ndarray,
    w: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float]:
    sw = np.sqrt(np.clip(w, 0.0, None))
    design = np.concatenate([xd, np.ones((xd.shape[0], 1), dtype=np.float64)], axis=1)
    dw = design * sw[:, None]
    yw = yd * sw
    lam = max(0.0, float(ridge))
    xtx = dw.T @ dw
    xtx.flat[:: xtx.shape[0] + 1] += lam
    try:
        coef = np.linalg.solve(xtx, dw.T @ yw)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx, dw.T @ yw, rcond=None)[0]
    return coef[:-1].astype(np.float64), float(coef[-1])


def _huber_irls(
    xd: np.ndarray,
    yd: np.ndarray,
    w: np.ndarray,
    ridge: float,
    delta: float,
    iters: int = 8,
) -> tuple[np.ndarray, float]:
    weights, bias = _solve_weighted_ridge(xd, yd, w, ridge)
    if delta <= 0:
        return weights, bias
    for _ in range(max(1, int(iters))):
        resid = yd - (xd @ weights + bias)
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) * 1.4826
        scale = max(mad, 1e-8)
        u = resid / (float(delta) * scale)
        irls = np.ones_like(u)
        big = np.abs(u) > 1.0
        irls[big] = 1.0 / np.abs(u[big])
        weights, bias = _solve_weighted_ridge(xd, yd, w * irls, ridge)
    return weights, bias


def fit_ridge_xy(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    ridge: float = 1.0,
    min_names: int = 8,
    cs_demean: bool = True,
    cs_zscore: bool = False,
    rank_target: bool = False,
    feature_mask_bool: np.ndarray | None = None,
    date_halflife: float = 0.0,
    exclude_dates: set[int] | None = None,
    y_winsor: float = 0.0,
    feat_winsor: float = 0.0,
    drop_disp_q: float = 0.0,
    huber_delta: float = 0.0,
    sign_constrain: bool = False,
    drop_crashes: bool = False,
    year_balance: bool = False,
) -> tuple[np.ndarray, float, float]:
    """CS ridge of ``y`` on ``x``. Returns weights, bias (0 if CS), in-sample mean CS IC."""
    n_features = int(x.shape[1]) if x.ndim == 2 else 0
    blocked = set(exclude_dates or set())
    if drop_crashes:
        blocked |= crash_date_set()
    prepared = _prepare_cs_design(
        x,
        y,
        dates,
        min_names=min_names,
        cs_demean=cs_demean,
        cs_zscore=cs_zscore,
        rank_target=rank_target,
        feature_mask_bool=feature_mask_bool,
        date_halflife=date_halflife,
        exclude_dates=blocked or None,
        y_winsor=y_winsor,
        feat_winsor=feat_winsor,
        drop_disp_q=drop_disp_q,
        year_balance=year_balance,
    )
    if prepared is None:
        return np.zeros(n_features, dtype=np.float32), 0.0, float("nan")
    xd, yd, w, used_dates = prepared
    if huber_delta > 0:
        raw_w, raw_b = _huber_irls(xd, yd, w, ridge, huber_delta)
    else:
        raw_w, raw_b = _solve_weighted_ridge(xd, yd, w, ridge)
    weights = raw_w.astype(np.float32)
    bias = 0.0 if (cs_demean or rank_target or cs_zscore) else float(raw_b)
    if sign_constrain and n_features:
        uni = univariate_cs_ics(x, y, dates, min_names=min_names)
        for i in range(n_features):
            if not np.isfinite(uni[i]) or abs(float(uni[i])) < 0.003:
                weights[i] = 0.0
            elif float(uni[i]) * float(weights[i]) < 0:
                weights[i] = 0.0
    pred = x @ weights.astype(np.float64) + bias
    kept = np.isin(dates.astype(np.int64), used_dates)
    if not bool(kept.any()):
        kept = np.ones(dates.shape[0], dtype=bool)
    ic = mean_cs_ic(pred[kept], y[kept], dates[kept], min_names=min_names)
    # Scale |w| so CS pred std matches CS y std on the fit sample (same as the old skip).
    if np.isfinite(ic) and int(kept.sum()) >= 2:
        p_std = float(pred[kept].std())
        y_std = float(y[kept].std())
        if p_std > 1e-8 and y_std > 1e-8:
            amp = abs(ic) * y_std / p_std
            weights = (weights * amp).astype(np.float32)
            bias = float(bias * amp)
    return weights, bias, ic


def mean_cs_ic(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> float:
    from forecast.training import mean_cs_stats

    return float(mean_cs_stats(pred, target, dates, min_names=min_names)["cs_ic"])


def cs_stats(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
    flat_as_zero: bool = False,
) -> dict[str, float]:
    from forecast.training import mean_cs_stats

    return mean_cs_stats(
        pred, target, dates, min_names=min_names, flat_as_zero=flat_as_zero
    )


def date_ics(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> np.ndarray:
    """Per-date CS Pearson; rows with too few names are omitted."""
    from forecast.training import _pearson

    rows: list[tuple[int, float]] = []
    for key in np.unique(dates):
        sel = dates == key
        if int(sel.sum()) < int(min_names):
            continue
        rows.append((int(key), _pearson(pred[sel], target[sel])))
    if not rows:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def trailing_skip_ic_stats(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    lookback_days: int,
    min_names: int = 8,
    min_obs: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal trailing mean CS IC of a *frozen* skip.

    For each unique date ``t`` in ``dates``, the mean and t-stat use per-date
    CS ICs on ``d`` in ``[t - lookback_days, t)``. The label on ``d`` realizes
    at ``d+1``, so ``d < t`` is known at close ``t``. Date ``t``'s own ``y``
    is never in the window.

    Returns ``keys, mean_ic, tstat, n_obs`` aligned to sorted unique dates.
    Insufficient history is NaN (caller should leave those preds unscaled).
    """
    all_keys = np.unique(np.asarray(dates, dtype=np.int64))
    n_all = int(all_keys.size)
    mean_out = np.full(n_all, np.nan, dtype=np.float64)
    t_out = np.full(n_all, np.nan, dtype=np.float64)
    n_out = np.zeros(n_all, dtype=np.int64)
    ics = date_ics(pred, target, dates, min_names=min_names)
    if ics.size == 0 or n_all == 0:
        return all_keys, mean_out, t_out, n_out
    ic_keys = ics[:, 0].astype(np.int64)
    ic_vals = ics[:, 1].astype(np.float64)
    order = np.argsort(ic_keys, kind="mergesort")
    ic_keys = ic_keys[order]
    ic_vals = ic_vals[order]
    finite = np.isfinite(ic_vals)
    ic_keys = ic_keys[finite]
    ic_vals = ic_vals[finite]
    if ic_keys.size == 0:
        return all_keys, mean_out, t_out, n_out
    csum = np.cumsum(ic_vals)
    csum2 = np.cumsum(ic_vals * ic_vals)
    span = max(1, int(lookback_days))
    need = max(1, int(min_obs))
    for i, t in enumerate(all_keys):
        lo_val = int(t) - span
        lo = int(np.searchsorted(ic_keys, lo_val, side="left"))
        hi = int(np.searchsorted(ic_keys, int(t), side="left"))
        n = hi - lo
        n_out[i] = n
        if n < need:
            continue
        s = float(csum[hi - 1] - (csum[lo - 1] if lo > 0 else 0.0))
        s2 = float(csum2[hi - 1] - (csum2[lo - 1] if lo > 0 else 0.0))
        mu = s / n
        var = (s2 - s * s / n) / max(n - 1, 1)
        se = float(np.sqrt(max(var, 0.0)) / np.sqrt(n))
        mean_out[i] = mu
        t_out[i] = mu / se if se > 1e-12 else float("nan")
    return all_keys, mean_out, t_out, n_out


def apply_skip_ic_shrink(
    pred: np.ndarray,
    dates: np.ndarray,
    *,
    date_keys: np.ndarray,
    trailing_ic: np.ndarray,
    trailing_t: np.ndarray,
    train_ic: float,
    mode: str = "scale",
) -> np.ndarray:
    """Scale or flatten frozen-skip scores from causal trailing CS IC.

    ``scale`` / ``scale_cap``: ``max(0, trailing_ic / train_ic)`` (cap clips at 1).
    Positive scale is a no-op for Pearson CS IC; flattening (factor=0) is the
    timing overlay. ``flatten_tstat`` zeros dates with trailing t-stat < 0;
    ``flatten_weak`` zeros trailing t-stat < 1.
    Dates without enough trailing history keep the raw skip.
    """
    dates_i = np.asarray(dates, dtype=np.int64)
    keys = np.asarray(date_keys, dtype=np.int64)
    out = np.asarray(pred, dtype=np.float64).copy()
    if dates_i.size == 0 or keys.size == 0:
        return out
    loc = np.searchsorted(keys, dates_i)
    loc = np.clip(loc, 0, keys.size - 1)
    match = keys[loc] == dates_i
    mu = np.full(dates_i.shape[0], np.nan, dtype=np.float64)
    tt = np.full(dates_i.shape[0], np.nan, dtype=np.float64)
    mu[match] = np.asarray(trailing_ic, dtype=np.float64)[loc[match]]
    tt[match] = np.asarray(trailing_t, dtype=np.float64)[loc[match]]
    have = np.isfinite(mu)
    train = float(train_ic)
    if (not np.isfinite(train)) or abs(train) < 1e-8:
        train = 1.0
    raw = (mode or "scale").strip().lower()
    factor = np.ones(out.shape[0], dtype=np.float64)
    if raw in ("scale", "shrink"):
        factor = np.where(have, np.maximum(0.0, mu / train), 1.0)
    elif raw in ("scale_cap", "cap"):
        factor = np.where(have, np.clip(mu / train, 0.0, 1.0), 1.0)
    elif raw in ("flatten_tstat", "flatten"):
        dead = have & np.isfinite(tt) & (tt < 0.0)
        dead |= have & (~np.isfinite(tt)) & (mu <= 0.0)
        factor = np.where(dead, 0.0, 1.0)
    elif raw in ("flatten_weak", "weak"):
        dead = have & ((~np.isfinite(tt)) | (tt < 1.0))
        factor = np.where(dead, 0.0, 1.0)
    elif raw in ("flatten_ic",):
        factor = np.where(have & (mu <= 0.0), 0.0, 1.0)
    else:
        raise ValueError(f"unknown skip-IC shrink mode {mode!r}")
    out *= factor
    return out


def univariate_cs_ics(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> np.ndarray:
    """Mean CS IC of each feature column vs ``y`` (sign = raw association)."""
    f = int(x.shape[1]) if x.ndim == 2 else 0
    out = np.full(f, np.nan, dtype=np.float64)
    for j in range(f):
        out[j] = mean_cs_ic(x[:, j], y, dates, min_names=min_names)
    return out


def year_cs_ics(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
    flat_as_zero: bool = False,
) -> list[dict[str, float]]:
    """Mean CS IC / t-stat grouped by calendar year of ``dates`` (days since epoch)."""
    if pred.size == 0:
        return []
    yr = dates_to_year(dates)
    rows: list[dict[str, float]] = []
    for year in sorted(set(int(v) for v in yr)):
        sel = yr == year
        stats = cs_stats(
            pred[sel],
            target[sel],
            dates[sel],
            min_names=min_names,
            flat_as_zero=flat_as_zero,
        )
        stats["year"] = float(year)
        rows.append(stats)
    return rows


def year_feature_ics(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> dict[int, np.ndarray]:
    """Univariate mean CS IC of each column, keyed by calendar year."""
    out: dict[int, np.ndarray] = {}
    yr = dates_to_year(dates)
    for year in sorted(set(int(v) for v in yr)):
        sel = yr == year
        out[int(year)] = univariate_cs_ics(x[sel], y[sel], dates[sel], min_names=min_names)
    return out


def year_stable_mask(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
    min_frac: float = 0.7,
    min_abs: float = 0.003,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    d_val: np.ndarray | None = None,
) -> np.ndarray:
    """Keep columns whose univariate CS IC sign is stable across train years.

    Optional val arrays require the same overall val sign (val-gated, not test).
    """
    f = int(x.shape[1]) if x.ndim == 2 else 0
    by_year = year_feature_ics(x, y, dates, min_names=min_names)
    if not by_year:
        return np.ones(f, dtype=bool)
    stacked = np.stack(list(by_year.values()), axis=0)
    keep = np.zeros(f, dtype=bool)
    for j in range(f):
        col = stacked[:, j]
        finite = col[np.isfinite(col) & (np.abs(col) >= float(min_abs))]
        if finite.size < 2:
            continue
        pos = float((finite > 0).mean())
        if pos >= float(min_frac) or (1.0 - pos) >= float(min_frac):
            keep[j] = True
    if x_val is not None and y_val is not None and d_val is not None:
        val = univariate_cs_ics(x_val, y_val, d_val, min_names=min_names)
        train = univariate_cs_ics(x, y, dates, min_names=min_names)
        agree = np.isfinite(val) & np.isfinite(train) & (val * train > 0)
        keep &= agree
    if not bool(keep.any()):
        return np.ones(f, dtype=bool)
    return keep


def stable_feature_mask(
    x_train: np.ndarray,
    y_train: np.ndarray,
    d_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    d_val: np.ndarray,
    *,
    min_names: int = 3,
    min_abs_val: float = 0.0,
) -> np.ndarray:
    """Keep columns whose univariate CS IC has the same sign on train and val."""
    tr = univariate_cs_ics(x_train, y_train, d_train, min_names=min_names)
    va = univariate_cs_ics(x_val, y_val, d_val, min_names=min_names)
    keep = np.isfinite(tr) & np.isfinite(va) & (tr * va > 0)
    if min_abs_val > 0:
        keep &= np.abs(va) >= float(min_abs_val)
    if not bool(keep.any()):
        return np.ones(tr.size, dtype=bool)
    return keep


def _iter_date_designs(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int,
    cs_demean: bool,
    cs_zscore: bool,
    rank_target: bool,
    feature_mask_bool: np.ndarray | None,
    exclude_dates: set[int] | None = None,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    blocked = exclude_dates or set()
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for key in np.unique(dates):
        if int(key) in blocked:
            continue
        sel = dates == key
        des = _one_date_design(
            x[sel],
            y[sel],
            min_names=min_names,
            cs_demean=cs_demean,
            cs_zscore=cs_zscore,
            rank_target=rank_target,
            feature_mask_bool=feature_mask_bool,
        )
        if des is None:
            continue
        out.append((int(key), des[0], des[1]))
    return out


def fit_listnet_xy(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    ridge: float = 10.0,
    min_names: int = 8,
    cs_demean: bool = True,
    cs_zscore: bool = False,
    rank_target: bool = True,
    feature_mask_bool: np.ndarray | None = None,
    exclude_dates: set[int] | None = None,
    steps: int = 250,
    lr: float = 0.08,
    drop_crashes: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Linear ListNet on within-date designs. Same skip interface as ridge."""
    n_features = int(x.shape[1]) if x.ndim == 2 else 0
    blocked = set(exclude_dates or set())
    if drop_crashes:
        blocked |= crash_date_set()
    groups = _iter_date_designs(
        x,
        y,
        dates,
        min_names=min_names,
        cs_demean=cs_demean,
        cs_zscore=cs_zscore,
        rank_target=rank_target,
        feature_mask_bool=feature_mask_bool,
        exclude_dates=blocked or None,
    )
    if not groups:
        return np.zeros(n_features, dtype=np.float32), 0.0, float("nan")
    w = np.zeros(n_features, dtype=np.float64)
    lam = max(0.0, float(ridge))
    n_g = max(1, len(groups))
    for _ in range(max(1, int(steps))):
        grad = lam * w
        for _, xd, yd in groups:
            logits = xd @ w
            scale = max(float(np.std(logits)), 1.0)
            y_scale = max(float(np.std(yd)), 1.0)
            z = logits / scale
            z = z - z.max()
            ez = np.exp(z)
            p = ez / max(float(ez.sum()), 1e-12)
            qz = yd / y_scale
            qz = qz - qz.max()
            eq = np.exp(qz)
            q = eq / max(float(eq.sum()), 1e-12)
            grad = grad + (xd.T @ (p - q)) / scale
        w -= float(lr) * grad / n_g
    weights = w.astype(np.float32)
    pred = x @ weights.astype(np.float64)
    ic = mean_cs_ic(pred, y, dates, min_names=min_names)
    if np.isfinite(ic) and pred.size >= 2:
        p_std = float(pred.std())
        y_std = float(y.std())
        if p_std > 1e-8 and y_std > 1e-8:
            weights = (weights * (abs(ic) * y_std / p_std)).astype(np.float32)
    return weights, 0.0, ic


def fit_ranknet_xy(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    ridge: float = 10.0,
    min_names: int = 8,
    cs_demean: bool = True,
    cs_zscore: bool = False,
    rank_target: bool = True,
    feature_mask_bool: np.ndarray | None = None,
    exclude_dates: set[int] | None = None,
    steps: int = 120,
    lr: float = 0.05,
    drop_crashes: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Linear pairwise RankNet on within-date designs."""
    n_features = int(x.shape[1]) if x.ndim == 2 else 0
    blocked = set(exclude_dates or set())
    if drop_crashes:
        blocked |= crash_date_set()
    groups = _iter_date_designs(
        x,
        y,
        dates,
        min_names=min_names,
        cs_demean=cs_demean,
        cs_zscore=cs_zscore,
        rank_target=rank_target,
        feature_mask_bool=feature_mask_bool,
        exclude_dates=blocked or None,
    )
    if not groups:
        return np.zeros(n_features, dtype=np.float32), 0.0, float("nan")
    w = np.zeros(n_features, dtype=np.float64)
    lam = max(0.0, float(ridge))
    n_g = max(1, len(groups))
    for _ in range(max(1, int(steps))):
        grad = lam * w
        for _, xd, yd in groups:
            s = xd @ w
            # P(i beats j) = sigmoid(s_i - s_j); target 1 iff y_i > y_j.
            diff = s[:, None] - s[None, :]
            tdiff = yd[:, None] - yd[None, :]
            pair = tdiff != 0
            if not bool(pair.any()):
                continue
            # logistic gradient vs score: (sigmoid(diff) - T)
            sig = 1.0 / (1.0 + np.exp(-np.clip(diff, -30.0, 30.0)))
            target = (tdiff > 0).astype(np.float64)
            gs = ((sig - target) * pair).sum(axis=1)
            grad = grad + xd.T @ gs / max(float(pair.sum()), 1.0)
        w -= float(lr) * grad / n_g
    weights = w.astype(np.float32)
    pred = x @ weights.astype(np.float64)
    ic = mean_cs_ic(pred, y, dates, min_names=min_names)
    if np.isfinite(ic) and pred.size >= 2:
        p_std = float(pred.std())
        y_std = float(y.std())
        if p_std > 1e-8 and y_std > 1e-8:
            weights = (weights * (abs(ic) * y_std / p_std)).astype(np.float32)
    return weights, 0.0, ic


def fit_regime_ridge(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    regime: np.ndarray,
    *,
    ridge: float = 10.0,
    min_names: int = 8,
    rank_target: bool = True,
    feature_mask_bool: np.ndarray | None = None,
    exclude_dates: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Two frozen ridges split by a date-level feature known at t.

    ``regime`` is a per-row score (e.g. |mkt_ret_1|). Dates at/above the train
    median use ``w_high``. Returns ``w_low, w_high, split, train_ic``.
    """
    date_score: dict[int, float] = {}
    for key in np.unique(dates):
        sel = dates == key
        date_score[int(key)] = float(np.nanmean(np.abs(regime[sel])))
    vals = np.asarray(list(date_score.values()), dtype=np.float64)
    split = float(np.median(vals)) if vals.size else 0.0
    low_dates = {k for k, v in date_score.items() if v < split}
    high_dates = {k for k, v in date_score.items() if v >= split}
    blocked = set(exclude_dates or set())
    w_low, _, _ = fit_ridge_xy(
        x,
        y,
        dates,
        ridge=ridge,
        min_names=min_names,
        rank_target=rank_target,
        feature_mask_bool=feature_mask_bool,
        exclude_dates=blocked | high_dates,
    )
    w_high, _, _ = fit_ridge_xy(
        x,
        y,
        dates,
        ridge=ridge,
        min_names=min_names,
        rank_target=rank_target,
        feature_mask_bool=feature_mask_bool,
        exclude_dates=blocked | low_dates,
    )
    pred = np.zeros(x.shape[0], dtype=np.float64)
    for key, score in date_score.items():
        sel = dates == key
        w = w_high if score >= split else w_low
        pred[sel] = x[sel] @ w.astype(np.float64)
    ic = mean_cs_ic(pred, y, dates, min_names=min_names)
    return w_low, w_high, split, ic


def predict_regime(
    x: np.ndarray,
    dates: np.ndarray,
    regime: np.ndarray,
    w_low: np.ndarray,
    w_high: np.ndarray,
    split: float,
) -> np.ndarray:
    pred = np.zeros(x.shape[0], dtype=np.float64)
    for key in np.unique(dates):
        sel = dates == key
        score = float(np.nanmean(np.abs(regime[sel])))
        w = w_high if score >= split else w_low
        pred[sel] = x[sel] @ np.asarray(w, dtype=np.float64)
    return pred


def date_level_mean(
    x: np.ndarray,
    dates: np.ndarray,
    col: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique dates and the within-date mean of one column."""
    keys = np.unique(np.asarray(dates, dtype=np.int64))
    vals = np.empty(keys.size, dtype=np.float64)
    col_i = int(col)
    for i, key in enumerate(keys):
        sel = dates == key
        vals[i] = float(np.nanmean(x[sel, col_i]))
    return keys.astype(np.int64), vals


def date_level_std(
    x: np.ndarray,
    dates: np.ndarray,
    col: int,
) -> dict[int, float]:
    """Within-date std of one feature column (known at close; not a label)."""
    out: dict[int, float] = {}
    col_i = int(col)
    for key in np.unique(np.asarray(dates, dtype=np.int64)):
        sel = dates == key
        sl = x[sel, col_i]
        if sl.size < 2:
            continue
        out[int(key)] = float(np.nanstd(sl))
    return out


def trailing_realized_vol(
    date_keys: np.ndarray,
    date_ret: np.ndarray,
    *,
    window: int = 60,
    min_obs: int = 20,
) -> dict[int, float]:
    """Causal trailing std of a date-level return series, including today.

    ``date_keys`` must be sorted. Score at ``t`` uses returns through ``t`` only.
    """
    keys = np.asarray(date_keys, dtype=np.int64)
    rets = np.asarray(date_ret, dtype=np.float64)
    n = int(keys.size)
    out: dict[int, float] = {}
    if n == 0:
        return out
    csum = np.cumsum(rets)
    csum2 = np.cumsum(rets * rets)
    win = max(2, int(window))
    need = max(2, int(min_obs))
    for i in range(n):
        lo = max(0, i + 1 - win)
        cnt = i - lo + 1
        if cnt < need:
            continue
        s = float(csum[i] - (csum[lo - 1] if lo > 0 else 0.0))
        s2 = float(csum2[i] - (csum2[lo - 1] if lo > 0 else 0.0))
        var = (s2 - s * s / cnt) / max(cnt - 1, 1)
        out[int(keys[i])] = float(np.sqrt(max(var, 0.0)))
    return out


def regime_score_map(
    x: np.ndarray,
    dates: np.ndarray,
    *,
    kind: str,
    names: Sequence[str] | None = None,
    window: int = 60,
    min_obs: int = 20,
) -> dict[int, float]:
    """Date-level decision-time regime scores. Never uses ``y``.

    ``spy_vol``: trailing SPY/market 60d realized vol from ``mkt_ret_1``.
    ``cs_disp``: within-date std of ``ret_1``.
    """
    feat = list(names) if names is not None else list(FEATURE_NAMES)
    raw = (kind or "spy_vol").strip().lower()
    if raw in ("spy_vol", "mkt_vol", "spy60"):
        if "mkt_ret_1" not in feat:
            raise ValueError("mkt_ret_1 required for spy_vol regime")
        keys, rets = date_level_mean(x, dates, feat.index("mkt_ret_1"))
        return trailing_realized_vol(keys, rets, window=window, min_obs=min_obs)
    if raw in ("cs_disp", "cs_dispersion", "ret1_disp"):
        col = feat.index("ret_1") if "ret_1" in feat else 0
        return date_level_std(x, dates, col)
    raise ValueError(f"unknown regime kind {kind!r}")


def bucket_edges_from_train(
    train_scores: dict[int, float],
    n_buckets: int,
) -> np.ndarray:
    """Quantile edges from train dates only. Length ``n_buckets + 1``."""
    k = max(2, int(n_buckets))
    vals = np.asarray(list(train_scores.values()), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, k + 1)
    edges = np.quantile(vals, qs).astype(np.float64)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_score_buckets(
    scores: dict[int, float],
    edges: np.ndarray,
) -> dict[int, int]:
    """Map each date to a bucket in ``[0, n_buckets)``. Missing scores -> middle."""
    k = max(1, int(np.asarray(edges).size) - 1)
    mid = k // 2
    out: dict[int, int] = {}
    ed = np.asarray(edges, dtype=np.float64)
    for key, val in scores.items():
        if not np.isfinite(val):
            out[int(key)] = mid
            continue
        b = int(np.searchsorted(ed, val, side="right") - 1)
        out[int(key)] = int(np.clip(b, 0, k - 1))
    return out


def dates_to_buckets(
    dates: np.ndarray,
    score_map: dict[int, float],
    edges: np.ndarray,
) -> np.ndarray:
    """Per-row bucket ids aligned with ``dates``."""
    assigned = assign_score_buckets(score_map, edges)
    k = max(1, int(np.asarray(edges).size) - 1)
    mid = k // 2
    out = np.empty(np.asarray(dates).shape[0], dtype=np.int64)
    for i, key in enumerate(np.asarray(dates, dtype=np.int64)):
        out[i] = assigned.get(int(key), mid)
    return out


def fit_regime_heads(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    buckets: dict[int, int],
    *,
    n_buckets: int,
    ridge: float = 10.0,
    min_names: int = 8,
    rank_target: bool = True,
    feat_winsor: float = 3.0,
    feature_mask_bool: np.ndarray | None = None,
    min_dates_per_bucket: int = 60,
) -> tuple[np.ndarray, float, list[int]]:
    """Fit one last-bar CS ridge per train-declared bucket.

    Returns ``W [K, F]``, train hard-assignment CS IC, and counts per bucket.
    """
    k = max(2, int(n_buckets))
    f = int(x.shape[1]) if x.ndim == 2 else 0
    weights = np.zeros((k, f), dtype=np.float32)
    counts: list[int] = []
    date_keys = np.unique(np.asarray(dates, dtype=np.int64))
    for b in range(k):
        in_b = {int(d) for d in date_keys if int(buckets.get(int(d), -1)) == b}
        counts.append(len(in_b))
        if len(in_b) < int(min_dates_per_bucket):
            continue
        other = {int(d) for d in date_keys if int(d) not in in_b}
        w, _, _ = fit_ridge_xy(
            x,
            y,
            dates,
            ridge=ridge,
            min_names=min_names,
            rank_target=rank_target,
            feat_winsor=feat_winsor,
            feature_mask_bool=feature_mask_bool,
            exclude_dates=other,
        )
        weights[b] = w.astype(np.float32)
    pred = predict_regime_heads(x, dates, buckets, weights)
    ic = mean_cs_ic(pred, y, dates, min_names=min_names)
    return weights, ic, counts


def predict_regime_heads(
    x: np.ndarray,
    dates: np.ndarray,
    buckets: dict[int, int],
    weights: np.ndarray,
) -> np.ndarray:
    """Hard-assign each date to its train-edge bucket head."""
    w = np.asarray(weights, dtype=np.float64)
    k = int(w.shape[0])
    mid = max(k // 2, 0)
    pred = np.zeros(x.shape[0], dtype=np.float64)
    for key in np.unique(np.asarray(dates, dtype=np.int64)):
        sel = dates == key
        b = int(buckets.get(int(key), mid))
        b = int(np.clip(b, 0, k - 1))
        pred[sel] = x[sel] @ w[b]
    return pred


def predict_regime_mixture(
    x: np.ndarray,
    dates: np.ndarray,
    score_map: dict[int, float],
    weights: np.ndarray,
    edges: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Soft mix of bucket heads using distance of the date score to centers."""
    w = np.asarray(weights, dtype=np.float64)
    ed = np.asarray(edges, dtype=np.float64)
    finite = ed[np.isfinite(ed)]
    if finite.size == 0:
        return predict_regime_heads(x, dates, assign_score_buckets(score_map, edges), w)
    # Restore finite span for centers; inf edges are the outer buckets.
    lo = float(finite.min())
    hi = float(finite.max())
    ed_c = ed.copy()
    if not np.isfinite(ed_c[0]):
        ed_c[0] = lo
    if not np.isfinite(ed_c[-1]):
        ed_c[-1] = hi
    centers = 0.5 * (ed_c[:-1] + ed_c[1:])
    temp = max(float(temperature), 1e-6)
    k = int(w.shape[0])
    pred = np.zeros(x.shape[0], dtype=np.float64)
    default = float(np.mean(centers)) if centers.size else 0.0
    for key in np.unique(np.asarray(dates, dtype=np.int64)):
        sel = dates == key
        s = float(score_map.get(int(key), default))
        if not np.isfinite(s):
            s = default
        logits = -((s - centers) / temp) ** 2
        logits = logits - float(np.max(logits))
        gate = np.exp(logits)
        gate = gate / max(float(gate.sum()), 1e-12)
        mixed = gate[:k] @ w[:k]
        pred[sel] = x[sel] @ mixed
    return pred


def trailing_window_keep(
    dates: np.ndarray,
    *,
    end_days: int,
    years: float,
) -> np.ndarray:
    """Train rows in ``[end - years, end)``. ``end_days`` is typically val start."""
    span = int(round(float(years) * 365.25))
    lo = int(end_days) - span
    d = np.asarray(dates, dtype=np.int64)
    return (d >= lo) & (d < int(end_days))


def late_train_holdout_mask(
    dates: np.ndarray,
    *,
    holdout_years: int = 2,
    min_holdout_dates: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Split train dates into fit vs leak-free late-train selection slice.

    Returns boolean masks over rows. Falls back to last 20% of unique dates
    when the last ``holdout_years`` calendar years are too thin.
    """
    d = np.asarray(dates, dtype=np.int64)
    uniq = np.unique(d)
    if uniq.size == 0:
        empty = np.zeros(d.shape[0], dtype=bool)
        return empty, empty
    years = dates_to_year(uniq)
    max_year = int(years.max())
    cut_year = max_year - max(1, int(holdout_years)) + 1
    hold_keys = set(int(k) for k, y in zip(uniq, years) if int(y) >= cut_year)
    if len(hold_keys) < int(min_holdout_dates):
        n_hold = max(int(min_holdout_dates), int(round(0.2 * uniq.size)))
        n_hold = min(n_hold, max(1, uniq.size - 1))
        hold_keys = set(int(k) for k in uniq[-n_hold:])
    sel = np.array([int(v) in hold_keys for v in d], dtype=bool)
    fit = ~sel
    if not bool(fit.any()) or not bool(sel.any()):
        n_hold = max(1, int(round(0.2 * uniq.size)))
        hold_keys = set(int(k) for k in uniq[-n_hold:])
        sel = np.array([int(v) in hold_keys for v in d], dtype=bool)
        fit = ~sel
    return fit, sel


def augment_cs_products(x: np.ndarray, cols: Sequence[int]) -> np.ndarray:
    """Append pairwise products of selected columns (CS interactions)."""
    idx = [int(i) for i in cols]
    extras: list[np.ndarray] = []
    for a in range(len(idx)):
        for b in range(a, len(idx)):
            extras.append(x[:, idx[a]] * x[:, idx[b]])
    if not extras:
        return x
    return np.concatenate([x, np.stack(extras, axis=1)], axis=1)


def fit_residual_mlp(
    x: np.ndarray,
    residual: np.ndarray,
    dates: np.ndarray,
    *,
    hidden: int = 8,
    ridge: float = 25.0,
    steps: int = 200,
    lr: float = 0.03,
    min_names: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiny ReLU MLP on CS-demeaned ``x`` to fit ``residual``. Returns W1, b1, W2."""
    rng = np.random.default_rng(int(seed))
    f = int(x.shape[1])
    h = max(2, int(hidden))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for key in np.unique(dates):
        sel = dates == key
        if int(sel.sum()) < int(min_names):
            continue
        xd = x[sel] - x[sel].mean(axis=0, keepdims=True)
        yd = residual[sel] - residual[sel].mean()
        xs.append(xd)
        ys.append(yd)
    if not xs:
        return (
            np.zeros((f, h), dtype=np.float64),
            np.zeros(h, dtype=np.float64),
            np.zeros(h, dtype=np.float64),
        )
    xd = np.concatenate(xs)
    yd = np.concatenate(ys)
    w1 = rng.normal(scale=1.0 / max(np.sqrt(f), 1.0), size=(f, h))
    b1 = np.zeros(h, dtype=np.float64)
    w2 = rng.normal(scale=1.0 / max(np.sqrt(h), 1.0), size=(h,))
    lam = max(0.0, float(ridge))
    n = max(1, xd.shape[0])
    for _ in range(max(1, int(steps))):
        hpre = xd @ w1 + b1
        hh = np.maximum(hpre, 0.0)
        pred = hh @ w2
        err = pred - yd
        g_w2 = (hh.T @ err) / n + lam * w2
        g_h = np.outer(err, w2) / n
        g_hpre = g_h * (hpre > 0)
        g_w1 = xd.T @ g_hpre + lam * w1
        g_b1 = g_hpre.sum(axis=0)
        w2 -= float(lr) * g_w2
        w1 -= float(lr) * g_w1
        b1 -= float(lr) * g_b1
    return w1, b1, w2


def predict_residual_mlp(
    x: np.ndarray,
    dates: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    *,
    min_names: int = 8,
) -> np.ndarray:
    pred = np.zeros(x.shape[0], dtype=np.float64)
    for key in np.unique(dates):
        sel = dates == key
        if int(sel.sum()) < int(min_names):
            continue
        xd = x[sel] - x[sel].mean(axis=0, keepdims=True)
        hh = np.maximum(xd @ w1 + b1, 0.0)
        pred[sel] = hh @ w2
    return pred


def _one_date_design(
    xd: np.ndarray,
    yd: np.ndarray,
    *,
    min_names: int,
    cs_demean: bool,
    cs_zscore: bool,
    rank_target: bool,
    feature_mask_bool: np.ndarray | None,
    feat_winsor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(xd.shape[0])
    if n < int(min_names):
        return None
    x = np.asarray(xd, dtype=np.float64).copy()
    y = np.asarray(yd, dtype=np.float64).copy()
    if feature_mask_bool is not None:
        mask = np.asarray(feature_mask_bool, dtype=bool)
        x[:, ~mask] = 0.0
    if feat_winsor > 0:
        for j in range(x.shape[1]):
            x[:, j] = _winsor_1d(x[:, j], feat_winsor)
    if rank_target:
        order = np.argsort(y, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        y = ranks - ranks.mean()
        if n > 1:
            y = y / max(float(ranks.std()), 1e-8)
    if cs_demean or cs_zscore:
        x = x - x.mean(axis=0, keepdims=True)
        if not rank_target:
            y = y - y.mean()
    if cs_zscore:
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        x = x / std
    return x, y


def walk_forward_predict(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    score_dates: np.ndarray,
    lookback_days: int | None,
    ridge: float = 1.0,
    min_names: int = 8,
    cs_demean: bool = True,
    cs_zscore: bool = False,
    rank_target: bool = False,
    feature_mask_bool: np.ndarray | None = None,
    date_halflife: float = 0.0,
    min_train_dates: int = 60,
    feat_winsor: float = 0.0,
) -> np.ndarray:
    """Causal scores for ``score_dates``: fit on ``dates < d`` (and ``>= d - lookback``).

    Same-day ``y`` is never in the design. ``lookback_days=None`` is expanding.
    Recency ``date_halflife`` is ignored here (window membership is the clock).
    """
    del date_halflife  # walk-forward uses a hard lookback, not exp weights
    pred = np.full(x.shape[0], np.nan, dtype=np.float64)
    f = int(x.shape[1])
    uniq_score = set(int(k) for k in np.unique(np.asarray(score_dates, dtype=np.int64)))
    uniq_all = [int(k) for k in np.unique(dates)]
    by_date_x: dict[int, np.ndarray] = {}
    by_date_y: dict[int, np.ndarray] = {}
    by_date_idx: dict[int, np.ndarray] = {}
    contrib: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key in uniq_all:
        sel = np.flatnonzero(dates == key)
        by_date_x[key] = x[sel]
        by_date_y[key] = y[sel]
        by_date_idx[key] = sel
        des = _one_date_design(
            x[sel],
            y[sel],
            min_names=min_names,
            cs_demean=cs_demean,
            cs_zscore=cs_zscore,
            rank_target=rank_target,
            feature_mask_bool=feature_mask_bool,
            feat_winsor=feat_winsor,
        )
        if des is None:
            continue
        xd, yd = des
        contrib[key] = (xd.T @ xd, xd.T @ yd)

    acc_xtx = np.zeros((f, f), dtype=np.float64)
    acc_xty = np.zeros((f,), dtype=np.float64)
    lam = max(0.0, float(ridge))
    eye = np.eye(f, dtype=np.float64)
    hist_keys: list[int] = []
    cursor = 0
    n_fit_dates = 0

    def _add(key: int) -> None:
        nonlocal n_fit_dates, acc_xtx, acc_xty
        block = contrib.get(key)
        if block is None:
            hist_keys.append(key)
            return
        acc_xtx = acc_xtx + block[0]
        acc_xty = acc_xty + block[1]
        hist_keys.append(key)
        n_fit_dates += 1

    def _pop() -> None:
        nonlocal n_fit_dates, acc_xtx, acc_xty
        key = hist_keys.pop(0)
        block = contrib.get(key)
        if block is None:
            return
        acc_xtx = acc_xtx - block[0]
        acc_xty = acc_xty - block[1]
        n_fit_dates -= 1

    for d in uniq_all:
        while cursor < len(uniq_all) and uniq_all[cursor] < d:
            _add(uniq_all[cursor])
            cursor += 1
        if lookback_days is not None and lookback_days > 0:
            lo = d - int(lookback_days)
            while hist_keys and hist_keys[0] < lo:
                _pop()
        if d not in uniq_score:
            continue
        if n_fit_dates < int(min_train_dates):
            continue
        try:
            weights = np.linalg.solve(acc_xtx + lam * eye, acc_xty)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(acc_xtx + lam * eye, acc_xty, rcond=None)[0]
        idx = by_date_idx[d]
        xd = by_date_x[d].copy()
        if feature_mask_bool is not None:
            mask = np.asarray(feature_mask_bool, dtype=bool)
            xd[:, ~mask] = 0.0
        pred[idx] = xd @ weights
    return pred


def fit_skip_xy(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_cfg: Any,
    *,
    min_names: int = 8,
) -> tuple[np.ndarray, float, float]:
    """Dispatch ridge / ListNet / RankNet from ``ForecastTrainConfig``."""
    kwargs = ridge_kwargs_from_train_cfg(train_cfg, {"cs_min_names": min_names})
    objective = str(getattr(train_cfg, "ridge_objective", "ridge") or "ridge").lower()
    shared = dict(
        ridge=kwargs["ridge"],
        min_names=min_names,
        cs_demean=kwargs["cs_demean"],
        cs_zscore=kwargs["cs_zscore"],
        rank_target=kwargs["rank_target"],
        feature_mask_bool=kwargs["feature_mask_bool"],
        drop_crashes=kwargs["drop_crashes"],
    )
    if objective == "listnet":
        return fit_listnet_xy(x, y, dates, **shared)
    if objective == "ranknet":
        return fit_ranknet_xy(x, y, dates, **shared)
    if objective in ("long_only", "longonly", "long-only"):
        from forecast.levers import fit_long_only_xy, promoted_feature_mask

        mask = kwargs["feature_mask_bool"]
        stable = str(getattr(train_cfg, "ridge_year_stable", "") or "")
        if stable:
            mask = promoted_feature_mask(
                str(getattr(train_cfg, "ridge_features", "all") or "all"),
                x,
                y,
                dates,
                year_stable=stable,
                min_names=min_names,
            )
        return fit_long_only_xy(
            x,
            y,
            dates,
            ridge=kwargs["ridge"],
            min_names=min_names,
            quantile=float(getattr(train_cfg, "long_only_quantile", 0.2) or 0.2),
            feat_winsor=kwargs["feat_winsor"],
            feature_mask_bool=mask,
            date_halflife=kwargs["date_halflife"],
        )
    return fit_ridge_xy(
        x,
        y,
        dates,
        date_halflife=kwargs["date_halflife"],
        y_winsor=kwargs["y_winsor"],
        feat_winsor=kwargs["feat_winsor"],
        drop_disp_q=kwargs["drop_disp_q"],
        huber_delta=kwargs["huber_delta"],
        sign_constrain=kwargs["sign_constrain"],
        year_balance=kwargs["year_balance"],
        **shared,
    )


def ridge_kwargs_from_train_cfg(train_cfg: Any, bundle: dict[str, Any]) -> dict[str, Any]:
    mask_mode = str(getattr(train_cfg, "ridge_features", "all") or "all")
    return {
        "ridge": float(getattr(train_cfg, "ridge_skip", 1.0)),
        "min_names": int(bundle.get("cs_min_names", 8)),
        "cs_demean": bool(getattr(train_cfg, "ridge_cs_demean", True)),
        "cs_zscore": bool(getattr(train_cfg, "ridge_cs_zscore", False)),
        "rank_target": bool(getattr(train_cfg, "ridge_rank_target", False)),
        "feature_mask_bool": feature_mask(mask_mode),
        "date_halflife": float(getattr(train_cfg, "ridge_date_halflife", 0.0) or 0.0),
        "y_winsor": float(getattr(train_cfg, "ridge_y_winsor", 0.0) or 0.0),
        "feat_winsor": float(getattr(train_cfg, "ridge_feat_winsor", 0.0) or 0.0),
        "drop_disp_q": float(getattr(train_cfg, "ridge_drop_disp_q", 0.0) or 0.0),
        "huber_delta": float(getattr(train_cfg, "ridge_huber", 0.0) or 0.0),
        "sign_constrain": bool(getattr(train_cfg, "ridge_sign_constrain", False)),
        "drop_crashes": bool(getattr(train_cfg, "ridge_drop_crashes", False)),
        "year_balance": bool(getattr(train_cfg, "ridge_year_balance", False)),
    }
