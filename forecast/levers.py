"""Accuracy levers 1–6 and 8 on the overnight residual CS book.

Locked-val honesty
------------------
Promote only if locked-val mean CS IC lifts by ``VAL_LIFT`` (>= 0.003) and
val-2017 is not killed (``VAL_2017_KEEP``). Do **not** retarget from test or
from 2023. Close-to-close already lost bigger Mamba / Dynamic A. Overnight
gap residual is the active path; the runnable book is live long-only at 15%
causal vol. Never headline ``vol_target=1``.

When CS IC and live long-only IR disagree, keep the live long-only book.

Lever 7 (external vendor ingest) is **not** implemented. Hooks only: drop a
vendor parquet with the same OHLCV contract into ``data/``; do not add a
new download client here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from forecast.overnight import VAL_2017_KEEP, VAL_LIFT
from forecast.ridge import (
    apply_skip_ic_shrink,
    cs_stats,
    feature_mask,
    fit_residual_mlp,
    fit_ridge_xy,
    late_train_holdout_mask,
    mean_cs_ic,
    predict_residual_mlp,
    trailing_skip_ic_stats,
    year_cs_ics,
    year_feature_ics,
    year_stable_mask,
)

# Re-export the overnight val-gate so ablations share one number.
__all__ = [
    "VAL_LIFT",
    "VAL_2017_KEEP",
    "VENDOR_INGEST_HOOK",
    "LABEL_ESTIMANDS",
    "val_gate",
    "year_sign_consistency_mask",
    "causal_regime_scale",
    "fit_long_only_xy",
    "blend_skip_mlp",
    "calibration_scale",
    "apply_gap_risk_cap",
    "adv_cost_scale",
    "train_era_liquid_names",
    "dollar_adv_series",
    "ablation_row",
    "format_ablation_table",
]

VENDOR_INGEST_HOOK = (
    "Vendor data-quality ingest is intentionally not implemented. "
    "Owen will supply datasets later. Drop vendor parquets into data/ with "
    "columns datetime, open, high, low, close, volume (same contract as "
    "Yahoo/Stooq). Do not add a vendor client in forecast.download."
)

# Separate overnight-family estimands. Do not mix ICs into close-to-close.
LABEL_ESTIMANDS: tuple[tuple[str, int], ...] = (
    ("overnight", 0),
    ("open_fill", 15),
    ("open_fill", 30),
    ("session", 0),
)


def val_gate(
    candidate_val: float,
    baseline_val: float,
    *,
    candidate_2017: float | None = None,
    baseline_2017: float | None = None,
    min_lift: float = VAL_LIFT,
    keep_2017: float = VAL_2017_KEEP,
) -> bool:
    """True iff locked-val lift clears and val-2017 is not killed.

    Test / 2023 numbers are for reporting only. Never pass them here.
    """
    cand = float(candidate_val)
    base = float(baseline_val)
    if not np.isfinite(cand) or not np.isfinite(base):
        return False
    if cand - base < float(min_lift):
        return False
    if candidate_2017 is not None and np.isfinite(candidate_2017):
        if float(candidate_2017) < float(keep_2017):
            return False
        if (
            baseline_2017 is not None
            and np.isfinite(baseline_2017)
            and float(candidate_2017) < float(baseline_2017) - 0.01
        ):
            return False
    return True


def year_sign_consistency_mask(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
    min_frac: float = 0.7,
    min_abs: float = 0.003,
    recency_halflife_years: float = 5.0,
    trailing_years: int = 4,
    trailing_min_frac: float = 0.75,
) -> np.ndarray:
    """Train-only year-stable mask with recency and a trailing-year sign lock.

    Later train years weigh more (2023-like structure in late train matters
    more than 1999). ``trailing_years`` requires the most recent train years
    to agree on sign so a feature that died before the val cut is dropped.
    Never uses val/test arrays — those belong to the val-gate, not the mask.
    """
    f = int(x.shape[1]) if x.ndim == 2 else 0
    by_year = year_feature_ics(x, y, dates, min_names=min_names)
    if not by_year:
        return np.ones(f, dtype=bool)
    years = np.asarray(sorted(by_year), dtype=np.int64)
    stacked = np.stack([by_year[int(y)] for y in years], axis=0)
    ymax = int(years.max())
    hl = max(float(recency_halflife_years), 1e-6)
    w = 0.5 ** ((ymax - years.astype(np.float64)) / hl)
    keep = np.zeros(f, dtype=bool)
    trail_n = max(1, int(trailing_years))
    for j in range(f):
        col = stacked[:, j]
        finite = np.isfinite(col) & (np.abs(col) >= float(min_abs))
        if int(finite.sum()) < 2:
            continue
        ww = w[finite]
        signed = np.sign(col[finite])
        pos = float((ww * (signed > 0)).sum() / max(float(ww.sum()), 1e-12))
        stable = pos >= float(min_frac) or (1.0 - pos) >= float(min_frac)
        if not stable:
            continue
        idx = np.flatnonzero(finite)
        if idx.size:
            last = idx[-min(trail_n, idx.size) :]
            tcol = col[last]
            tpos = float((tcol > 0).mean())
            if tpos < float(trailing_min_frac) and (1.0 - tpos) < float(trailing_min_frac):
                continue
        keep[j] = True
    if not bool(keep.any()):
        return np.ones(f, dtype=bool)
    return keep


def causal_regime_scale(
    dates: np.ndarray,
    *,
    date_keys: np.ndarray,
    trailing_ic: np.ndarray,
    trailing_t: np.ndarray,
    train_ic: float,
    spy_vol: dict[int, float] | None = None,
    train_vol_cut: float | None = None,
    mode: str = "flatten_ic",
    vol_and_dead: bool = True,
) -> np.ndarray:
    """Per-row sizing scale in ``[0, 1]`` from causal trailing CS IC + vol.

    Identical scale on every name of a date does **not** change Pearson CS IC
    or quantile ranks. Apply this to **leverage / NAV**, not to scores, unless
    ``mode`` flattens (factor=0), which zeros the book that night.

    ``vol_and_dead``: also flatten when trailing IC is dead *and* spy vol is
    above the train-era cut (known at t; ``spy_vol`` must be causal).
    """
    dummy = np.ones(np.asarray(dates).shape[0], dtype=np.float64)
    scaled = apply_skip_ic_shrink(
        dummy,
        dates,
        date_keys=date_keys,
        trailing_ic=trailing_ic,
        trailing_t=trailing_t,
        train_ic=train_ic,
        mode=mode,
    )
    if not vol_and_dead or not spy_vol or train_vol_cut is None:
        return scaled
    cut = float(train_vol_cut)
    dates_i = np.asarray(dates, dtype=np.int64)
    keys = np.asarray(date_keys, dtype=np.int64)
    loc = np.searchsorted(keys, dates_i)
    loc = np.clip(loc, 0, max(keys.size - 1, 0))
    match = keys.size > 0
    match = (keys[loc] == dates_i) if keys.size else np.zeros(dates_i.shape, dtype=bool)
    mu = np.full(dates_i.shape[0], np.nan, dtype=np.float64)
    if keys.size:
        mu[match] = np.asarray(trailing_ic, dtype=np.float64)[loc[match]]
    for i, key in enumerate(dates_i):
        vol = spy_vol.get(int(key))
        if vol is None or not np.isfinite(vol):
            continue
        if vol >= cut and np.isfinite(mu[i]) and mu[i] <= 0.0:
            scaled[i] = 0.0
    return scaled


def calibration_scale(
    pred: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 8,
    lookback_days: int = 63,
    min_obs: int = 15,
    train_ic: float | None = None,
    mode: str = "flatten_ic",
    spy_vol: dict[int, float] | None = None,
    train_vol_cut: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal per-date sizing scale aligned with ``dates``.

    Trailing CS IC uses labels strictly before ``t``. Returns
    ``(unique_date_keys, scale_per_date)``.
    """
    keys, mu, tt, _n = trailing_skip_ic_stats(
        pred,
        y,
        dates,
        lookback_days=int(lookback_days),
        min_names=min_names,
        min_obs=min_obs,
    )
    if train_ic is None or not np.isfinite(train_ic):
        stats = cs_stats(pred, y, dates, min_names=min_names)
        train_ic = float(stats.get("cs_ic", 1.0) or 1.0)
    row_scale = causal_regime_scale(
        dates,
        date_keys=keys,
        trailing_ic=mu,
        trailing_t=tt,
        train_ic=float(train_ic),
        spy_vol=spy_vol,
        train_vol_cut=train_vol_cut,
        mode=mode,
    )
    date_scale = np.ones(keys.shape[0], dtype=np.float64)
    dates_i = np.asarray(dates, dtype=np.int64)
    for i, key in enumerate(keys):
        sel = dates_i == int(key)
        if bool(sel.any()):
            date_scale[i] = float(row_scale[sel][0])
    return keys.astype(np.int64), date_scale


def apply_gap_risk_cap(
    weights: np.ndarray,
    name_vol: np.ndarray | None,
    *,
    cap: float,
) -> np.ndarray:
    """Scale a date-row of weights so ``sum(|w| * vol) <= cap``.

    ``name_vol`` is known at t (``vol_level`` or trailing overnight vol).
    ``cap`` is in the same units as ``vol`` times NAV (e.g. 0.02 ≈ 2% overnight
    vol budget). 0 or missing vol is no cap. Does not use realized labels.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    if cap is None or float(cap) <= 0 or name_vol is None:
        return w
    squeeze = w.ndim == 1
    if squeeze:
        w = w.reshape(1, -1)
    vol = np.asarray(name_vol, dtype=np.float64)
    if vol.ndim == 1:
        vol = np.broadcast_to(vol.reshape(1, -1), w.shape).copy()
    vol = np.nan_to_num(np.clip(vol, 0.0, None), nan=0.0)
    budget = float(cap)
    for i in range(w.shape[0]):
        exposure = float((np.abs(w[i]) * vol[i]).sum())
        if exposure > budget and exposure > 1e-12:
            w[i] *= budget / exposure
    return w[0] if squeeze else w


def adv_cost_scale(
    dollar_adv: np.ndarray,
    *,
    k: float,
    max_mult: float = 4.0,
    min_mult: float = 1.0,
) -> np.ndarray:
    """Within-row ``(median ADV / ADV)^k``, clipped. Thin names cost more.

    Missing/zero ADV → ``max_mult`` (treat as expensive, not free). Causal:
    ``dollar_adv`` is ``volume_t * close_t``.
    """
    adv = np.asarray(dollar_adv, dtype=np.float64)
    squeeze = adv.ndim == 1
    if squeeze:
        adv = adv.reshape(1, -1)
    out = np.ones_like(adv)
    kk = float(k)
    if kk <= 0:
        return out[0] if squeeze else out
    lo, hi = float(min_mult), float(max_mult)
    for i in range(adv.shape[0]):
        row = adv[i]
        finite = np.isfinite(row) & (row > 0)
        if int(finite.sum()) < 2:
            out[i] = np.where(finite, lo, hi)
            continue
        med = float(np.median(row[finite]))
        if med <= 0:
            continue
        ratio = med / np.clip(row, 1e-12, None)
        scaled = np.clip(np.power(ratio, kk), lo, hi)
        out[i] = np.where(finite, scaled, hi)
    return out[0] if squeeze else out


def dollar_adv_series(volume: np.ndarray, close: np.ndarray) -> np.ndarray:
    """``volume * close``; non-positive prices/volume → NaN, not a fake ADV."""
    v = np.asarray(volume, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    out = v * c
    bad = (~np.isfinite(out)) | (v <= 0) | (c <= 0)
    out = np.where(bad, np.nan, out)
    return out.astype(np.float64)


def train_era_liquid_names(
    medians: Mapping[str, float],
    names: Sequence[str],
    *,
    floor_usd: float = 0.0,
    floor_pctile: float = 0.0,
    min_names: int = 0,
) -> list[str]:
    """Lock membership from train-era median dollar ADV. No val/test peek.

    If the floor would empty the book below ``min_names``, keep the top-N by
    train-era ADV rather than silently using the full universe.
    """
    ordered = [str(s) for s in names]
    if not ordered:
        return []
    vals = {s: float(medians.get(s, float("nan"))) for s in ordered}
    keep = list(ordered)
    usd = float(floor_usd or 0.0)
    pct = float(floor_pctile or 0.0)
    if usd > 0:
        keep = [s for s in keep if np.isfinite(vals[s]) and vals[s] >= usd]
    if pct > 0:
        finite = np.asarray([vals[s] for s in ordered if np.isfinite(vals[s])], dtype=np.float64)
        if finite.size:
            cut = float(np.nanpercentile(finite, 100.0 * min(max(pct, 0.0), 1.0)))
            keep = [s for s in keep if np.isfinite(vals[s]) and vals[s] >= cut]
    need = int(min_names or 0)
    if need > 0 and len(keep) < need:
        ranked = sorted(
            ordered,
            key=lambda s: vals[s] if np.isfinite(vals[s]) else -1.0,
            reverse=True,
        )
        keep = ranked[: max(need, 1)]
    return keep


def fit_long_only_xy(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    ridge: float = 10.0,
    min_names: int = 8,
    quantile: float = 0.2,
    feat_winsor: float = 3.0,
    feature_mask_bool: np.ndarray | None = None,
    date_halflife: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Closed-form ridge whose target is the long sleeve (top CS quantile).

    Names below the within-date top quantile get rank 0 so the skip learns
    the runnable long-only book, not a 50/50 long-short.
    """
    y_lo = np.asarray(y, dtype=np.float64).copy()
    d = np.asarray(dates, dtype=np.int64)
    q = min(0.49, max(0.05, float(quantile)))
    for key in np.unique(d):
        sel = d == key
        n = int(sel.sum())
        if n < int(min_names):
            y_lo[sel] = 0.0
            continue
        yd = y_lo[sel]
        k = max(1, int(np.floor(n * q)))
        order = np.argsort(yd, kind="mergesort")
        ranks = np.zeros(n, dtype=np.float64)
        top = order[-k:]
        ranks[top] = np.arange(1, k + 1, dtype=np.float64)
        ranks = ranks - ranks.mean()
        if n > 1:
            s = float(ranks.std())
            if s > 1e-8:
                ranks = ranks / s
        y_lo[sel] = ranks
    return fit_ridge_xy(
        x,
        y_lo,
        d,
        ridge=ridge,
        min_names=min_names,
        cs_demean=True,
        rank_target=False,
        feat_winsor=feat_winsor,
        feature_mask_bool=feature_mask_bool,
        date_halflife=date_halflife,
    )


def blend_skip_mlp(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    skip_pred: np.ndarray,
    *,
    min_names: int = 8,
    hidden: int = 8,
    ridge: float = 25.0,
    steps: int = 200,
    mixes: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    seed: int = 0,
) -> dict[str, Any]:
    """Skip + tiny ReLU residual. Mix chosen on a leak-free late-train slice.

    Returns weights and the selected mix. Caller val-gates against skip-only
    on locked val; a mix of 0 is skip-only (discard the nonlinear head).
    """
    fit_mask, hold_mask = late_train_holdout_mask(dates)
    resid = np.asarray(y, dtype=np.float64) - np.asarray(skip_pred, dtype=np.float64)
    if not bool(fit_mask.any()):
        fit_mask = np.ones(dates.shape[0], dtype=bool)
        hold_mask = fit_mask
    w1, b1, w2 = fit_residual_mlp(
        x[fit_mask],
        resid[fit_mask],
        dates[fit_mask],
        hidden=hidden,
        ridge=ridge,
        steps=steps,
        min_names=min_names,
        seed=seed,
    )
    mlp = predict_residual_mlp(x, dates, w1, b1, w2, min_names=min_names)
    skip = np.asarray(skip_pred, dtype=np.float64)
    best_mix = 0.0
    best_ic = mean_cs_ic(skip[hold_mask], y[hold_mask], dates[hold_mask], min_names=min_names)
    hold_ics: list[dict[str, float]] = []
    for mix in mixes:
        m = float(mix)
        pred = skip + m * mlp
        ic = mean_cs_ic(pred[hold_mask], y[hold_mask], dates[hold_mask], min_names=min_names)
        hold_ics.append({"mix": m, "cs_ic": float(ic)})
        if np.isfinite(ic) and ((not np.isfinite(best_ic)) or ic > best_ic):
            best_ic = ic
            best_mix = m
    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "mix": float(best_mix),
        "hold_ic": float(best_ic) if np.isfinite(best_ic) else float("nan"),
        "hold_ics": hold_ics,
        "mlp_pred": mlp,
        "pred": skip + float(best_mix) * mlp,
    }


def year_slice_stats(
    pred: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int,
    year: int,
) -> dict[str, float]:
    rows = year_cs_ics(pred, y, dates, min_names=min_names)
    for row in rows:
        if int(row.get("year", -1)) == int(year):
            return row
    return {"year": float(year), "cs_ic": float("nan"), "cs_ic_tstat": float("nan")}


def ablation_row(
    *,
    name: str,
    lever: str,
    baseline_val: float,
    baseline_2017: float | None,
    val: Mapping[str, Any],
    test: Mapping[str, Any] | None = None,
    live_long_only_ir: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    """One ablation line. ``promote`` uses locked val only."""
    val_ic = float(val.get("cs_ic", float("nan")))
    val_2017 = float(val.get("cs_ic_2017", val.get("val_2017", float("nan"))))
    promote = val_gate(val_ic, float(baseline_val), candidate_2017=val_2017, baseline_2017=baseline_2017)
    test = test or {}
    return {
        "name": str(name),
        "lever": str(lever),
        "val_cs_ic": val_ic,
        "val_t": float(val.get("cs_ic_tstat", float("nan"))),
        "val_2017": val_2017,
        "val_lift": float(val_ic - float(baseline_val)) if np.isfinite(val_ic) else float("nan"),
        "test_cs_ic": float(test.get("cs_ic", float("nan"))),
        "test_t": float(test.get("cs_ic_tstat", float("nan"))),
        "test_2023": float(test.get("cs_ic_2023", test.get("test_2023", float("nan")))),
        "live_long_only_ir": (
            float(live_long_only_ir) if live_long_only_ir is not None else float("nan")
        ),
        "promote": bool(promote),
        "decision": "promote" if promote else "discard",
        "note": str(note),
    }


def format_ablation_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Markdown table: promoted vs discarded. Test columns are report-only."""
    lines = [
        "| lever | name | val CS IC | lift | val-2017 | test CS IC | live LO IR | decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {lever} | {name} | {val:+.4f} | {lift:+.4f} | {v17:+.4f} | {te:+.4f} | {lo:+.3f} | **{dec}** |".format(
                lever=row.get("lever", ""),
                name=row.get("name", ""),
                val=float(row.get("val_cs_ic", float("nan"))),
                lift=float(row.get("val_lift", float("nan"))),
                v17=float(row.get("val_2017", float("nan"))),
                te=float(row.get("test_cs_ic", float("nan"))),
                lo=float(row.get("live_long_only_ir", float("nan"))),
                dec=row.get("decision", "discard"),
            )
        )
    return "\n".join(lines)


def promoted_feature_mask(
    mode: str,
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    year_stable: str = "",
    min_names: int = 8,
) -> np.ndarray:
    """Combine ``no_long_ts`` (etc.) with an optional train-only year-stable mask."""
    keep = feature_mask(mode)
    raw = (year_stable or "").strip().lower()
    if raw in ("train", "train_recency", "recency"):
        extra = year_sign_consistency_mask(x, y, dates, min_names=min_names)
        keep = keep & extra
    elif raw in ("train_simple", "simple"):
        extra = year_stable_mask(x, y, dates, min_names=min_names)
        keep = keep & extra
    return keep


def univariate_sign_flip_count(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
    min_abs: float = 0.003,
) -> np.ndarray:
    """Per-column count of train-year sign flips (diagnostic for 2023)."""
    by_year = year_feature_ics(x, y, dates, min_names=min_names)
    f = int(x.shape[1]) if x.ndim == 2 else 0
    out = np.zeros(f, dtype=np.int64)
    if not by_year:
        return out
    years = sorted(by_year)
    for j in range(f):
        prev = None
        flips = 0
        for year in years:
            v = float(by_year[int(year)][j])
            if not np.isfinite(v) or abs(v) < float(min_abs):
                continue
            sign = 1.0 if v > 0 else -1.0
            if prev is not None and sign != prev:
                flips += 1
            prev = sign
        out[j] = flips
    return out
