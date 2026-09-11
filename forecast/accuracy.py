"""Locked-window TRUE/FALSE overnight accuracy (not live P&L).

The promoted overnight skip predicts a *residual* in vol units. This module
converts that to an implied overnight log-return ``pred * sigma`` (same as
``generate.py``) and scores:

- direction vs realized ``r_on = log(open_{t+1}) - log(close_t)``
- excess hit rate vs the unconditional overnight-up drift (always-long)
- implied next-open vs actual next open
- long-only book up-rate on the within-date top residual names
- short-sleeve overnight down-rate on the within-date bottom residual names

Default recipe is the PR #5 overnight skip (rank-target ridge, ``no_long_ts``).
PR #7 levers stay off unless a checkpoint documents them.

Train-only accuracy readouts (affine residual→raw overnight, drift-veto,
piecewise/bin calibration, weekday intercepts, TS overnight ridge, sign ridge,
ADV sleeve, confidence slices, conditional high-|pred| left-tail/confidence
blend) are fit on TRAIN and gated on locked VAL. They do not replace the
residual CS skip. Next open is never a feature.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from forecast.alphavantage import symbol_parquet_path
from forecast.config import DataConfig, interval_data_kwargs
from forecast.data import (
    FEATURE_NAMES,
    SymbolArrays,
    _date_keys,
    assemble_panel,
    build_datasets,
    load_bars,
    symbol_from_path,
)
from forecast.overnight import OVERNIGHT_FORMULA, VAL_2017_KEEP, VAL_LIFT, formula_log_line
from forecast.ridge import (
    cs_stats,
    dates_to_year,
    feature_mask,
    fit_ridge_xy,
    labelled_rows as ridge_labelled_rows,
)

# Accuracy readout gates (locked VAL). CS IC gate stays VAL_LIFT / VAL_2017_KEEP.
DIR_LIFT = 0.002  # 0.2 pp hit-rate vs the residual*sigma baseline
TURNOVER_COL = FEATURE_NAMES.index("turnover_z") if "turnover_z" in FEATURE_NAMES else None
VOL_LEVEL_COL = FEATURE_NAMES.index("vol_level") if "vol_level" in FEATURE_NAMES else None
# PR #8 locked-TEST residual*sigma print (do not retarget; compare on the same window).
BASELINE_TEST = {
    "dir_pct": 51.1,
    "mae_usd": 1.17,
    "mae_pct": 0.00686,
}

# Match scripts/cs_overnight.py. Last-bar ridge does not use seq_len windows.
PROMOTED_SKIP = dict(
    ridge=10.0,
    rank_target=True,
    feat_winsor=3.0,
    mask_mode="no_long_ts",
)


def overnight_skip_data_config(
    data_dir: str,
    universe: str = "liquid",
    *,
    label_return: str = "overnight",
    interval: str = "daily",
    sector_residual: bool = True,
) -> DataConfig:
    """Locked overnight skip DataConfig (same cuts/features as ``cs_overnight``).

    ``sector_residual=True`` residualizes overnight y vs the mapped sector
    ETF's same-night overnight (SPY if that parquet is missing).
    ``sector_residual=False`` is the SPY/market overnight residual baseline.
    Sector/index ETFs stay out of the trading book (``equities_only=True``).
    """
    preset = interval_data_kwargs(interval)
    seq_len = 32 if interval == "daily" else int(preset["seq_len"])
    synthetic = universe in ("", "synthetic")
    return DataConfig(
        data_dir=data_dir,
        interval=interval,
        horizon=1,
        seq_len=seq_len,
        stride=1,
        min_context=8 if interval == "daily" else int(preset["min_context"]),
        warmup_bars=16 if interval == "daily" else int(preset["warmup_bars"]),
        vol_halflife=preset["vol_halflife"],
        z_window=preset["z_window"],
        z_min_periods=preset["z_min_periods"],
        max_abs_log_return=preset["max_abs_log_return"],
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=8 if synthetic else 30,
        allow_mixed_prices=synthetic,
        cs_zscore=True,
        universe="" if synthetic else universe,
        sector_residual=bool(sector_residual),
        equities_only=True,
        train_from="" if synthetic else "1999-01-01",
        label_return=label_return,
        fill_minutes=0,
    )


def two_sided_normal_p(z: float) -> float:
    """Two-sided p-value from a standard-normal z (erfc)."""
    if not np.isfinite(z):
        return float("nan")
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def weekday_of_dates(dates: np.ndarray) -> np.ndarray:
    """Monday=0 … Sunday=6 from day-since-epoch keys (known at close t)."""
    d = np.asarray(dates, dtype=np.int64)
    if d.size == 0:
        return np.zeros((0,), dtype=np.int64)
    cal = np.datetime64("1970-01-01") + d.astype("timedelta64[D]")
    return pd.to_datetime(cal).dayofweek.to_numpy(dtype=np.int64)


def recency_weights(dates: np.ndarray, *, halflife_years: float = 6.0) -> np.ndarray:
    """Train-only year recency. Newer train years get more weight; no val/test."""
    years = dates_to_year(np.asarray(dates, dtype=np.int64)).astype(np.float64)
    if years.size == 0:
        return np.zeros((0,), dtype=np.float64)
    hl = max(float(halflife_years), 1e-6)
    ymax = float(np.max(years))
    return np.power(0.5, (ymax - years) / hl)


def weighted_median(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64)
    if weights is None:
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else 0.0
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return 0.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    idx = int(np.searchsorted(cw, 0.5 * cw[-1], side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def hit_rate_vs_p0(hits: np.ndarray, p0: float) -> dict[str, float]:
    """Hit rate vs a known base rate (overnight-up drift), not vs 50%."""
    h = np.asarray(hits, dtype=np.float64)
    h = h[np.isfinite(h)]
    n = int(h.size)
    p_base = float(p0)
    empty = {
        "n": float(n),
        "hit_rate": float("nan"),
        "hit_rate_pct": float("nan"),
        "p0": p_base,
        "p0_pct": float(100.0 * p_base) if np.isfinite(p_base) else float("nan"),
        "excess": float("nan"),
        "excess_pp": float("nan"),
        "z_vs_p0": float("nan"),
        "p_vs_p0": float("nan"),
    }
    if n <= 0 or not np.isfinite(p_base) or p_base <= 0.0 or p_base >= 1.0:
        return empty
    k = float(h.sum())
    p = k / n
    se = math.sqrt(p_base * (1.0 - p_base) / n)
    z = (p - p_base) / se if se > 0 else float("nan")
    return {
        "n": float(n),
        "hit_rate": float(p),
        "hit_rate_pct": float(100.0 * p),
        "p0": p_base,
        "p0_pct": float(100.0 * p_base),
        "excess": float(p - p_base),
        "excess_pp": float(100.0 * (p - p_base)),
        "z_vs_p0": float(z),
        "p_vs_p0": two_sided_normal_p(z),
    }


def hit_rate_inference(hits: np.ndarray) -> dict[str, float]:
    """Hit rate vs 50%, with binomial-normal z/p and a 0/1 t-stat."""
    h = np.asarray(hits, dtype=np.float64)
    h = h[np.isfinite(h)]
    n = int(h.size)
    if n <= 0:
        return {
            "n": 0.0,
            "n_correct": 0.0,
            "hit_rate": float("nan"),
            "hit_rate_pct": float("nan"),
            "z_vs_half": float("nan"),
            "p_vs_half": float("nan"),
            "t_vs_half": float("nan"),
        }
    k = float(h.sum())
    p = k / n
    se = math.sqrt(0.25 / n)
    z = (p - 0.5) / se if se > 0 else float("nan")
    if n >= 2:
        s = float(h.std(ddof=1))
        t = (p - 0.5) / (s / math.sqrt(n)) if s > 0 else float("nan")
    else:
        t = float("nan")
    return {
        "n": float(n),
        "n_correct": k,
        "hit_rate": float(p),
        "hit_rate_pct": float(100.0 * p),
        "z_vs_half": float(z),
        "p_vs_half": two_sided_normal_p(z),
        "t_vs_half": float(t),
    }


def abs_error_block(err: np.ndarray) -> dict[str, float]:
    e = np.asarray(err, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {
            "n": 0.0,
            "mae": float("nan"),
            "median_ae": float("nan"),
            "rmse": float("nan"),
        }
    return {
        "n": float(e.size),
        "mae": float(np.mean(e)),
        "median_ae": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean(e**2))),
    }


def direction_hits(pred: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Boolean hits where realized != 0. ``sign(0)`` realized rows are dropped."""
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(realized, dtype=np.float64)
    moved = np.isfinite(p) & np.isfinite(r) & (r != 0.0)
    return (np.sign(p[moved]) == np.sign(r[moved])).astype(np.float64)


def mean_cs_sign_hit(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> dict[str, float]:
    """Mean within-date sign accuracy on demeaned pred/target (CS direction)."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    d = np.asarray(dates, dtype=np.int64)
    rates: list[float] = []
    for key in np.unique(d):
        sel = d == key
        if int(sel.sum()) < int(min_names):
            continue
        pd_ = p[sel] - np.mean(p[sel])
        yd = y[sel] - np.mean(y[sel])
        hits = direction_hits(pd_, yd)
        if hits.size:
            rates.append(float(hits.mean()))
    if not rates:
        return {"cs_sign_hit": float("nan"), "cs_n_dates": 0.0}
    arr = np.asarray(rates, dtype=np.float64)
    return {
        "cs_sign_hit": float(arr.mean()),
        "cs_sign_hit_pct": float(100.0 * arr.mean()),
        "cs_n_dates": float(arr.size),
    }


def overnight_px_lookup(
    path: str | Path,
    cfg: DataConfig,
) -> dict[int, tuple[float, float]]:
    """date_key -> (close_t, open_{t+h}) on the same daily session grid as training."""
    panel = assemble_panel(load_bars(path), cfg, symbol_from_path(path))
    dates = _date_keys(panel)
    close = panel["close"].to_numpy(dtype=np.float64)
    opn = panel["open"].to_numpy(dtype=np.float64)
    h = max(1, int(cfg.horizon))
    nxt = np.full_like(close, np.nan)
    if close.size > h:
        nxt[:-h] = opn[h:]
    out: dict[int, tuple[float, float]] = {}
    for i, key in enumerate(dates):
        c = float(close[i])
        o = float(nxt[i])
        if c > 0 and np.isfinite(c) and o > 0 and np.isfinite(o):
            out[int(key)] = (c, o)
    return out


def collect_eval_frame(
    symbols: Sequence[SymbolArrays],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    px_by_symbol: dict[str, dict[int, tuple[float, float]]],
    *,
    weights: np.ndarray,
    bias: float,
    return_features: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """One labelled last-bar row per name/date with pred, y, r_on, prices.

    ``turnover_z`` is the causal feature known at close ``t`` (not a label).
    Next open is looked up only to form ``r_on`` / ``next_open``.
    """
    mean = np.asarray(feature_mean, dtype=np.float64)
    std = np.asarray(feature_std, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    feat_rows: list[np.ndarray] = []
    for sym in symbols:
        if not bool(sym.valid.any()) or sym.dates is None:
            continue
        lookup = px_by_symbol.get(sym.symbol) or {}
        raw = sym.features[sym.valid].astype(np.float64, copy=False)
        x = (raw - mean) / std
        pred = x @ w + float(bias)
        y = sym.target[sym.valid].astype(np.float64, copy=False)
        scale = sym.scale[sym.valid].astype(np.float64, copy=False)
        dates = sym.dates[sym.valid].astype(np.int64, copy=False)
        for i in range(y.size):
            key = int(dates[i])
            px = lookup.get(key)
            if px is None:
                continue
            close, nxt_open = px
            r_on = math.log(nxt_open) - math.log(close)
            turn = float("nan")
            vol = float("nan")
            if TURNOVER_COL is not None and TURNOVER_COL < raw.shape[1]:
                turn = float(raw[i, TURNOVER_COL])
            if VOL_LEVEL_COL is not None and VOL_LEVEL_COL < raw.shape[1]:
                vol = float(raw[i, VOL_LEVEL_COL])
            rows.append(
                {
                    "symbol": sym.symbol,
                    "date": key,
                    "pred": float(pred[i]),
                    "y": float(y[i]),
                    "scale": float(scale[i]),
                    "close": float(close),
                    "next_open": float(nxt_open),
                    "r_on": float(r_on),
                    "turnover_z": turn,
                    "vol_level": vol,
                }
            )
            if return_features:
                feat_rows.append(x[i])
    empty_cols = [
        "symbol",
        "date",
        "pred",
        "y",
        "scale",
        "close",
        "next_open",
        "r_on",
        "turnover_z",
        "vol_level",
        "weekday",
        "pred_r",
        "implied_open",
        "implied_open_given_hedge",
    ]
    if not rows:
        df = pd.DataFrame(columns=empty_cols)
        if return_features:
            return df, np.zeros((0, int(mean.shape[0])), dtype=np.float64)
        return df
    df = pd.DataFrame(rows)
    df["weekday"] = weekday_of_dates(df["date"].to_numpy(dtype=np.int64))
    df["pred_r"] = df["pred"] * df["scale"]
    df["implied_open"] = df["close"] * np.exp(df["pred_r"])
    # resid = y * scale = r_on - beta * r_hedge. Adding the realized hedge
    # term is *not* a live next-open forecast (uses next hedge open).
    hedge = df["r_on"] - df["y"] * df["scale"]
    df["implied_open_given_hedge"] = df["close"] * np.exp(df["pred_r"] + hedge)
    if return_features:
        return df, np.stack(feat_rows, axis=0).astype(np.float64)
    return df


def sleeve_book_block(
    df: pd.DataFrame,
    *,
    score_col: str = "pred",
    side: str = "long",
    q: float = 0.80,
    min_names: int = 3,
) -> dict[str, float]:
    """Overnight hit rate on a within-date residual-pred quantile sleeve.

    ``side='long'``: names with score >= q (default top 20% if q=0.80),
    overnight *up*-rate vs the unconditional up-rate.
    ``side='short'``: names with score <= q (default bottom 20% if q=0.20),
    overnight *down*-rate vs the unconditional down-rate.
    """
    is_long = str(side or "long").strip().lower() != "short"
    empty = {
        "side": "long" if is_long else "short",
        "quantile": float(q),
        "n": 0.0,
        "coverage": float("nan"),
        "hit_pct": float("nan"),
        "up_pct": float("nan"),
        "down_pct": float("nan"),
        "uncond_up_pct": float("nan"),
        "uncond_down_pct": float("nan"),
        "excess_pp": float("nan"),
        "n_dates": 0.0,
    }
    if df.empty or score_col not in df.columns:
        return empty
    dates = df["date"].to_numpy(dtype=np.int64)
    score = df[score_col].to_numpy(dtype=np.float64)
    r_on = df["r_on"].to_numpy(dtype=np.float64)
    mask = np.zeros(len(df), dtype=bool)
    n_dates = 0
    for key in np.unique(dates):
        sel = dates == key
        row = score[sel]
        if int(np.isfinite(row).sum()) < int(min_names):
            continue
        cut = float(np.nanquantile(row, float(q)))
        if is_long:
            mask[sel] = np.isfinite(row) & (row >= cut)
        else:
            mask[sel] = np.isfinite(row) & (row <= cut)
        n_dates += 1
    moved = mask & np.isfinite(r_on) & (r_on != 0.0)
    uncond = r_on[np.isfinite(r_on) & (r_on != 0.0)]
    uncond_up = float((uncond > 0).mean()) if uncond.size else float("nan")
    uncond_down = float((uncond < 0).mean()) if uncond.size else float("nan")
    empty["n_dates"] = float(n_dates)
    empty["uncond_up_pct"] = (
        float(100.0 * uncond_up) if np.isfinite(uncond_up) else float("nan")
    )
    empty["uncond_down_pct"] = (
        float(100.0 * uncond_down) if np.isfinite(uncond_down) else float("nan")
    )
    empty["coverage"] = float(mask.mean()) if mask.size else float("nan")
    if int(moved.sum()) == 0:
        return empty
    up = float((r_on[moved] > 0).mean())
    down = float((r_on[moved] < 0).mean())
    if is_long:
        hit, baseline = up, uncond_up
    else:
        hit, baseline = down, uncond_down
    return {
        "side": "long" if is_long else "short",
        "quantile": float(q),
        "n": float(int(moved.sum())),
        "coverage": float(mask.mean()) if mask.size else float("nan"),
        "hit_pct": float(100.0 * hit),
        "up_pct": float(100.0 * up),
        "down_pct": float(100.0 * down),
        "uncond_up_pct": float(100.0 * uncond_up) if np.isfinite(uncond_up) else float("nan"),
        "uncond_down_pct": (
            float(100.0 * uncond_down) if np.isfinite(uncond_down) else float("nan")
        ),
        "excess_pp": float(100.0 * (hit - baseline)) if np.isfinite(baseline) else float("nan"),
        "n_dates": float(n_dates),
    }


def long_only_book_block(
    df: pd.DataFrame,
    *,
    score_col: str = "pred",
    q: float = 0.80,
    min_names: int = 3,
) -> dict[str, float]:
    """Overnight up-rate on the within-date top residual names (long-only sleeve)."""
    return sleeve_book_block(
        df, score_col=score_col, side="long", q=q, min_names=min_names
    )


def _restrict_cs_dates(df: pd.DataFrame, min_names: int) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df.groupby("date").size()
    keep = counts[counts >= int(min_names)].index
    return df[df["date"].isin(keep)].copy()


def _date_span(dates: np.ndarray) -> tuple[str, str]:
    if dates.size == 0:
        return "", ""
    cal = np.datetime64("1970-01-01") + dates.astype("timedelta64[D]")
    return str(cal.min()), str(cal.max())


def score_eval_frame(
    df: pd.DataFrame,
    *,
    min_names: int,
) -> dict[str, Any]:
    """Direction + price MAE on a labelled overnight frame (already test-only)."""
    if df.empty:
        return {"n_samples": 0, "n_dates": 0, "n_names": 0, "empty": True}
    dates = df["date"].to_numpy(dtype=np.int64)
    names = df["symbol"].astype(str).unique()
    start, end = _date_span(dates)
    pred = df["pred"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    pred_r = df["pred_r"].to_numpy(dtype=np.float64)
    r_on = df["r_on"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    nxt = df["next_open"].to_numpy(dtype=np.float64)
    implied = df["implied_open"].to_numpy(dtype=np.float64)
    implied_h = df["implied_open_given_hedge"].to_numpy(dtype=np.float64)

    # Time-series / pooled name-date: implied overnight move vs realized r_on.
    ts_hits = direction_hits(pred_r, r_on)
    resid_hits = direction_hits(pred, y)
    long_mask = np.isfinite(pred_r) & np.isfinite(r_on) & (r_on != 0.0) & (pred_r > 0.0)
    long_hits = (np.sign(pred_r[long_mask]) == np.sign(r_on[long_mask])).astype(np.float64)
    moved = np.isfinite(pred_r) & np.isfinite(r_on) & (r_on != 0.0)
    mag = np.abs(pred[moved])
    hi = mag >= (np.median(mag) if mag.size else 0.0)
    conv_hits = (
        np.sign(pred_r[moved][hi]) == np.sign(r_on[moved][hi])
    ).astype(np.float64)

    ae_usd = np.abs(implied - nxt)
    ae_pct = np.abs(implied - nxt) / np.clip(close, 1e-12, None)
    ae_usd_0 = np.abs(close - nxt)
    ae_pct_0 = np.abs(close - nxt) / np.clip(close, 1e-12, None)
    ae_usd_h = np.abs(implied_h - nxt)
    ae_pct_h = np.abs(implied_h - nxt) / np.clip(close, 1e-12, None)

    cs = cs_stats(pred, y, dates, min_names=min_names)
    cs_sign = mean_cs_sign_hit(pred, y, dates, min_names=min_names)
    finite_r = r_on[np.isfinite(r_on) & (r_on != 0.0)]
    finite_p = pred_r[np.isfinite(pred_r)]
    realized_up = float((finite_r > 0).mean()) if finite_r.size else float("nan")
    pred_up = float((finite_p > 0).mean()) if finite_p.size else float("nan")
    vs_drift = hit_rate_vs_p0(ts_hits, realized_up)
    book_top20 = long_only_book_block(df, score_col="pred", q=0.80, min_names=min_names)
    book_top30 = long_only_book_block(df, score_col="pred", q=0.70, min_names=min_names)
    book_short20 = sleeve_book_block(
        df, score_col="pred", side="short", q=0.20, min_names=min_names
    )
    book_short30 = sleeve_book_block(
        df, score_col="pred", side="short", q=0.30, min_names=min_names
    )
    down_hi = (
        np.isfinite(pred_r)
        & np.isfinite(r_on)
        & (r_on != 0.0)
        & (pred_r < 0.0)
    )
    if mag.size:
        tail_cut = float(np.quantile(np.abs(pred_r[moved]), 0.70)) if moved.any() else 0.0
    else:
        tail_cut = 0.0
    strong_down = down_hi & (np.abs(pred_r) >= tail_cut)
    down_hits = (r_on[strong_down] < 0.0).astype(np.float64) if int(strong_down.sum()) else np.zeros(0)

    return {
        "empty": False,
        "n_samples": int(len(df)),
        "n_dates": int(df["date"].nunique()),
        "n_names": int(len(names)),
        "date_start": start,
        "date_end": end,
        "direction": {
            "kind": (
                "pooled name-date time-series: sign(pred*sigma) vs "
                "sign(realized overnight log return). Not a within-date rank."
            ),
            "overall": hit_rate_inference(ts_hits),
            "residual_vs_residual": hit_rate_inference(resid_hits),
            "long_only_pred_positive": hit_rate_inference(long_hits),
            "abs_pred_above_median": hit_rate_inference(conv_hits),
            "cross_sectional_residual_sign": cs_sign,
            "realized_overnight_up_pct": float(100.0 * realized_up),
            "pred_positive_pct": float(100.0 * pred_up),
            "overall_vs_drift": vs_drift,
            "excess_pp": _as_float(vs_drift.get("excess_pp")),
            "strong_down_when_pred_negative_and_high_abs": hit_rate_inference(
                down_hits
            ),
        },
        "book": {
            "kind": (
                "CS quantile sleeves on residual pred vs realized overnight r_on. "
                "Long = top names' up-rate vs unconditional up-rate. "
                "Short = bottom names' down-rate vs unconditional down-rate. "
                "This is the ranking object, not pooled TS direction."
            ),
            "long_only_top20": book_top20,
            "long_only_top30": book_top30,
            "short_bottom20": book_short20,
            "short_bottom30": book_short30,
        },
        "cs_ic": {
            "cs_ic": float(cs.get("cs_ic", float("nan"))),
            "cs_ic_spearman": float(cs.get("cs_ic_spearman", float("nan"))),
            "cs_ic_tstat": float(cs.get("cs_ic_tstat", float("nan"))),
            "cs_n_dates": float(cs.get("cs_n_dates", 0.0)),
            "cs_mean_n": float(cs.get("cs_mean_n", float("nan"))),
        },
        "price_error": {
            "implied": (
                "next_open_hat = close_t * exp(pred * sigma); "
                "pred*sigma is the residual overnight move, not beta*hedge"
            ),
            "dollars": abs_error_block(ae_usd),
            "pct_of_prior_close": abs_error_block(ae_pct),
            "bp_of_prior_close": {
                k: (v * 1e4 if k != "n" and np.isfinite(v) else v)
                for k, v in abs_error_block(ae_pct).items()
            },
            "zero_pred_baseline_dollars": abs_error_block(ae_usd_0),
            "zero_pred_baseline_pct": abs_error_block(ae_pct_0),
            "given_realized_hedge_dollars": abs_error_block(ae_usd_h),
            "given_realized_hedge_pct": abs_error_block(ae_pct_h),
        },
    }


def fit_promoted_overnight_skip(
    bundle: dict[str, Any],
) -> tuple[np.ndarray, float, float]:
    """Frozen rank-target ridge on train last bars (PR #5 overnight skip)."""
    x, y, d = ridge_labelled_rows(
        bundle["train_symbols"],
        bundle["feature_mean"],
        bundle["feature_std"],
    )
    keep = d.astype(np.int64) >= int(
        (np.datetime64("1999-01-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    )
    if bool(keep.any()) and int(keep.sum()) >= 8:
        x, y, d = x[keep], y[keep], d[keep]
    mask = feature_mask(str(PROMOTED_SKIP["mask_mode"]))
    w, b, ic = fit_ridge_xy(
        x,
        y,
        d,
        ridge=float(PROMOTED_SKIP["ridge"]),
        min_names=int(bundle.get("cs_min_names", 30)),
        cs_demean=True,
        rank_target=bool(PROMOTED_SKIP["rank_target"]),
        feat_winsor=float(PROMOTED_SKIP["feat_winsor"]),
        feature_mask_bool=mask,
    )
    return w, b, float(ic)


def evaluate_overnight_skip(
    data_dir: str,
    universe: str = "liquid",
    *,
    log_fn: Any | None = print,
) -> dict[str, Any]:
    """Fit default overnight skip on train; score locked TEST only."""
    cfg = overnight_skip_data_config(data_dir, universe)
    if log_fn:
        log_fn(formula_log_line("overnight"))
        log_fn(
            f"recipe=promoted overnight skip {PROMOTED_SKIP}  "
            "PR#7 levers off  locked TEST only"
        )
    bundle = build_datasets(cfg, log_fn=log_fn)
    w, b, train_ic = fit_promoted_overnight_skip(bundle)
    meta = {row["symbol"]: row for row in bundle.get("meta") or []}
    px: dict[str, dict[int, tuple[float, float]]] = {}
    missing_px = 0
    for sym in bundle["test_symbols"]:
        row = meta.get(sym.symbol) or {}
        path = row.get("path") or str(
            symbol_parquet_path(cfg.data_dir, sym.symbol, interval=cfg.interval)
        )
        try:
            px[sym.symbol] = overnight_px_lookup(path, cfg)
        except (OSError, ValueError) as exc:
            missing_px += 1
            if log_fn:
                log_fn(f"skip prices {sym.symbol}: {exc}")
            px[sym.symbol] = {}
    frame = collect_eval_frame(
        bundle["test_symbols"],
        bundle["feature_mean"],
        bundle["feature_std"],
        px,
        weights=w,
        bias=b,
    )
    min_names = int(bundle.get("cs_min_names", 30))
    scored = _restrict_cs_dates(frame, min_names)
    stats = score_eval_frame(scored, min_names=min_names)
    cuts = {}
    if bundle.get("meta"):
        cuts = {
            "train_end": bundle["meta"][0].get("train_end"),
            "val_end": bundle["meta"][0].get("val_end"),
        }
    payload: dict[str, Any] = {
        "estimand": "overnight residual skip; accuracy vs realized r_on / next open",
        "formula": OVERNIGHT_FORMULA,
        "recipe": dict(PROMOTED_SKIP),
        "levers": "default overnight skip only (PR #7 flags not applied)",
        "split": "locked TEST last bars; dates with >= cs_min_names names",
        "cs_min_names": min_names,
        "n_trading_names": int(bundle.get("n_trading_names") or 0),
        "n_test_symbols_with_rows": int(scored["symbol"].nunique()) if not scored.empty else 0,
        "missing_price_files": missing_px,
        "calendar_cuts": cuts,
        "train_cs_ic": float(train_ic),
        **stats,
        "n_samples_before_cs_floor": int(len(frame)),
        "n_dates_before_cs_floor": int(frame["date"].nunique()) if not frame.empty else 0,
    }
    return payload


def format_accuracy_report(payload: dict[str, Any]) -> str:
    """Human headline for the locked-test overnight accuracy eval."""
    direction = payload.get("direction") or {}
    overall = direction.get("overall") or {}
    errors = payload.get("price_error") or {}
    price = errors.get("dollars") or {}
    pct = errors.get("pct_of_prior_close") or {}
    bp = errors.get("bp_of_prior_close") or {}
    zero_d = errors.get("zero_pred_baseline_dollars") or {}
    zero_p = errors.get("zero_pred_baseline_pct") or {}
    cs = payload.get("cs_ic") or {}
    p = overall.get("p_vs_half", float("nan"))
    z = overall.get("z_vs_half", float("nan"))
    hit = overall.get("hit_rate_pct", float("nan"))
    better = (
        "Yes, above 50% at conventional significance."
        if np.isfinite(p) and p < 0.05 and float(overall.get("hit_rate", 0.5)) > 0.5
        else (
            "No: not distinguishable from a coin flip at 5%."
            if np.isfinite(p) and p >= 0.05
            else "Inconclusive (empty or non-finite hit rate)."
        )
    )
    long_only = direction.get("long_only_pred_positive") or {}
    conv = direction.get("abs_pred_above_median") or {}
    resid = direction.get("residual_vs_residual") or {}
    cs_sign = direction.get("cross_sectional_residual_sign") or {}
    vs_drift = direction.get("overall_vs_drift") or {}
    book_top = (payload.get("book") or {}).get("long_only_top20") or {}
    book_short = (payload.get("book") or {}).get("short_bottom20") or {}
    up_pct = _as_float(direction.get("realized_overnight_up_pct"))
    excess = _as_float(vs_drift.get("excess_pp"))
    if not np.isfinite(excess) and np.isfinite(hit) and np.isfinite(up_pct):
        excess = float(hit) - float(up_pct)
    z_drift = _as_float(vs_drift.get("z_vs_p0"))
    lines = [
        "OVERNIGHT SKIP ACCURACY (locked TEST, predictive only — not live P&L)",
        f"  recipe: {payload.get('recipe')}  {payload.get('levers')}",
        f"  coverage: n={payload.get('n_samples')} name-dates, "
        f"{payload.get('n_dates')} dates, {payload.get('n_names')} names  "
        f"{payload.get('date_start')} -> {payload.get('date_end')}",
        f"  calendar: {payload.get('calendar_cuts')}",
        f"  CS IC (residual, locked test)={cs.get('cs_ic'):+.4f} "
        f"t={cs.get('cs_ic_tstat')}",
        "",
        f"HEADLINE direction accuracy = {hit:.1f}%  (vs 50% chance; "
        f"vs {up_pct:.1f}% overnight-up drift: excess {excess:+.2f} pp"
        f"{'' if not np.isfinite(z_drift) else f', z_vs_drift={z_drift:.2f}'})",
        f"  {better}  z={z:.2f}  p={p:.3g}  t={overall.get('t_vs_half')}",
        f"  this is pooled name-date TIME-SERIES direction of implied overnight "
        f"move (pred*sigma) vs realized r_on = log(open_{{t+1}})-log(close_t).",
        f"  residual-vs-residual pooled dir={resid.get('hit_rate_pct'):.1f}%  "
        f"CS demeaned residual sign={float(cs_sign.get('cs_sign_hit_pct') or float('nan')):.1f}% "
        f"(that last one is the cross-sectional object).",
        f"  long-only (pred>0): {long_only.get('hit_rate_pct'):.1f}%  "
        f"(unconditional overnight up-rate {up_pct:.1f}%; "
        f"model predicts up {direction.get('pred_positive_pct'):.1f}% of the time)  "
        f"|pred|>=median: {conv.get('hit_rate_pct'):.1f}%",
        f"  long-only BOOK top 20% CS residual: up-rate "
        f"{_as_float(book_top.get('up_pct')):.1f}%  "
        f"excess {_as_float(book_top.get('excess_pp')):+.2f} pp vs {up_pct:.1f}% "
        f"n={int(_as_float(book_top.get('n'), 0.0))}",
        f"  short BOOK bottom 20% CS residual: down-rate "
        f"{_as_float(book_short.get('down_pct', book_short.get('hit_pct'))):.1f}%  "
        f"excess {_as_float(book_short.get('excess_pp')):+.2f} pp vs "
        f"{_as_float(book_short.get('uncond_down_pct')):.1f}% overnight-down "
        f"n={int(_as_float(book_short.get('n'), 0.0))}",
        "",
        f"HEADLINE |price error|  MAE ${price.get('mae'):.4f}  "
        f"median ${price.get('median_ae'):.4f}  RMSE ${price.get('rmse'):.4f}",
        f"  vs prior close: MAE {100.0 * float(pct.get('mae') or float('nan')):.4f}% "
        f"({float(bp.get('mae') or float('nan')):.1f} bp)  "
        f"median {100.0 * float(pct.get('median_ae') or float('nan')):.4f}%",
        f"  zero-move baseline MAE ${float(zero_d.get('mae') or float('nan')):.4f} / "
        f"{100.0 * float(zero_p.get('mae') or float('nan')):.4f}%",
        "  implied next open = close_t * exp(pred * sigma); residual omits the hedge overnight.",
        "  Do not treat this as a tradable edge or live profitability.",
    ]
    promo = payload.get("promotion") or {}
    default_name = str(promo.get("accuracy_default") or "residual_sigma")
    if default_name != "residual_sigma":
        row = next(
            (
                r
                for r in ((payload.get("ablation") or {}).get("rows") or [])
                if r.get("name") == default_name
            ),
            None,
        )
        cal = payload.get("calibrate") or {}
        if row:
            t = row.get("test") or {}
            v = row.get("val") or {}
            lines.extend(
                [
                    "",
                    f"VAL-GATED ACCURACY READOUT = {default_name}  "
                    f"(CS skip unchanged; selected on locked VAL, reported on TEST)",
                    f"  TEST dir={_as_float(t.get('dir_pct')):.1f}%  "
                    f"excess {_as_float(t.get('excess_pp')):+.2f} pp vs "
                    f"{_as_float(t.get('up_pct')):.1f}% up-rate  "
                    f"MAE ${_as_float(t.get('mae_usd')):.4f} / "
                    f"{100.0 * _as_float(t.get('mae_pct')):.4f}%  "
                    f"vs residual*sigma {hit:.1f}% / "
                    f"${_as_float(price.get('mae')):.4f} / "
                    f"{100.0 * _as_float(pct.get('mae')):.4f}%  "
                    f"vs zero-move ${_as_float(zero_d.get('mae')):.4f} / "
                    f"{100.0 * _as_float(zero_p.get('mae')):.4f}%",
                    f"  VAL dir={_as_float(v.get('dir_pct')):.1f}%  "
                    f"excess {_as_float(v.get('excess_pp')):+.2f} pp  "
                    f"MAE%={100.0 * _as_float(v.get('mae_pct')):.4f}  "
                    f"kind={cal.get('kind')!r} a={cal.get('a')} b={cal.get('b')}  "
                    f"promote_dir={promo.get('direction')!r} promote_mae={promo.get('price')!r}",
                    "  Skill vs overnight drift is excess hit rate vs the unconditional "
                    "up-rate / train-median gap (always-up). CS skip is unchanged. "
                    "This is not a live P&L claim.",
                ]
            )
    ablate = payload.get("ablation")
    if ablate:
        lines.extend(["", format_ablation_table(ablate, payload.get("promotion") or {})])
        cond_txt = format_cond_dir_block(ablate, payload.get("promotion") or {})
        if cond_txt:
            lines.extend(["", cond_txt])
        dec_txt = format_decile_reliability_block(ablate, payload.get("promotion") or {})
        if dec_txt:
            lines.extend(["", dec_txt])
        csv_txt = format_cs_left_veto_block(ablate, payload.get("promotion") or {})
        if csv_txt:
            lines.extend(["", csv_txt])
        log_txt = format_logistic_up_block(ablate, payload.get("promotion") or {})
        if log_txt:
            lines.extend(["", log_txt])
    conf = payload.get("confidence")
    if conf:
        lines.extend(["", format_confidence_block(conf)])
    years = payload.get("year_slices")
    if years:
        lines.extend(["", format_year_slices(years, title="YEAR SLICES (locked TEST, residual*sigma — report only)")])
    years_r = payload.get("year_slices_readout")
    default_name = str((payload.get("promotion") or {}).get("accuracy_default") or "residual_sigma")
    if years_r and default_name != "residual_sigma":
        lines.extend(
            [
                "",
                format_year_slices(
                    years_r,
                    title=f"YEAR SLICES (locked TEST, {default_name} readout — report only)",
                ),
            ]
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Train-only overnight readout (direction + next-open MAE). Locked val gate.
# --------------------------------------------------------------------------


def apply_readout(df: pd.DataFrame, pred_r: np.ndarray) -> pd.DataFrame:
    """Replace the live overnight log-return forecast. Residual ``pred``/``y`` stay."""
    out = df.copy()
    pr = np.asarray(pred_r, dtype=np.float64)
    if pr.shape[0] != len(out):
        raise ValueError(f"pred_r length {pr.shape[0]} != frame {len(out)}")
    out["pred_r"] = pr
    close = out["close"].to_numpy(dtype=np.float64)
    out["implied_open"] = close * np.exp(pr)
    return out


def apply_affine(pred_r: np.ndarray, a: float, b: float) -> np.ndarray:
    """``r_on_hat = a * pred_r + b``. ``a=b=0`` is the zero-move baseline."""
    return float(a) * np.asarray(pred_r, dtype=np.float64) + float(b)


def fit_affine_ols(pred_r: np.ndarray, r_on: np.ndarray) -> tuple[float, float]:
    """Train-only OLS: ``r_on ~ a * pred_r + b``. No val/test rows."""
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if y.size < 8:
        return 0.0, float(np.median(y) if y.size else 0.0)
    design = np.column_stack([p, np.ones(p.size, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_affine_l1(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    *,
    n_grid: int = 41,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Train-only L1 affine: grid ``a``, ``b = median(r_on - a*pred_r)``.

    ``a=0`` recovers the train-median overnight gap (MAE-optimal constant).
    Optional ``weights`` are train-only (recency); they do not use val/test.
    """
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    if weights is not None:
        w_all = np.asarray(weights, dtype=np.float64)
        if w_all.shape[0] != p.shape[0]:
            raise ValueError("weights must align with pred_r")
        ok = ok & np.isfinite(w_all) & (w_all > 0)
        w = w_all[ok]
    else:
        w = None
    p, y = p[ok], y[ok]
    if y.size < 8:
        return 0.0, weighted_median(y, w)
    p_std = float(np.std(p))
    span = 3.0 if p_std < 1e-12 else max(2.0, 4.0 * float(np.std(y)) / max(p_std, 1e-12))
    grid = np.linspace(-span, span, max(9, int(n_grid)))
    best_a = 0.0
    best_b = weighted_median(y, w)
    if w is None:
        best = float(np.mean(np.abs(y - best_b)))
    else:
        best = float(np.average(np.abs(y - best_b), weights=w))

    def _mae(a: float) -> tuple[float, float, float]:
        resid = y - float(a) * p
        b = weighted_median(resid, w)
        if w is None:
            mae = float(np.mean(np.abs(resid - b)))
        else:
            mae = float(np.average(np.abs(resid - b), weights=w))
        return mae, float(a), b

    for a in grid:
        mae, aa, bb = _mae(float(a))
        if mae < best:
            best, best_a, best_b = mae, aa, bb
    half = (grid[1] - grid[0]) if grid.size > 1 else 0.05
    fine = np.linspace(best_a - half, best_a + half, 21)
    for a in fine:
        mae, aa, bb = _mae(float(a))
        if mae < best:
            best, best_a, best_b = mae, aa, bb
    return best_a, best_b


def train_constant(r_on: np.ndarray, how: str = "median") -> float:
    y = np.asarray(r_on, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    if how == "mean":
        return float(y.mean())
    if how == "zero":
        return 0.0
    return float(np.median(y))


def fit_affine_huber(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    *,
    n_iter: int = 25,
) -> tuple[float, float]:
    """Train-only Huber IRLS affine. Starts at OLS; no val/test rows."""
    a, b = fit_affine_ols(pred_r, r_on)
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if y.size < 8:
        return a, b
    resid0 = y - (a * p + b)
    mad = float(np.median(np.abs(resid0 - np.median(resid0))))
    delta = max(1.345 * mad / 0.6745, 1e-8) if mad > 0 else max(float(np.std(resid0)), 1e-8)
    for _ in range(int(n_iter)):
        resid = y - (a * p + b)
        absr = np.abs(resid)
        w = np.ones_like(resid)
        big = absr > delta
        w[big] = delta / np.clip(absr[big], 1e-12, None)
        sw = np.sqrt(w)
        design = np.column_stack([p * sw, sw])
        coef, *_ = np.linalg.lstsq(design, y * sw, rcond=None)
        a, b = float(coef[0]), float(coef[1])
    return a, b


def fit_piecewise_l1(
    pred_r: np.ndarray,
    r_on: np.ndarray,
) -> tuple[float, float, float, float]:
    """Separate train-only L1 affine for ``pred_r >= 0`` and ``pred_r < 0``."""
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    pos = p >= 0.0
    if int(pos.sum()) >= 8:
        a_pos, b_pos = fit_affine_l1(p[pos], y[pos])
    else:
        a_pos, b_pos = 0.0, float(np.median(y) if y.size else 0.0)
    if int((~pos).sum()) >= 8:
        a_neg, b_neg = fit_affine_l1(p[~pos], y[~pos])
    else:
        a_neg, b_neg = 0.0, float(np.median(y) if y.size else 0.0)
    return a_pos, b_pos, a_neg, b_neg


def apply_piecewise_l1(
    pred_r: np.ndarray,
    a_pos: float,
    b_pos: float,
    a_neg: float,
    b_neg: float,
) -> np.ndarray:
    p = np.asarray(pred_r, dtype=np.float64)
    out = np.empty_like(p)
    pos = p >= 0.0
    out[pos] = float(a_pos) * p[pos] + float(b_pos)
    out[~pos] = float(a_neg) * p[~pos] + float(b_neg)
    return out


def fit_bin_constants(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-only quantile bins of ``x`` → median ``y``. Edges do not use val/test."""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    n_bins = max(3, int(n_bins))
    fallback = float(np.median(yv) if yv.size else 0.0)
    if xv.size < n_bins * 4:
        edges = np.array([-1e18, 1e18], dtype=np.float64)
        return edges, np.array([fallback], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    cuts = np.quantile(xv, qs[1:-1])
    cuts = np.unique(cuts)
    edges = np.concatenate([[-1e18], cuts, [1e18]]).astype(np.float64)
    values = np.zeros(edges.size - 1, dtype=np.float64)
    for i in range(values.size):
        if i == values.size - 1:
            sel = xv >= edges[i]
        else:
            sel = (xv >= edges[i]) & (xv < edges[i + 1])
        values[i] = float(np.median(yv[sel])) if int(sel.sum()) else fallback
    return edges, values


def apply_bin_constants(
    x: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros_like(xv)
    inner = e[1:-1] if e.size >= 2 else np.zeros(0)
    if inner.size == 0:
        return np.full_like(xv, float(v[0]))
    idx = np.digitize(xv, inner, right=False)
    idx = np.clip(idx, 0, v.size - 1)
    out = v[idx]
    out = out.astype(np.float64, copy=True)
    out[~np.isfinite(xv)] = np.nan
    return out


def _bin_index(x: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    inner = e[1:-1] if e.size >= 2 else np.zeros(0)
    if inner.size == 0:
        return np.zeros(xv.shape[0], dtype=np.int64)
    idx = np.digitize(xv, inner, right=False)
    return np.clip(idx, 0, int(n_bins) - 1).astype(np.int64)


def apply_decile_reliability(
    pred_r: np.ndarray,
    edges: np.ndarray,
    keep: np.ndarray,
    b_up: float,
) -> np.ndarray:
    """Residual*sigma in TRAIN-reliable bins; train-median always-up otherwise."""
    p = np.asarray(pred_r, dtype=np.float64)
    k = np.asarray(keep, dtype=np.bool_)
    idx = _bin_index(p, edges, k.size)
    reliable = k[idx]
    out = np.full(p.shape, float(b_up), dtype=np.float64)
    out[reliable] = p[reliable]
    out[~np.isfinite(p)] = np.nan
    return out


def fit_decile_reliability(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    *,
    n_bins_grid: Sequence[int] = (5, 8, 10),
    min_n: int = 24,
) -> dict[str, Any]:
    """TRAIN-only: keep residual sign only in pred_r bins that beat always-up.

    A bin is reliable if residual-sign hit rate ≥ TRAIN overnight-up + DIR_LIFT
    and it has at least ``min_n`` rows. Unreliable bins become the train-median
    gap (always-up). ``n_bins`` is chosen on TRAIN direction excess, then MAE.
    """
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    b_up = float(np.median(y) if y.size else 0.0)
    moved = y != 0.0
    up = float((y[moved] > 0).mean()) if int(moved.sum()) else 0.5
    floor = up + float(DIR_LIFT)
    fallback = {
        "n_bins": 10,
        "edges": [-1e18, 1e18],
        "keep": [True],
        "hit": [float("nan")],
        "b_up": b_up,
        "train_up": up,
        "floor": floor,
        "n_keep": 1,
    }
    if p.size < 80:
        return fallback
    best = dict(fallback)
    best_key = (-1e9, 1e9)
    for n_bins in n_bins_grid:
        n_bins = max(3, int(n_bins))
        if p.size < n_bins * min_n:
            continue
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        cuts = np.unique(np.quantile(p, qs[1:-1]))
        edges = np.concatenate([[-1e18], cuts, [1e18]]).astype(np.float64)
        n_eff = int(edges.size - 1)
        keep = np.zeros(n_eff, dtype=np.bool_)
        hit = np.full(n_eff, np.nan, dtype=np.float64)
        idx = _bin_index(p, edges, n_eff)
        for i in range(n_eff):
            sel = idx == i
            n_i = int(sel.sum())
            if n_i < int(min_n):
                continue
            hits = direction_hits(p[sel], y[sel])
            if hits.size == 0:
                continue
            hr = float(hits.mean())
            hit[i] = hr
            keep[i] = hr >= floor
        hat = apply_decile_reliability(p, edges, keep, b_up)
        excess = _direction_excess(hat, y)
        mae = float(np.mean(np.abs(hat - y)))
        if not np.isfinite(excess):
            continue
        key = (excess, -mae)
        if key > best_key:
            best_key = key
            best = {
                "n_bins": n_eff,
                "edges": edges.tolist(),
                "keep": [bool(v) for v in keep],
                "hit": [float(v) if np.isfinite(v) else float("nan") for v in hit],
                "b_up": b_up,
                "train_up": up,
                "floor": floor,
                "n_keep": int(keep.sum()),
            }
    return best


def fit_group_median(
    values: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[int, float], float]:
    y = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups, dtype=np.int64)
    default = float(np.median(y[np.isfinite(y)])) if y.size else 0.0
    table: dict[int, float] = {}
    for key in np.unique(g):
        sel = g == int(key)
        yy = y[sel]
        yy = yy[np.isfinite(yy)]
        table[int(key)] = float(np.median(yy)) if yy.size else default
    return table, default


def apply_group_median(
    groups: np.ndarray,
    table: dict[Any, float],
    default: float,
) -> np.ndarray:
    g = np.asarray(groups, dtype=np.int64)
    out = np.full(g.shape[0], float(default), dtype=np.float64)
    for key, val in table.items():
        out[g == int(key)] = float(val)
    return out


def _direction_excess(pred_r: np.ndarray, r_on: np.ndarray) -> float:
    hits = direction_hits(pred_r, r_on)
    r = np.asarray(r_on, dtype=np.float64)
    moved = np.isfinite(r) & (r != 0.0)
    up = float((r[moved] > 0).mean()) if int(moved.sum()) else float("nan")
    if hits.size == 0 or not np.isfinite(up):
        return float("nan")
    return float(hits.mean() - up)


def fit_drift_veto(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    *,
    quantiles: Sequence[float] = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30),
) -> dict[str, float]:
    """Always-up except a train-chosen left tail of residual*sigma.

    Threshold and region constants are train-only. Predicting down on the tail
    beats the overnight-up base rate iff that tail is more than 50% down.
    """
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    b_up = float(np.median(y) if y.size else 0.0)
    fallback = {
        "tau": float(np.quantile(p, 0.10) if p.size else 0.0),
        "a_dn": 0.0,
        "b_dn": b_up,
        "b_up": b_up,
        "q": 0.10,
        "mode": 0.0,
    }
    if p.size < 32:
        return fallback
    best_spec = dict(fallback)
    best_key = (-1e9, 1e9)
    for q in quantiles:
        tau = float(np.quantile(p, float(q)))
        down = p <= tau
        n_dn = int(down.sum())
        n_up = int((~down).sum())
        if n_dn < 24 or n_up < 24:
            continue
        cover = n_dn / float(p.size)
        if cover < 0.02 or cover > 0.45:
            continue
        b_dn_c = float(np.median(y[down]))
        b_up_c = float(np.median(y[~down]))
        candidates: list[tuple[str, np.ndarray, dict[str, float]]] = [
            (
                "const",
                np.where(down, b_dn_c, b_up_c),
                {"tau": tau, "a_dn": 0.0, "b_dn": b_dn_c, "b_up": b_up_c, "q": float(q), "mode": 0.0},
            )
        ]
        if n_dn >= 32:
            a_dn, b_dn_a = fit_affine_l1(p[down], y[down])
            hat_a = np.where(down, apply_affine(p, a_dn, b_dn_a), b_up_c)
            candidates.append(
                (
                    "affine",
                    hat_a,
                    {
                        "tau": tau,
                        "a_dn": float(a_dn),
                        "b_dn": float(b_dn_a),
                        "b_up": b_up_c,
                        "q": float(q),
                        "mode": 1.0,
                    },
                )
            )
        for _name, hat, spec in candidates:
            if float(np.mean(hat < 0.0)) < 0.01:
                continue
            excess = _direction_excess(hat, y)
            mae = float(np.mean(np.abs(hat - y)))
            if not np.isfinite(excess):
                continue
            key = (excess, -mae)
            if key > best_key:
                best_key = key
                best_spec = spec
    return best_spec


def apply_drift_veto(
    pred_r: np.ndarray,
    tau: float,
    a_dn: float,
    b_dn: float,
    b_up: float,
) -> np.ndarray:
    p = np.asarray(pred_r, dtype=np.float64)
    down = p <= float(tau)
    out = np.full(p.shape, float(b_up), dtype=np.float64)
    out[down] = float(a_dn) * p[down] + float(b_dn)
    return out


CS_LEFT_QS = (0.10, 0.15, 0.20, 0.30)
CS_LEFT_TAUS = (0.05, 0.10, 0.15, 0.20, 0.30)


def cs_bottom_mask(
    scores: np.ndarray,
    dates: np.ndarray,
    q: float,
) -> np.ndarray:
    """Within-date bottom quantile of ``scores``. Causal (no labels)."""
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    d = pd.Series(np.asarray(dates))
    pct = s.groupby(d, sort=False).rank(pct=True, method="average")
    return pct.to_numpy(dtype=np.float64) <= float(q)


def apply_cs_left_veto(
    pred_r: np.ndarray,
    cs_score: np.ndarray,
    dates: np.ndarray,
    tau: float,
    q: float,
    a_dn: float,
    b_dn: float,
    b_up: float,
) -> np.ndarray:
    """Always-up except the intersection of CS-bottom and TS left tail."""
    p = np.asarray(pred_r, dtype=np.float64)
    bottom = cs_bottom_mask(cs_score, dates, q)
    down = bottom & (p <= float(tau))
    out = np.full(p.shape, float(b_up), dtype=np.float64)
    out[down] = float(a_dn) * p[down] + float(b_dn)
    return out


def fit_cs_left_veto(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    cs_score: np.ndarray,
    dates: np.ndarray,
    *,
    qs: Sequence[float] = CS_LEFT_QS,
    tau_qs: Sequence[float] = CS_LEFT_TAUS,
) -> dict[str, float]:
    """TRAIN-only CS-bottom ∩ TS-left veto. No VAL/TEST rows."""
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    c = np.asarray(cs_score, dtype=np.float64)
    d = np.asarray(dates)
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(c)
    p, y, c, d = p[ok], y[ok], c[ok], d[ok]
    b_up = float(np.median(y) if y.size else 0.0)
    fallback = {
        "tau": float(np.quantile(p, 0.15) if p.size else 0.0),
        "q": 0.20,
        "a_dn": 0.0,
        "b_dn": b_up,
        "b_up": b_up,
        "tau_q": 0.15,
    }
    if p.size < 64:
        return fallback
    best = dict(fallback)
    best_key = (-1e9, 1e9)
    for q in qs:
        bottom = cs_bottom_mask(c, d, float(q))
        if int(bottom.sum()) < 24:
            continue
        for tq in tau_qs:
            tau = float(np.quantile(p, float(tq)))
            down = bottom & (p <= tau)
            n_dn = int(down.sum())
            if n_dn < 16 or n_dn > 0.45 * p.size:
                continue
            b_dn_c = float(np.median(y[down]))
            candidates = [
                (0.0, b_dn_c),
            ]
            if n_dn >= 32:
                a_dn, b_dn_a = fit_affine_l1(p[down], y[down])
                candidates.append((float(a_dn), float(b_dn_a)))
            for a_dn, b_dn in candidates:
                hat = apply_cs_left_veto(p, c, d, tau, float(q), a_dn, b_dn, b_up)
                if float(np.mean(hat < 0.0)) < 0.01:
                    continue
                excess = _direction_excess(hat, y)
                mae = float(np.mean(np.abs(hat - y)))
                if not np.isfinite(excess):
                    continue
                key = (excess, -mae)
                if key > best_key:
                    best_key = key
                    best = {
                        "tau": tau,
                        "q": float(q),
                        "a_dn": float(a_dn),
                        "b_dn": float(b_dn),
                        "b_up": b_up,
                        "tau_q": float(tq),
                    }
    return best


LOGISTIC_TAUS = (0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zc))


def fit_logistic_up(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    *,
    n_iter: int = 20,
    taus: Sequence[float] = LOGISTIC_TAUS,
) -> dict[str, float]:
    """TRAIN-only logistic P(up | pred_r) with a direction threshold τ.

    ``hat = +mag`` if P≥τ else ``-mag``. ``mag`` is the train mean |r_on|.
    ``(a,b,τ)`` never see VAL/TEST.
    """
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    mag = float(np.mean(np.abs(y))) if y.size else 0.0
    fallback = {"a": 0.0, "b": 0.0, "tau": 0.50, "mag": mag}
    if p.size < 64:
        return fallback
    yb = (y > 0.0).astype(np.float64)
    a, b = 0.0, float(np.log(max(yb.mean(), 1e-3) / max(1.0 - yb.mean(), 1e-3)))
    ones = np.ones_like(p)
    for _ in range(int(n_iter)):
        pr = _sigmoid(a * p + b)
        w = np.clip(pr * (1.0 - pr), 1e-6, None)
        z = (a * p + b) + (yb - pr) / w
        sw = np.sqrt(w)
        design = np.column_stack([p * sw, ones * sw])
        coef, *_ = np.linalg.lstsq(design, z * sw, rcond=None)
        a, b = float(coef[0]), float(coef[1])
    best = dict(fallback)
    best.update({"a": a, "b": b})
    best_key = (-1e9, 1e9)
    for tau in taus:
        hat = apply_logistic_up(p, a, b, float(tau), mag)
        excess = _direction_excess(hat, y)
        mae = float(np.mean(np.abs(hat - y)))
        if not np.isfinite(excess):
            continue
        key = (excess, -mae)
        if key > best_key:
            best_key = key
            best = {"a": a, "b": b, "tau": float(tau), "mag": mag}
    return best


def apply_logistic_up(
    pred_r: np.ndarray,
    a: float,
    b: float,
    tau: float,
    mag: float,
) -> np.ndarray:
    p = np.asarray(pred_r, dtype=np.float64)
    pr = _sigmoid(float(a) * p + float(b))
    sign = np.where(pr >= float(tau), 1.0, -1.0)
    return sign * float(mag)


def fit_left_tail_l1(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    *,
    quantiles: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50),
) -> dict[str, float]:
    """``r_hat = b_up + a_neg * min(pred_r - tau, 0)``. Grid tau on train L1."""
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    b_up = float(np.median(y) if y.size else 0.0)
    best = {"tau": 0.0, "a_neg": 0.0, "b_up": b_up}
    if p.size < 16:
        return best
    best_mae = float(np.mean(np.abs(y - b_up)))
    for q in quantiles:
        tau = float(np.quantile(p, float(q)))
        z = np.minimum(p - tau, 0.0)
        if float(np.std(z)) < 1e-15:
            continue
        a_neg, b_extra = fit_affine_l1(z, y - b_up)
        hat = b_up + apply_affine(z, a_neg, b_extra)
        mae = float(np.mean(np.abs(hat - y)))
        if mae < best_mae:
            best_mae = mae
            best = {
                "tau": tau,
                "a_neg": float(a_neg),
                "b_up": float(b_up + b_extra),
            }
    return best


def apply_left_tail_l1(
    pred_r: np.ndarray,
    tau: float,
    a_neg: float,
    b_up: float,
) -> np.ndarray:
    p = np.asarray(pred_r, dtype=np.float64)
    z = np.minimum(p - float(tau), 0.0)
    return float(b_up) + float(a_neg) * z


def fit_confidence_blend(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    a: float,
    b: float,
    b_up: float,
    *,
    abs_quantiles: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90),
) -> dict[str, float]:
    """Use affine when |pred_r| is large, else the train-median drift."""
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    mag = np.abs(p)
    best = {"tau": float(np.quantile(mag, 0.70) if mag.size else 0.0), "a": float(a), "b": float(b), "b_up": float(b_up)}
    if p.size < 16:
        return best
    best_key = (-1e9, 1e9)
    for q in abs_quantiles:
        tau = float(np.quantile(mag, float(q)))
        hat = np.where(mag >= tau, apply_affine(p, a, b), b_up)
        excess = _direction_excess(hat, y)
        mae = float(np.mean(np.abs(hat - y)))
        if not np.isfinite(excess):
            continue
        key = (excess, -mae)
        if key > best_key:
            best_key = key
            best = {"tau": tau, "a": float(a), "b": float(b), "b_up": float(b_up), "q": float(q)}
    return best


def apply_confidence_blend(
    pred_r: np.ndarray,
    tau: float,
    a: float,
    b: float,
    b_up: float,
) -> np.ndarray:
    p = np.asarray(pred_r, dtype=np.float64)
    affine = apply_affine(p, a, b)
    return np.where(np.abs(p) >= float(tau), affine, float(b_up))


COND_DIR_ABS_QS = (0.50, 0.60, 0.70, 0.80, 0.90)
COND_DIR_LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)


def apply_cond_dir_blend(
    pred_r: np.ndarray,
    *,
    tau_abs: float,
    lam: float,
    left_tau: float,
    left_a_neg: float,
    left_b_up: float,
    conf_tau: float,
    conf_a: float,
    conf_b: float,
    conf_b_up: float,
    b_up: float,
) -> np.ndarray:
    """High-|pred| mix of left-tail L1 and confidence affine; else train-median.

    Low-|pred| stays the MAE-optimal always-up constant. Next open is never used.
    """
    p = np.asarray(pred_r, dtype=np.float64)
    left = apply_left_tail_l1(p, left_tau, left_a_neg, left_b_up)
    conf = apply_confidence_blend(p, conf_tau, conf_a, conf_b, conf_b_up)
    w = float(np.clip(lam, 0.0, 1.0))
    mix = w * left + (1.0 - w) * conf
    return np.where(np.abs(p) >= float(tau_abs), mix, float(b_up))


def fit_cond_dir_blend(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    left: Mapping[str, Any],
    blend: Mapping[str, Any],
    b_up: float,
    *,
    abs_quantiles: Sequence[float] = COND_DIR_ABS_QS,
    lambdas: Sequence[float] = COND_DIR_LAMBDAS,
) -> dict[str, float]:
    """TRAIN-only (q, λ) for the conditional left-tail / confidence mix.

    Picks the pair with the best direction excess vs always-up, then MAE.
    Thresholds and λ never see VAL/TEST.
    """
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    mag = np.abs(p)
    fallback = {
        "tau_abs": float(np.quantile(mag, 0.70) if mag.size else 0.0),
        "q": 0.70,
        "lam": 0.50,
        "left_tau": float(left.get("tau") or 0.0),
        "left_a_neg": float(left.get("a_neg") or 0.0),
        "left_b_up": float(left.get("b_up") or b_up),
        "conf_tau": float(blend.get("tau") or 0.0),
        "conf_a": float(blend.get("a") or 0.0),
        "conf_b": float(blend.get("b") or 0.0),
        "conf_b_up": float(blend.get("b_up") or b_up),
        "b_up": float(b_up),
    }
    if p.size < 32:
        return fallback
    best = dict(fallback)
    best_key = (-1e9, 1e9)
    for q in abs_quantiles:
        tau_abs = float(np.quantile(mag, float(q)))
        for lam in lambdas:
            hat = apply_cond_dir_blend(
                p,
                tau_abs=tau_abs,
                lam=float(lam),
                left_tau=fallback["left_tau"],
                left_a_neg=fallback["left_a_neg"],
                left_b_up=fallback["left_b_up"],
                conf_tau=fallback["conf_tau"],
                conf_a=fallback["conf_a"],
                conf_b=fallback["conf_b"],
                conf_b_up=fallback["conf_b_up"],
                b_up=float(b_up),
            )
            excess = _direction_excess(hat, y)
            mae = float(np.mean(np.abs(hat - y)))
            if not np.isfinite(excess):
                continue
            key = (excess, -mae)
            if key > best_key:
                best_key = key
                best = {
                    **fallback,
                    "tau_abs": tau_abs,
                    "q": float(q),
                    "lam": float(lam),
                }
    return best


def cond_abs_mask(pred_r: np.ndarray, tau_abs: float) -> np.ndarray:
    p = np.asarray(pred_r, dtype=np.float64)
    return np.isfinite(p) & (np.abs(p) >= float(tau_abs))


def _table_to_jsonable(table: dict[int, float]) -> dict[str, float]:
    return {str(int(k)): float(v) for k, v in table.items()}


def _table_from_spec(table: Any) -> dict[int, float]:
    if not table:
        return {}
    out: dict[int, float] = {}
    if isinstance(table, dict):
        for k, v in table.items():
            try:
                out[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def apply_calibrate_spec(
    pred_r: np.ndarray,
    spec: Mapping[str, Any] | None,
    *,
    dates: np.ndarray | None = None,
    vol_level: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a VAL-gated overnight readout spec to ``pred*sigma``.

    Affine ``{a,b}`` stays the PR #9 generate.py contract. Other kinds are
    train-only maps (veto, bins, weekday, piecewise) stored in calibrate JSON.
    """
    p = np.asarray(pred_r, dtype=np.float64)
    if not spec:
        return p
    params = dict(spec)
    nested = params.get("params")
    if isinstance(nested, dict):
        for k, v in nested.items():
            params.setdefault(k, v)
    kind = str(params.get("kind") or params.get("name") or "").strip().lower()
    if kind in ("", "affine", "affine_l1", "affine_ols", "huber_affine", "residual_sigma"):
        if params.get("a") is None and params.get("b") is None and kind != "residual_sigma":
            return p
        a = 1.0 if params.get("a") is None else float(params["a"])
        b = 0.0 if params.get("b") is None else float(params["b"])
        if kind == "residual_sigma":
            a, b = 1.0, 0.0
        return apply_affine(p, a, b)
    if kind in ("zero_move",):
        return np.zeros_like(p)
    if kind in ("train_median_gap", "train_mean_gap", "recency_median"):
        b = float(params.get("b") or params.get("default") or 0.0)
        return np.full_like(p, b)
    if kind in ("piecewise_l1", "piecewise"):
        return apply_piecewise_l1(
            p,
            float(params.get("a_pos") or 0.0),
            float(params.get("b_pos") or 0.0),
            float(params.get("a_neg") or 0.0),
            float(params.get("b_neg") or 0.0),
        )
    if kind in ("bin_calibrate", "vol_regime_gap"):
        src = vol_level if kind == "vol_regime_gap" and vol_level is not None else p
        edges = np.asarray(params.get("edges") or [-np.inf, np.inf], dtype=np.float64)
        values = np.asarray(params.get("values") or [0.0], dtype=np.float64)
        return apply_bin_constants(np.asarray(src, dtype=np.float64), edges, values)
    if kind in ("drift_veto",):
        return apply_drift_veto(
            p,
            float(params.get("tau") or 0.0),
            float(params.get("a_dn") or 0.0),
            float(params.get("b_dn") or 0.0),
            float(params.get("b_up") or 0.0),
        )
    if kind in ("cs_left_veto",):
        if dates is None:
            d = np.arange(p.shape[0], dtype=np.int64)
        else:
            d = np.asarray(dates)
        return apply_cs_left_veto(
            p,
            p,
            d,
            float(params.get("tau") or 0.0),
            float(params.get("q") or 0.20),
            float(params.get("a_dn") or 0.0),
            float(params.get("b_dn") or 0.0),
            float(params.get("b_up") or 0.0),
        )
    if kind in ("logistic_up",):
        return apply_logistic_up(
            p,
            float(params.get("a") or 0.0),
            float(params.get("b") or 0.0),
            float(params.get("tau") or 0.50),
            float(params.get("mag") or 0.0),
        )
    if kind in ("left_tail_l1",):
        return apply_left_tail_l1(
            p,
            float(params.get("tau") or 0.0),
            float(params.get("a_neg") or 0.0),
            float(params.get("b_up") or 0.0),
        )
    if kind in ("confidence_blend",):
        return apply_confidence_blend(
            p,
            float(params.get("tau") or 0.0),
            float(params.get("a") or 0.0),
            float(params.get("b") or 0.0),
            float(params.get("b_up") or 0.0),
        )
    if kind in ("cond_dir_blend",):
        return apply_cond_dir_blend(
            p,
            tau_abs=float(params.get("tau_abs") or 0.0),
            lam=float(params.get("lam") or 0.0),
            left_tau=float(params.get("left_tau") or 0.0),
            left_a_neg=float(params.get("left_a_neg") or 0.0),
            left_b_up=float(params.get("left_b_up") or 0.0),
            conf_tau=float(params.get("conf_tau") or 0.0),
            conf_a=float(params.get("conf_a") or 0.0),
            conf_b=float(params.get("conf_b") or 0.0),
            conf_b_up=float(params.get("conf_b_up") or 0.0),
            b_up=float(params.get("b_up") or 0.0),
        )
    if kind in ("decile_reliability",):
        edges = np.asarray(params.get("edges") or [-np.inf, np.inf], dtype=np.float64)
        keep_raw = params.get("keep") or [True]
        keep = np.asarray([bool(v) for v in keep_raw], dtype=np.bool_)
        return apply_decile_reliability(
            p, edges, keep, float(params.get("b_up") or 0.0)
        )
    if kind in ("dow_gap", "dow_plus_residual"):
        if dates is None:
            b = float(params.get("default") or params.get("b") or 0.0)
            dow = np.full_like(p, b)
        else:
            table = _table_from_spec(params.get("by_dow") or params.get("table"))
            default = float(params.get("default") or 0.0)
            dow = apply_group_median(weekday_of_dates(dates), table, default)
        if kind == "dow_gap":
            return dow
        a = 0.0 if params.get("a") is None else float(params["a"])
        b = 0.0 if params.get("b") is None else float(params["b"])
        return dow + apply_affine(p, a, b)
    if params.get("a") is not None or params.get("b") is not None:
        a = 1.0 if params.get("a") is None else float(params["a"])
        b = 0.0 if params.get("b") is None else float(params["b"])
        return apply_affine(p, a, b)
    return p


def _restrict_aligned(
    df: pd.DataFrame,
    x: np.ndarray | None,
    min_names: int,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    if df.empty:
        return df, x
    counts = df.groupby("date").size()
    keep_dates = set(int(k) for k in counts[counts >= int(min_names)].index)
    mask = df["date"].astype(np.int64).isin(keep_dates).to_numpy()
    out = df.loc[mask].reset_index(drop=True)
    if x is None:
        return out, x
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[0] != len(df):
        raise ValueError("feature rows must align with the eval frame")
    return out, arr[mask]


def adv_sleeve_mask(df: pd.DataFrame, *, pctile: float = 0.67) -> np.ndarray:
    """Within-date high ``turnover_z`` names. Feature known at close t."""
    if df.empty or "turnover_z" not in df.columns:
        return np.ones(len(df), dtype=bool)
    z = df["turnover_z"].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy(dtype=np.int64)
    out = np.zeros(len(df), dtype=bool)
    p = float(pctile)
    for key in np.unique(dates):
        sel = dates == key
        row = z[sel]
        finite = np.isfinite(row)
        if int(finite.sum()) < 3:
            out[sel] = finite
            continue
        cut = float(np.nanpercentile(row[finite], 100.0 * p))
        out[sel] = finite & (row >= cut)
    return out


def confidence_mask(pred_r: np.ndarray, threshold: float) -> np.ndarray:
    p = np.asarray(pred_r, dtype=np.float64)
    return np.isfinite(p) & (np.abs(p) >= float(threshold))


def fit_ts_overnight_ridge(
    x: np.ndarray,
    r_on: np.ndarray,
    dates: np.ndarray,
    *,
    mask_mode: str = "all",
    ridge: float = 10.0,
    feat_winsor: float = 3.0,
    min_names: int = 8,
) -> tuple[np.ndarray, float, float]:
    """Frozen ridge of *raw* overnight return on last-bar features (train only).

    Not CS-demeaned: intercept may capture overnight drift. Next open is the
    label. ``mask_mode='all'`` keeps calendar / long-TS that the CS skip drops.
    """
    return fit_ridge_xy(
        np.asarray(x, dtype=np.float64),
        np.asarray(r_on, dtype=np.float64),
        np.asarray(dates, dtype=np.int64),
        ridge=float(ridge),
        min_names=int(min_names),
        cs_demean=False,
        rank_target=False,
        feat_winsor=float(feat_winsor),
        feature_mask_bool=feature_mask(mask_mode),
    )


def fit_sign_overnight_ridge(
    x: np.ndarray,
    r_on: np.ndarray,
    dates: np.ndarray,
    *,
    mask_mode: str = "all",
    ridge: float = 10.0,
    feat_winsor: float = 3.0,
    min_names: int = 8,
) -> tuple[np.ndarray, float, float]:
    """Ridge on ``sign(r_on)``. Direction object, not residual CS ranks."""
    y = np.sign(np.asarray(r_on, dtype=np.float64))
    return fit_ridge_xy(
        np.asarray(x, dtype=np.float64),
        y,
        np.asarray(dates, dtype=np.int64),
        ridge=float(ridge),
        min_names=int(min_names),
        cs_demean=False,
        rank_target=False,
        feat_winsor=float(feat_winsor),
        feature_mask_bool=feature_mask(mask_mode),
    )


def fit_date_level_overnight(
    x: np.ndarray,
    r_on: np.ndarray,
    dates: np.ndarray,
    *,
    ridge: float = 10.0,
) -> tuple[np.ndarray, float]:
    """Date-mean features → date-mean overnight (common/hedge component)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    d = np.asarray(dates, dtype=np.int64)
    keys = np.unique(d)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ds: list[int] = []
    for key in keys:
        sel = d == key
        if int(sel.sum()) < 1:
            continue
        xs.append(np.mean(x[sel], axis=0))
        ys.append(float(np.mean(y[sel])))
        ds.append(int(key))
    if not xs:
        return np.zeros(x.shape[1], dtype=np.float32), 0.0
    xd = np.stack(xs, axis=0)
    yd = np.asarray(ys, dtype=np.float64)
    dd = np.asarray(ds, dtype=np.int64)
    w, b, _ic = fit_ridge_xy(
        xd,
        yd,
        dd,
        ridge=float(ridge),
        min_names=1,
        cs_demean=False,
        rank_target=False,
        feat_winsor=0.0,
    )
    return w, b


def predict_date_level(
    x: np.ndarray,
    dates: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(dates, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    out = np.zeros(x.shape[0], dtype=np.float64)
    for key in np.unique(d):
        sel = d == key
        mean_x = np.mean(x[sel], axis=0)
        out[sel] = float(mean_x @ w + float(bias))
    return out


def predict_linear(
    x: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + float(
        bias
    )


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def slim_accuracy(stats: dict[str, Any]) -> dict[str, float]:
    if stats.get("empty"):
        return {
            "n": 0.0,
            "dir_pct": float("nan"),
            "dir_z": float("nan"),
            "mae_usd": float("nan"),
            "mae_pct": float("nan"),
            "zero_mae_pct": float("nan"),
            "cs_ic": float("nan"),
            "long_only_pct": float("nan"),
            "up_pct": float("nan"),
            "excess_pp": float("nan"),
            "z_vs_drift": float("nan"),
            "lo_top20_up_pct": float("nan"),
            "lo_top20_excess_pp": float("nan"),
        }
    direction = stats.get("direction") or {}
    overall = direction.get("overall") or {}
    pe = stats.get("price_error") or {}
    cs = stats.get("cs_ic") or {}
    long_only = direction.get("long_only_pred_positive") or {}
    vs_drift = direction.get("overall_vs_drift") or {}
    book = (stats.get("book") or {}).get("long_only_top20") or {}
    dir_pct = _as_float(overall.get("hit_rate_pct"))
    up_pct = _as_float(direction.get("realized_overnight_up_pct"))
    excess = _as_float(vs_drift.get("excess_pp"))
    if not np.isfinite(excess) and np.isfinite(dir_pct) and np.isfinite(up_pct):
        excess = dir_pct - up_pct
    return {
        "n": _as_float(stats.get("n_samples"), 0.0),
        "dir_pct": dir_pct,
        "dir_z": _as_float(overall.get("z_vs_half")),
        "mae_usd": _as_float((pe.get("dollars") or {}).get("mae")),
        "mae_pct": _as_float((pe.get("pct_of_prior_close") or {}).get("mae")),
        "zero_mae_pct": _as_float((pe.get("zero_pred_baseline_pct") or {}).get("mae")),
        "cs_ic": _as_float(cs.get("cs_ic")),
        "long_only_pct": _as_float(long_only.get("hit_rate_pct")),
        "up_pct": up_pct,
        "excess_pp": excess,
        "z_vs_drift": _as_float(vs_drift.get("z_vs_p0")),
        "lo_top20_up_pct": _as_float(book.get("up_pct")),
        "lo_top20_excess_pp": _as_float(book.get("excess_pp")),
    }


def _score_pred_r(df: pd.DataFrame, pred_r: np.ndarray, min_names: int) -> dict[str, Any]:
    return score_eval_frame(apply_readout(df, pred_r), min_names=min_names)


def load_split_px(
    bundle: dict[str, Any],
    cfg: DataConfig,
    *,
    log_fn: Any | None = None,
) -> tuple[dict[str, dict[int, tuple[float, float]]], int]:
    """Price lookup for every name that appears in train/val/test."""
    meta = {row["symbol"]: row for row in bundle.get("meta") or []}
    names: set[str] = set()
    for split in ("train", "val", "test"):
        for sym in bundle.get(f"{split}_symbols") or []:
            names.add(sym.symbol)
    px: dict[str, dict[int, tuple[float, float]]] = {}
    missing = 0
    for name in sorted(names):
        row = meta.get(name) or {}
        path = row.get("path") or str(
            symbol_parquet_path(cfg.data_dir, name, interval=cfg.interval)
        )
        try:
            px[name] = overnight_px_lookup(path, cfg)
        except (OSError, ValueError) as exc:
            missing += 1
            if log_fn:
                log_fn(f"skip prices {name}: {exc}")
            px[name] = {}
    return px, missing


def _frame_for_split(
    bundle: dict[str, Any],
    split: str,
    px: dict[str, dict[int, tuple[float, float]]],
    weights: np.ndarray,
    bias: float,
    min_names: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    df, x = collect_eval_frame(
        bundle[f"{split}_symbols"],
        bundle["feature_mean"],
        bundle["feature_std"],
        px,
        weights=weights,
        bias=bias,
        return_features=True,
    )
    return _restrict_aligned(df, x, min_names)


def _variant_row(
    name: str,
    *,
    kind: str,
    fit_split: str,
    val_stats: dict[str, Any],
    test_stats: dict[str, Any],
    params: dict[str, Any],
    baseline_val: dict[str, float],
    median_val: dict[str, float],
    zero_val_mae: float,
) -> dict[str, Any]:
    v = slim_accuracy(val_stats)
    t = slim_accuracy(test_stats)
    is_baseline_family = kind in ("baseline", "constant", "filter")
    better_dir = (not is_baseline_family) and np.isfinite(v["dir_pct"]) and np.isfinite(
        baseline_val["dir_pct"]
    ) and (v["dir_pct"] / 100.0 >= baseline_val["dir_pct"] / 100.0 + DIR_LIFT)
    beats_drift = (
        np.isfinite(v["dir_pct"])
        and np.isfinite(median_val["dir_pct"])
        and v["dir_pct"] >= median_val["dir_pct"] + 100.0 * DIR_LIFT
    )
    better_mae = (
        (not is_baseline_family)
        and np.isfinite(v["mae_pct"])
        and np.isfinite(baseline_val["mae_pct"])
        and np.isfinite(zero_val_mae)
        and np.isfinite(median_val["mae_pct"])
        and v["mae_pct"] < min(baseline_val["mae_pct"], zero_val_mae, median_val["mae_pct"]) - 1e-12
    )
    return {
        "name": name,
        "kind": kind,
        "fit": fit_split,
        "params": params,
        "val": v,
        "test": t,
        "promote_dir": bool(better_dir and beats_drift),
        "promote_mae": bool(better_mae),
        "promote": False,  # filled after CS-IC check / selection
    }


def _pick_promoted(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Locked-VAL selection. TEST numbers are never used."""
    price = [r for r in rows if r.get("promote_mae")]
    direction = [r for r in rows if r.get("promote_dir")]
    promoted_price = None
    promoted_dir = None
    if price:
        promoted_price = min(
            price,
            key=lambda r: (
                r["val"]["mae_pct"] if np.isfinite(r["val"]["mae_pct"]) else 1e9
            ),
        )
    if direction:
        promoted_dir = max(
            direction,
            key=lambda r: (
                r["val"]["dir_pct"] if np.isfinite(r["val"]["dir_pct"]) else -1.0
            ),
        )
    default_readout = "residual_sigma"
    if promoted_price is not None:
        default_readout = str(promoted_price["name"])
        promoted_price["promote"] = True
    elif promoted_dir is not None:
        default_readout = str(promoted_dir["name"])
        promoted_dir["promote"] = True
    return {
        "price": None if promoted_price is None else promoted_price["name"],
        "direction": None if promoted_dir is None else promoted_dir["name"],
        "accuracy_default": default_readout,
        "note": (
            "CS skip stays the ranking book. Accuracy default is a readout "
            "selected on locked VAL (direction lift vs residual*sigma AND vs "
            "train-median overnight-up drift, or % MAE below residual/zero/median). "
            "TEST is report-only. Excess hit rate vs the unconditional up-rate is "
            "the direction skill object — always-up is ~54.3% on liquid TEST."
        ),
        "cs_skip_unchanged": True,
        "val_dir_lift": DIR_LIFT,
        "val_cs_ic_lift": VAL_LIFT,
        "val_2017_keep": VAL_2017_KEEP,
    }


def _year_direction(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    years = dates_to_year(df["date"].to_numpy(dtype=np.int64))
    rows: list[dict[str, Any]] = []
    for year in sorted(set(int(v) for v in years)):
        sel = years == year
        sub = df.loc[sel]
        hits = direction_hits(
            sub["pred_r"].to_numpy(dtype=np.float64),
            sub["r_on"].to_numpy(dtype=np.float64),
        )
        inf = hit_rate_inference(hits)
        r = sub["r_on"].to_numpy(dtype=np.float64)
        moved = np.isfinite(r) & (r != 0.0)
        up = float((r[moved] > 0).mean()) if int(moved.sum()) else float("nan")
        vs = hit_rate_vs_p0(hits, up)
        inf["year"] = float(year)
        inf["n_dates"] = float(sub["date"].nunique())
        inf["up_pct"] = float(100.0 * up) if np.isfinite(up) else float("nan")
        inf["excess_pp"] = _as_float(vs.get("excess_pp"))
        inf["z_vs_drift"] = _as_float(vs.get("z_vs_p0"))
        rows.append(inf)
    return rows


def _confidence_block(
    df: pd.DataFrame,
    pred_r: np.ndarray,
    *,
    train_abs: np.ndarray,
    min_names: int,
) -> dict[str, Any]:
    """Train-quantile |pred_r| slices. Thresholds are not fit on test."""
    mag = np.abs(np.asarray(train_abs, dtype=np.float64))
    mag = mag[np.isfinite(mag)]
    out: dict[str, Any] = {}
    for label, q in (("median", 0.50), ("p70", 0.70), ("p80", 0.80), ("p90", 0.90)):
        thr = float(np.quantile(mag, q)) if mag.size else 0.0
        mask = confidence_mask(pred_r, thr)
        if int(mask.sum()) < 8:
            out[label] = {"threshold": thr, "empty": True}
            continue
        stats = _score_pred_r(df.loc[mask], pred_r[mask], min_names=max(2, min(min_names, 3)))
        slim = slim_accuracy(stats)
        slim["threshold"] = thr
        slim["coverage"] = float(mask.mean()) if mask.size else float("nan")
        out[label] = slim
    return out


def format_ablation_table(ablate: dict[str, Any], promotion: dict[str, Any]) -> str:
    rows = ablate.get("rows") or []
    lines = [
        "ABLATION (fit on TRAIN, gate on locked VAL, report locked TEST — not P&L)",
        f"  accuracy default={promotion.get('accuracy_default')!r}  "
        f"promote_dir={promotion.get('direction')!r}  "
        f"promote_mae={promotion.get('price')!r}",
        f"  {promotion.get('note')}",
        "  variant                         val dir%  val xs pp  val MAE%  "
        "test dir% test xs pp test MAE%  test MAE$  gate",
    ]
    for row in rows:
        name = str(row.get("name") or "")
        v = row.get("val") or {}
        t = row.get("test") or {}
        gate = []
        if row.get("promote_dir"):
            gate.append("dir")
        if row.get("promote_mae"):
            gate.append("mae")
        if row.get("promote"):
            gate.append("DEFAULT")
        mark = ",".join(gate) if gate else "no"
        lines.append(
            f"  {name:<30} {_as_float(v.get('dir_pct')):8.2f} "
            f"{_as_float(v.get('excess_pp')):9.2f} "
            f"{100.0 * _as_float(v.get('mae_pct')):9.4f} "
            f"{_as_float(t.get('dir_pct')):9.2f} "
            f"{_as_float(t.get('excess_pp')):9.2f} "
            f"{100.0 * _as_float(t.get('mae_pct')):9.4f} "
            f"{_as_float(t.get('mae_usd')):9.4f}  "
            f"{mark}"
        )
    lines.append(
        "  xs pp = direction excess vs that split's overnight-up rate (always-up). "
        "PR#8 locked TEST residual*sigma: 51.1% dir / $1.17 / 0.686% MAE."
    )
    lines.append(
        "  Honest drift baseline is train_median_gap (~54.3% on liquid TEST). "
        "Promote dir only if VAL beats that by ≥0.2 pp."
    )
    return "\n".join(lines)


def format_cond_dir_block(ablate: dict[str, Any], promotion: dict[str, Any]) -> str:
    """IDEA A: high-|pred| left-tail / confidence blend vs always-up + residual*σ."""
    row = next(
        (r for r in (ablate.get("rows") or []) if r.get("name") == "cond_dir_blend"),
        None,
    )
    if not row:
        return ""
    params = row.get("params") or {}
    gate = row.get("cond_gate") or {}
    cv = row.get("cond_val") or {}
    ct = row.get("cond_test") or {}
    v = row.get("val") or {}
    t = row.get("test") or {}
    yes = bool(row.get("promote_dir"))
    default = str(promotion.get("direction") or "") == "cond_dir_blend"

    def _slim_line(label: str, slim: dict[str, Any]) -> str:
        slim = slim or {}
        return (
            f"  {label:<22} dir {_as_float(slim.get('dir_pct')):6.2f}%  "
            f"xs {_as_float(slim.get('excess_pp')):+6.2f}pp  "
            f"MAE% {100.0 * _as_float(slim.get('mae_pct')):7.4f}  "
            f"MAE$ {_as_float(slim.get('mae_usd')):7.4f}  "
            f"cover {100.0 * _as_float(slim.get('coverage')):5.1f}%"
        )

    lines = [
        f"PROMOTE COND-DIR BLEND? {'YES' if yes else 'NO'}"
        + ("  (accuracy default)" if default else ""),
        "  high-|pred_r| mix of train left_tail_l1 and confidence_blend; "
        "else train-median always-up. Fit (q, λ) on TRAIN. "
        "VAL gate: slice dir ≥ train-median floor +0.2pp AND ≥ residual*σ "
        "on the same slice +0.2pp. CS skip / live book unchanged.",
        f"  TRAIN q={_as_float(params.get('q')):.2f}  "
        f"λ={_as_float(params.get('lam')):.2f}  "
        f"τ_|pred|={_as_float(params.get('tau_abs')):.6f}  "
        f"(fit_split=train)",
        f"  VAL floor train-median {_as_float(gate.get('val_median_floor_dir_pct')):.2f}%  "
        f"slice resid {_as_float(gate.get('val_resid_slice_dir_pct')):.2f}%  "
        f"slice blend {_as_float(gate.get('val_cond_dir_pct')):.2f}%  "
        f"cover {100.0 * _as_float(gate.get('val_cover')):.1f}%",
        f"  full-frame VAL dir {_as_float(v.get('dir_pct')):.2f}%  "
        f"xs {_as_float(v.get('excess_pp')):+.2f}pp  "
        f"MAE {_as_float(v.get('mae_usd')):.4f}$ / "
        f"{100.0 * _as_float(v.get('mae_pct')):.4f}%",
        "  VAL high-|pred| slice (gate):",
        _slim_line("blend", cv.get("blend") or {}),
        _slim_line("residual*sigma", cv.get("residual_sigma") or {}),
        _slim_line("train-median", cv.get("train_median") or {}),
        f"  full-frame TEST dir {_as_float(t.get('dir_pct')):.2f}%  "
        f"xs {_as_float(t.get('excess_pp')):+.2f}pp  "
        f"MAE {_as_float(t.get('mae_usd')):.4f}$ / "
        f"{100.0 * _as_float(t.get('mae_pct')):.4f}%  (report-only)",
        "  TEST high-|pred| slice (report-only):",
        _slim_line("blend", ct.get("blend") or {}),
        _slim_line("residual*sigma", ct.get("residual_sigma") or {}),
        _slim_line("train-median", ct.get("train_median") or {}),
    ]
    return "\n".join(lines)


def format_decile_reliability_block(
    ablate: dict[str, Any], promotion: dict[str, Any]
) -> str:
    row = next(
        (r for r in (ablate.get("rows") or []) if r.get("name") == "decile_reliability"),
        None,
    )
    if not row:
        return ""
    params = row.get("params") or {}
    v = row.get("val") or {}
    t = row.get("test") or {}
    yes = bool(row.get("promote_dir"))
    default = str(promotion.get("direction") or "") == "decile_reliability"
    n_keep = int(params.get("n_keep") or 0)
    n_bins = int(params.get("n_bins") or 0)
    lines = [
        f"PROMOTE DECILE RELIABILITY? {'YES' if yes else 'NO'}"
        + ("  (accuracy default)" if default else ""),
        "  Keep residual*sigma only in TRAIN pred_r bins whose sign-hit "
        "≥ TRAIN always-up + 0.2pp; else train-median. n_bins TRAIN-chosen. "
        "VAL gate: full-frame dir ≥ train-median +0.2pp AND ≥ residual*σ +0.2pp. "
        "CS skip / live book unchanged.",
        f"  TRAIN n_bins={n_bins}  keep {n_keep}/{n_bins}  "
        f"floor={100.0 * _as_float(params.get('floor')):.2f}%  "
        f"(fit_split=train)",
        f"  VAL dir {_as_float(v.get('dir_pct')):.2f}%  "
        f"xs {_as_float(v.get('excess_pp')):+.2f}pp  "
        f"MAE {_as_float(v.get('mae_usd')):.4f}$ / "
        f"{100.0 * _as_float(v.get('mae_pct')):.4f}%  "
        f"promote_dir={yes}",
        f"  TEST dir {_as_float(t.get('dir_pct')):.2f}%  "
        f"xs {_as_float(t.get('excess_pp')):+.2f}pp  "
        f"MAE {_as_float(t.get('mae_usd')):.4f}$ / "
        f"{100.0 * _as_float(t.get('mae_pct')):.4f}%  (report-only)",
    ]
    return "\n".join(lines)


def format_cs_left_veto_block(
    ablate: dict[str, Any], promotion: dict[str, Any]
) -> str:
    row = next(
        (r for r in (ablate.get("rows") or []) if r.get("name") == "cs_left_veto"),
        None,
    )
    if not row:
        return ""
    params = row.get("params") or {}
    v = row.get("val") or {}
    t = row.get("test") or {}
    yes = bool(row.get("promote_dir"))
    default = str(promotion.get("direction") or "") == "cs_left_veto"
    return "\n".join(
        [
            f"PROMOTE CS-LEFT VETO? {'YES' if yes else 'NO'}"
            + ("  (accuracy default)" if default else ""),
            "  Always-up except CS-bottom ∩ TS left-tail of pred_r. "
            "q and τ are TRAIN-only. VAL gate vs train-median +0.2pp AND "
            "residual*σ +0.2pp. CS skip / live book unchanged.",
            f"  TRAIN q={_as_float(params.get('q')):.2f}  "
            f"τ_q={_as_float(params.get('tau_q')):.2f}  "
            f"τ={_as_float(params.get('tau')):.6f}  "
            f"a_dn={_as_float(params.get('a_dn')):.4f}  "
            f"(fit_split=train)",
            f"  VAL dir {_as_float(v.get('dir_pct')):.2f}%  "
            f"xs {_as_float(v.get('excess_pp')):+.2f}pp  "
            f"MAE {_as_float(v.get('mae_usd')):.4f}$ / "
            f"{100.0 * _as_float(v.get('mae_pct')):.4f}%  "
            f"promote_dir={yes}",
            f"  TEST dir {_as_float(t.get('dir_pct')):.2f}%  "
            f"xs {_as_float(t.get('excess_pp')):+.2f}pp  "
            f"MAE {_as_float(t.get('mae_usd')):.4f}$ / "
            f"{100.0 * _as_float(t.get('mae_pct')):.4f}%  (report-only)",
        ]
    )


def format_logistic_up_block(
    ablate: dict[str, Any], promotion: dict[str, Any]
) -> str:
    row = next(
        (r for r in (ablate.get("rows") or []) if r.get("name") == "logistic_up"),
        None,
    )
    if not row:
        return ""
    params = row.get("params") or {}
    v = row.get("val") or {}
    t = row.get("test") or {}
    yes = bool(row.get("promote_dir"))
    default = str(promotion.get("direction") or "") == "logistic_up"
    return "\n".join(
        [
            f"PROMOTE LOGISTIC P(up)? {'YES' if yes else 'NO'}"
            + ("  (accuracy default)" if default else ""),
            "  TRAIN logistic P(up|pred_r); predict up iff P≥τ. "
            "τ grid on TRAIN direction excess. VAL vs train-median +0.2pp "
            "AND residual*σ +0.2pp. CS skip / live book unchanged.",
            f"  TRAIN a={_as_float(params.get('a')):+.3f}  "
            f"b={_as_float(params.get('b')):+.3f}  "
            f"τ={_as_float(params.get('tau')):.2f}  "
            f"mag={_as_float(params.get('mag')):.5f}  (fit_split=train)",
            f"  VAL dir {_as_float(v.get('dir_pct')):.2f}%  "
            f"xs {_as_float(v.get('excess_pp')):+.2f}pp  "
            f"MAE {_as_float(v.get('mae_usd')):.4f}$ / "
            f"{100.0 * _as_float(v.get('mae_pct')):.4f}%  "
            f"promote_dir={yes}",
            f"  TEST dir {_as_float(t.get('dir_pct')):.2f}%  "
            f"xs {_as_float(t.get('excess_pp')):+.2f}pp  "
            f"MAE {_as_float(t.get('mae_usd')):.4f}$ / "
            f"{100.0 * _as_float(t.get('mae_pct')):.4f}%  (report-only)",
        ]
    )


def format_confidence_block(conf: dict[str, Any]) -> str:
    lines = [
        "CONFIDENCE ( |pred_r| vs TRAIN quantiles; scored on locked TEST )",
    ]
    for key in ("median", "p70", "p80", "p90"):
        row = conf.get(key) or {}
        if row.get("empty"):
            lines.append(f"  {key}: empty")
            continue
        lines.append(
            f"  |pred_r|>={key} thr={float(row.get('threshold') or float('nan')):.6f}  "
            f"cover={100.0 * float(row.get('coverage') or float('nan')):.1f}%  "
            f"dir={float(row.get('dir_pct') or float('nan')):.1f}%  "
            f"excess={float(row.get('excess_pp') or float('nan')):+.2f}pp  "
            f"MAE%={100.0 * float(row.get('mae_pct') or float('nan')):.4f}"
        )
    return "\n".join(lines)


def format_year_slices(years: list[dict[str, Any]], title: str | None = None) -> str:
    lines = [
        title
        or "YEAR SLICES (locked TEST, residual*sigma readout — report only, no retarget)"
    ]
    for row in years:
        lines.append(
            f"  {int(row.get('year', 0))}: dir={float(row.get('hit_rate_pct') or float('nan')):.1f}% "
            f"up={float(row.get('up_pct') or float('nan')):.1f}% "
            f"excess={float(row.get('excess_pp') or float('nan')):+.2f}pp "
            f"n={int(row.get('n') or 0)} z50={float(row.get('z_vs_half') or float('nan')):.2f} "
            f"z_drift={float(row.get('z_vs_drift') or float('nan')):.2f}"
        )
    return "\n".join(lines)


def evaluate_overnight_accuracy(
    data_dir: str,
    universe: str = "liquid",
    *,
    log_fn: Any | None = print,
    ablate: bool = True,
) -> dict[str, Any]:
    """PR #8 locked-TEST baseline plus train-only readouts gated on locked VAL."""
    cfg = overnight_skip_data_config(data_dir, universe)
    if log_fn:
        log_fn(formula_log_line("overnight"))
        log_fn(
            f"recipe=promoted overnight skip {PROMOTED_SKIP}  "
            "accuracy readouts fit on TRAIN; promote on locked VAL only"
        )
    bundle = build_datasets(cfg, log_fn=log_fn)
    w, b, train_ic = fit_promoted_overnight_skip(bundle)
    min_names = int(bundle.get("cs_min_names", 30))
    px, missing_px = load_split_px(bundle, cfg, log_fn=log_fn)
    frames: dict[str, pd.DataFrame] = {}
    feats: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        df, x = _frame_for_split(bundle, split, px, w, b, min_names)
        frames[split] = df
        feats[split] = x if x is not None else np.zeros((0, 0))
        if log_fn:
            log_fn(
                f"{split}: {len(df)} name-dates, {df['date'].nunique() if not df.empty else 0} dates"
            )

    test_df = frames["test"]
    baseline_test = score_eval_frame(test_df, min_names=min_names)
    cuts = {}
    if bundle.get("meta"):
        cuts = {
            "train_end": bundle["meta"][0].get("train_end"),
            "val_end": bundle["meta"][0].get("val_end"),
        }
    payload: dict[str, Any] = {
        "estimand": "overnight residual skip; accuracy vs realized r_on / next open",
        "formula": OVERNIGHT_FORMULA,
        "recipe": dict(PROMOTED_SKIP),
        "levers": "default overnight skip; optional train-only accuracy readouts",
        "split": "locked TEST last bars; dates with >= cs_min_names names",
        "cs_min_names": min_names,
        "n_trading_names": int(bundle.get("n_trading_names") or 0),
        "n_test_symbols_with_rows": int(test_df["symbol"].nunique()) if not test_df.empty else 0,
        "missing_price_files": missing_px,
        "calendar_cuts": cuts,
        "train_cs_ic": float(train_ic),
        **baseline_test,
        "n_samples_before_cs_floor": int(len(test_df)),
        "n_dates_before_cs_floor": int(test_df["date"].nunique()) if not test_df.empty else 0,
        "baseline_locked_test_pr8": dict(BASELINE_TEST),
    }
    if not ablate or frames["train"].empty or frames["val"].empty:
        return payload

    tr = frames["train"]
    va = frames["val"]
    te = frames["test"]
    xtr, xva, xte = feats["train"], feats["val"], feats["test"]
    train_pred_r = tr["pred_r"].to_numpy(dtype=np.float64)
    train_r = tr["r_on"].to_numpy(dtype=np.float64)
    train_y_scale = tr["y"].to_numpy(dtype=np.float64) * tr["scale"].to_numpy(dtype=np.float64)
    train_hedge = train_r - train_y_scale

    a_ols, b_ols = fit_affine_ols(train_pred_r, train_r)
    a_l1, b_l1 = fit_affine_l1(train_pred_r, train_r)
    mu_med = train_constant(train_r, "median")
    mu_mean = train_constant(train_r, "mean")
    hedge_mu = train_constant(train_hedge, "mean")

    w_ts, b_ts, ic_ts = fit_ts_overnight_ridge(
        xtr, train_r, tr["date"].to_numpy(dtype=np.int64), mask_mode="all"
    )
    w_ts_cs, b_ts_cs, _ = fit_ts_overnight_ridge(
        xtr, train_r, tr["date"].to_numpy(dtype=np.int64), mask_mode="no_long_ts"
    )
    w_sgn, b_sgn, _ = fit_sign_overnight_ridge(
        xtr, train_r, tr["date"].to_numpy(dtype=np.int64), mask_mode="all"
    )
    w_date, b_date = fit_date_level_overnight(
        xtr, train_r, tr["date"].to_numpy(dtype=np.int64)
    )

    def _ts(split_x: np.ndarray) -> np.ndarray:
        return predict_linear(split_x, w_ts, b_ts)

    def _ts_cs(split_x: np.ndarray) -> np.ndarray:
        return predict_linear(split_x, w_ts_cs, b_ts_cs)

    def _sgn(split_x: np.ndarray) -> np.ndarray:
        raw = predict_linear(split_x, w_sgn, b_sgn)
        return raw

    a_sgn, b_sgn_aff = fit_affine_l1(_sgn(xtr), train_r)
    date_tr = predict_date_level(xtr, tr["date"].to_numpy(dtype=np.int64), w_date, b_date)
    a_hy, b_hy = fit_affine_l1(date_tr + train_pred_r, train_r)
    a_huber, b_huber = fit_affine_huber(train_pred_r, train_r)
    a_pos, b_pos, a_neg, b_neg = fit_piecewise_l1(train_pred_r, train_r)
    bin_edges, bin_vals = fit_bin_constants(train_pred_r, train_r, n_bins=7)
    veto = fit_drift_veto(train_pred_r, train_r)
    left_tail = fit_left_tail_l1(train_pred_r, train_r)
    blend = fit_confidence_blend(train_pred_r, train_r, a_l1, b_l1, mu_med)
    cond_blend = fit_cond_dir_blend(train_pred_r, train_r, left_tail, blend, mu_med)
    decile_rel = fit_decile_reliability(train_pred_r, train_r)
    cs_veto = fit_cs_left_veto(
        train_pred_r,
        train_r,
        tr["pred"].to_numpy(dtype=np.float64),
        tr["date"].to_numpy(dtype=np.int64),
    )
    logit_up = fit_logistic_up(train_pred_r, train_r)
    dow_table, dow_default = fit_group_median(
        train_r, weekday_of_dates(tr["date"].to_numpy(dtype=np.int64))
    )
    train_dow = apply_group_median(
        weekday_of_dates(tr["date"].to_numpy(dtype=np.int64)), dow_table, dow_default
    )
    a_dow, b_dow = fit_affine_l1(train_pred_r, train_r - train_dow)
    rec_w = recency_weights(tr["date"].to_numpy(dtype=np.int64), halflife_years=6.0)
    rec_med = weighted_median(train_r, rec_w)
    a_rec, b_rec = fit_affine_l1(train_pred_r, train_r, weights=rec_w)
    vol_tr = tr["vol_level"].to_numpy(dtype=np.float64) if "vol_level" in tr.columns else train_pred_r
    vol_edges, vol_vals = fit_bin_constants(vol_tr, train_r, n_bins=3)

    def _dow_mu(frame: pd.DataFrame) -> np.ndarray:
        if "weekday" in frame.columns:
            wd = frame["weekday"].to_numpy(dtype=np.int64)
        else:
            wd = weekday_of_dates(frame["date"].to_numpy(dtype=np.int64))
        return apply_group_median(wd, dow_table, dow_default)

    def _vol_mu(frame: pd.DataFrame) -> np.ndarray:
        if "vol_level" in frame.columns:
            return apply_bin_constants(
                frame["vol_level"].to_numpy(dtype=np.float64), vol_edges, vol_vals
            )
        return np.full(len(frame), mu_med, dtype=np.float64)

    piecewise_spec = {
        "kind": "piecewise_l1",
        "a_pos": a_pos,
        "b_pos": b_pos,
        "a_neg": a_neg,
        "b_neg": b_neg,
    }
    bin_spec = {
        "kind": "bin_calibrate",
        "edges": bin_edges.tolist(),
        "values": bin_vals.tolist(),
    }
    veto_spec = {"kind": "drift_veto", **veto}
    left_spec = {"kind": "left_tail_l1", **left_tail}
    blend_spec = {"kind": "confidence_blend", **blend}
    cond_spec = {"kind": "cond_dir_blend", **cond_blend}
    decile_spec = {"kind": "decile_reliability", **decile_rel}
    cs_veto_spec = {"kind": "cs_left_veto", **cs_veto}
    logit_spec = {"kind": "logistic_up", **logit_up}
    dow_spec = {
        "kind": "dow_gap",
        "by_dow": _table_to_jsonable(dow_table),
        "default": dow_default,
    }
    dow_res_spec = {
        "kind": "dow_plus_residual",
        "by_dow": _table_to_jsonable(dow_table),
        "default": dow_default,
        "a": a_dow,
        "b": b_dow,
    }
    vol_spec = {
        "kind": "vol_regime_gap",
        "edges": vol_edges.tolist(),
        "values": vol_vals.tolist(),
    }

    variants: list[tuple[str, str, dict[str, Any], Any, Any]] = [
        (
            "residual_sigma",
            "baseline",
            {"a": 1.0, "b": 0.0},
            lambda d, x: d["pred_r"].to_numpy(dtype=np.float64),
            lambda d, x: d["pred_r"].to_numpy(dtype=np.float64),
        ),
        (
            "zero_move",
            "constant",
            {"a": 0.0, "b": 0.0},
            lambda d, x: np.zeros(len(d), dtype=np.float64),
            lambda d, x: np.zeros(len(d), dtype=np.float64),
        ),
        (
            "train_median_gap",
            "constant",
            {"b": mu_med},
            lambda d, x, m=mu_med: np.full(len(d), m, dtype=np.float64),
            lambda d, x, m=mu_med: np.full(len(d), m, dtype=np.float64),
        ),
        (
            "train_mean_gap",
            "constant",
            {"b": mu_mean},
            lambda d, x, m=mu_mean: np.full(len(d), m, dtype=np.float64),
            lambda d, x, m=mu_mean: np.full(len(d), m, dtype=np.float64),
        ),
        (
            "affine_ols",
            "calibrate",
            {"a": a_ols, "b": b_ols},
            lambda d, x, a=a_ols, bb=b_ols: apply_affine(
                d["pred_r"].to_numpy(dtype=np.float64), a, bb
            ),
            lambda d, x, a=a_ols, bb=b_ols: apply_affine(
                d["pred_r"].to_numpy(dtype=np.float64), a, bb
            ),
        ),
        (
            "affine_l1",
            "calibrate",
            {"a": a_l1, "b": b_l1},
            lambda d, x, a=a_l1, bb=b_l1: apply_affine(
                d["pred_r"].to_numpy(dtype=np.float64), a, bb
            ),
            lambda d, x, a=a_l1, bb=b_l1: apply_affine(
                d["pred_r"].to_numpy(dtype=np.float64), a, bb
            ),
        ),
        (
            "residual_plus_hedge_mean",
            "calibrate",
            {"hedge_mean": hedge_mu},
            lambda d, x, h=hedge_mu: d["pred_r"].to_numpy(dtype=np.float64) + h,
            lambda d, x, h=hedge_mu: d["pred_r"].to_numpy(dtype=np.float64) + h,
        ),
        (
            "ts_ridge_all",
            "ts_overnight",
            {"mask": "all", "train_ic": float(ic_ts)},
            lambda d, x: _ts(x),
            lambda d, x: _ts(x),
        ),
        (
            "ts_ridge_no_long_ts",
            "ts_overnight",
            {"mask": "no_long_ts"},
            lambda d, x: _ts_cs(x),
            lambda d, x: _ts_cs(x),
        ),
        (
            "sign_ridge_calibrated",
            "direction",
            {"a": a_sgn, "b": b_sgn_aff},
            lambda d, x, a=a_sgn, bb=b_sgn_aff: apply_affine(_sgn(x), a, bb),
            lambda d, x, a=a_sgn, bb=b_sgn_aff: apply_affine(_sgn(x), a, bb),
        ),
        (
            "date_plus_residual",
            "hybrid",
            {"a": a_hy, "b": b_hy},
            lambda d, x, a=a_hy, bb=b_hy: apply_affine(
                predict_date_level(x, d["date"].to_numpy(dtype=np.int64), w_date, b_date)
                + d["pred_r"].to_numpy(dtype=np.float64),
                a,
                bb,
            ),
            lambda d, x, a=a_hy, bb=b_hy: apply_affine(
                predict_date_level(x, d["date"].to_numpy(dtype=np.int64), w_date, b_date)
                + d["pred_r"].to_numpy(dtype=np.float64),
                a,
                bb,
            ),
        ),
        (
            "huber_affine",
            "calibrate",
            {"kind": "huber_affine", "a": a_huber, "b": b_huber},
            lambda d, x, a=a_huber, bb=b_huber: apply_affine(
                d["pred_r"].to_numpy(dtype=np.float64), a, bb
            ),
            lambda d, x, a=a_huber, bb=b_huber: apply_affine(
                d["pred_r"].to_numpy(dtype=np.float64), a, bb
            ),
        ),
        (
            "piecewise_l1",
            "calibrate",
            piecewise_spec,
            lambda d, x, spec=piecewise_spec: apply_piecewise_l1(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["a_pos"],
                spec["b_pos"],
                spec["a_neg"],
                spec["b_neg"],
            ),
            lambda d, x, spec=piecewise_spec: apply_piecewise_l1(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["a_pos"],
                spec["b_pos"],
                spec["a_neg"],
                spec["b_neg"],
            ),
        ),
        (
            "bin_calibrate",
            "calibrate",
            bin_spec,
            lambda d, x, e=bin_edges, v=bin_vals: apply_bin_constants(
                d["pred_r"].to_numpy(dtype=np.float64), e, v
            ),
            lambda d, x, e=bin_edges, v=bin_vals: apply_bin_constants(
                d["pred_r"].to_numpy(dtype=np.float64), e, v
            ),
        ),
        (
            "drift_veto",
            "direction",
            veto_spec,
            lambda d, x, spec=veto: apply_drift_veto(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["tau"],
                spec["a_dn"],
                spec["b_dn"],
                spec["b_up"],
            ),
            lambda d, x, spec=veto: apply_drift_veto(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["tau"],
                spec["a_dn"],
                spec["b_dn"],
                spec["b_up"],
            ),
        ),
        (
            "left_tail_l1",
            "calibrate",
            left_spec,
            lambda d, x, spec=left_tail: apply_left_tail_l1(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["tau"],
                spec["a_neg"],
                spec["b_up"],
            ),
            lambda d, x, spec=left_tail: apply_left_tail_l1(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["tau"],
                spec["a_neg"],
                spec["b_up"],
            ),
        ),
        (
            "confidence_blend",
            "direction",
            blend_spec,
            lambda d, x, spec=blend: apply_confidence_blend(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["tau"],
                spec["a"],
                spec["b"],
                spec["b_up"],
            ),
            lambda d, x, spec=blend: apply_confidence_blend(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["tau"],
                spec["a"],
                spec["b"],
                spec["b_up"],
            ),
        ),
        (
            "cond_dir_blend",
            "direction",
            cond_spec,
            lambda d, x, spec=cond_blend: apply_cond_dir_blend(
                d["pred_r"].to_numpy(dtype=np.float64),
                tau_abs=spec["tau_abs"],
                lam=spec["lam"],
                left_tau=spec["left_tau"],
                left_a_neg=spec["left_a_neg"],
                left_b_up=spec["left_b_up"],
                conf_tau=spec["conf_tau"],
                conf_a=spec["conf_a"],
                conf_b=spec["conf_b"],
                conf_b_up=spec["conf_b_up"],
                b_up=spec["b_up"],
            ),
            lambda d, x, spec=cond_blend: apply_cond_dir_blend(
                d["pred_r"].to_numpy(dtype=np.float64),
                tau_abs=spec["tau_abs"],
                lam=spec["lam"],
                left_tau=spec["left_tau"],
                left_a_neg=spec["left_a_neg"],
                left_b_up=spec["left_b_up"],
                conf_tau=spec["conf_tau"],
                conf_a=spec["conf_a"],
                conf_b=spec["conf_b"],
                conf_b_up=spec["conf_b_up"],
                b_up=spec["b_up"],
            ),
        ),
        (
            "decile_reliability",
            "direction",
            decile_spec,
            lambda d, x, spec=decile_rel: apply_decile_reliability(
                d["pred_r"].to_numpy(dtype=np.float64),
                np.asarray(spec["edges"], dtype=np.float64),
                np.asarray(spec["keep"], dtype=np.bool_),
                float(spec["b_up"]),
            ),
            lambda d, x, spec=decile_rel: apply_decile_reliability(
                d["pred_r"].to_numpy(dtype=np.float64),
                np.asarray(spec["edges"], dtype=np.float64),
                np.asarray(spec["keep"], dtype=np.bool_),
                float(spec["b_up"]),
            ),
        ),
        (
            "cs_left_veto",
            "direction",
            cs_veto_spec,
            lambda d, x, spec=cs_veto: apply_cs_left_veto(
                d["pred_r"].to_numpy(dtype=np.float64),
                d["pred"].to_numpy(dtype=np.float64),
                d["date"].to_numpy(dtype=np.int64),
                spec["tau"],
                spec["q"],
                spec["a_dn"],
                spec["b_dn"],
                spec["b_up"],
            ),
            lambda d, x, spec=cs_veto: apply_cs_left_veto(
                d["pred_r"].to_numpy(dtype=np.float64),
                d["pred"].to_numpy(dtype=np.float64),
                d["date"].to_numpy(dtype=np.int64),
                spec["tau"],
                spec["q"],
                spec["a_dn"],
                spec["b_dn"],
                spec["b_up"],
            ),
        ),
        (
            "logistic_up",
            "direction",
            logit_spec,
            lambda d, x, spec=logit_up: apply_logistic_up(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["a"],
                spec["b"],
                spec["tau"],
                spec["mag"],
            ),
            lambda d, x, spec=logit_up: apply_logistic_up(
                d["pred_r"].to_numpy(dtype=np.float64),
                spec["a"],
                spec["b"],
                spec["tau"],
                spec["mag"],
            ),
        ),
        (
            "dow_gap",
            "calendar",
            dow_spec,
            lambda d, x: _dow_mu(d),
            lambda d, x: _dow_mu(d),
        ),
        (
            "dow_plus_residual",
            "hybrid",
            dow_res_spec,
            lambda d, x, a=a_dow, bb=b_dow: _dow_mu(d)
            + apply_affine(d["pred_r"].to_numpy(dtype=np.float64), a, bb),
            lambda d, x, a=a_dow, bb=b_dow: _dow_mu(d)
            + apply_affine(d["pred_r"].to_numpy(dtype=np.float64), a, bb),
        ),
        (
            "vol_regime_gap",
            "regime",
            vol_spec,
            lambda d, x: _vol_mu(d),
            lambda d, x: _vol_mu(d),
        ),
        (
            "recency_median",
            "regime",
            {"kind": "recency_median", "b": rec_med, "halflife_years": 6.0},
            lambda d, x, m=rec_med: np.full(len(d), m, dtype=np.float64),
            lambda d, x, m=rec_med: np.full(len(d), m, dtype=np.float64),
        ),
        (
            "recency_affine_l1",
            "calibrate",
            {"kind": "affine", "a": a_rec, "b": b_rec, "halflife_years": 6.0},
            lambda d, x, a=a_rec, bb=b_rec: apply_affine(
                d["pred_r"].to_numpy(dtype=np.float64), a, bb
            ),
            lambda d, x, a=a_rec, bb=b_rec: apply_affine(
                d["pred_r"].to_numpy(dtype=np.float64), a, bb
            ),
        ),
    ]

    val_base = slim_accuracy(score_eval_frame(va, min_names=min_names))
    val_med = slim_accuracy(
        _score_pred_r(va, np.full(len(va), mu_med, dtype=np.float64), min_names)
    )
    zero_val_mae = float(val_base["zero_mae_pct"])

    rows: list[dict[str, Any]] = []
    pred_r_by_name: dict[str, dict[str, np.ndarray]] = {}
    for name, kind, params, val_fn, test_fn in variants:
        val_pr = val_fn(va, xva)
        test_pr = test_fn(te, xte)
        pred_r_by_name[name] = {"val": val_pr, "test": test_pr}
        row = _variant_row(
            name,
            kind=kind,
            fit_split="train",
            val_stats=_score_pred_r(va, val_pr, min_names),
            test_stats=_score_pred_r(te, test_pr, min_names),
            params=params,
            baseline_val=val_base,
            median_val=val_med,
            zero_val_mae=zero_val_mae,
        )
        rows.append(row)

    # IDEA A: conditional direction on high-|pred|. Gate vs always-up floor
    # (train-median full-frame VAL) AND residual*sigma on the same TRAIN tau slice.
    cond_row = next((r for r in rows if r.get("name") == "cond_dir_blend"), None)
    if cond_row is not None:
        tau_abs = float(cond_blend.get("tau_abs") or 0.0)
        slice_min = max(2, min(3, min_names))

        def _slice_slim(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
            mask = cond_abs_mask(frame["pred_r"].to_numpy(dtype=np.float64), tau_abs)
            cover = float(mask.mean()) if mask.size else float("nan")
            if int(mask.sum()) < 8:
                out = slim_accuracy({"empty": True})
                out["coverage"] = cover
                out["n_slice"] = float(int(mask.sum()))
                return out
            out = slim_accuracy(_score_pred_r(frame.loc[mask], pred[mask], slice_min))
            out["coverage"] = cover
            out["n_slice"] = float(int(mask.sum()))
            return out

        va_resid = va["pred_r"].to_numpy(dtype=np.float64)
        te_resid = te["pred_r"].to_numpy(dtype=np.float64)
        va_med = np.full(len(va), mu_med, dtype=np.float64)
        te_med = np.full(len(te), mu_med, dtype=np.float64)
        va_hat = pred_r_by_name["cond_dir_blend"]["val"]
        te_hat = pred_r_by_name["cond_dir_blend"]["test"]
        cond_val = {
            "blend": _slice_slim(va, va_hat),
            "residual_sigma": _slice_slim(va, va_resid),
            "train_median": _slice_slim(va, va_med),
        }
        cond_test = {
            "blend": _slice_slim(te, te_hat),
            "residual_sigma": _slice_slim(te, te_resid),
            "train_median": _slice_slim(te, te_med),
        }
        cond_row["cond_val"] = cond_val
        cond_row["cond_test"] = cond_test
        cond_row["params"] = {
            **dict(cond_row.get("params") or {}),
            "tau_abs": tau_abs,
            "q": float(cond_blend.get("q") or 0.0),
            "lam": float(cond_blend.get("lam") or 0.0),
        }
        blend_dir = _as_float((cond_val.get("blend") or {}).get("dir_pct"))
        resid_dir = _as_float((cond_val.get("residual_sigma") or {}).get("dir_pct"))
        floor_dir = _as_float(val_med.get("dir_pct"))
        cover = _as_float((cond_val.get("blend") or {}).get("coverage"))
        beats_floor = (
            np.isfinite(blend_dir)
            and np.isfinite(floor_dir)
            and blend_dir >= floor_dir + 100.0 * DIR_LIFT
        )
        beats_resid = (
            np.isfinite(blend_dir)
            and np.isfinite(resid_dir)
            and blend_dir >= resid_dir + 100.0 * DIR_LIFT
        )
        cover_ok = (not np.isfinite(cover)) or cover >= 0.05
        cond_row["promote_dir"] = bool(beats_floor and beats_resid and cover_ok)
        cond_row["cond_gate"] = {
            "beats_train_median_floor": beats_floor,
            "beats_residual_sigma_slice": beats_resid,
            "cover_ok": cover_ok,
            "val_cond_dir_pct": blend_dir,
            "val_resid_slice_dir_pct": resid_dir,
            "val_median_floor_dir_pct": floor_dir,
            "val_cover": cover,
        }

    # Liquid sleeve on the residual skip (eval filter, same w).
    if "turnover_z" in te.columns and not te.empty:
        for split_name, sdf, sx in (("val", va, xva), ("test", te, xte)):
            del split_name, sdf, sx
        sleeve_va = adv_sleeve_mask(va, pctile=0.67)
        sleeve_te = adv_sleeve_mask(te, pctile=0.67)
        if int(sleeve_va.sum()) >= 8 and int(sleeve_te.sum()) >= 8:
            rows.append(
                _variant_row(
                    "adv_sleeve_residual",
                    kind="filter",
                    fit_split="train (eval filter)",
                    val_stats=_score_pred_r(
                        va.loc[sleeve_va],
                        va["pred_r"].to_numpy(dtype=np.float64)[sleeve_va],
                        min_names=max(3, min(8, min_names)),
                    ),
                    test_stats=_score_pred_r(
                        te.loc[sleeve_te],
                        te["pred_r"].to_numpy(dtype=np.float64)[sleeve_te],
                        min_names=max(3, min(8, min_names)),
                    ),
                    params={"adv_floor_pctile": 0.67},
                    baseline_val=val_base,
                    median_val=val_med,
                    zero_val_mae=zero_val_mae,
                )
            )

    promotion = _pick_promoted(rows)
    default_name = str(promotion.get("accuracy_default") or "residual_sigma")
    default_pr_test = pred_r_by_name.get(default_name, {}).get("test")
    if default_pr_test is None:
        default_pr_test = te["pred_r"].to_numpy(dtype=np.float64)

    payload["ablation"] = {
        "rows": rows,
        "calibrators": {
            "affine_ols": {"a": a_ols, "b": b_ols},
            "affine_l1": {"a": a_l1, "b": b_l1},
            "huber_affine": {"a": a_huber, "b": b_huber},
            "train_median_gap": mu_med,
            "train_mean_gap": mu_mean,
            "hedge_mean": hedge_mu,
            "sign_affine": {"a": a_sgn, "b": b_sgn_aff},
            "hybrid_affine": {"a": a_hy, "b": b_hy},
            "ts_ridge_train_ic": float(ic_ts),
            "drift_veto": veto,
            "left_tail_l1": left_tail,
            "confidence_blend": blend,
            "cond_dir_blend": cond_blend,
            "decile_reliability": decile_rel,
            "cs_left_veto": cs_veto,
            "logistic_up": logit_up,
            "piecewise_l1": piecewise_spec,
            "bin_calibrate": bin_spec,
            "dow_gap": dow_spec,
            "recency_median": rec_med,
        },
    }
    payload["promotion"] = promotion
    residual_affine = {
        "residual_sigma": (1.0, 0.0),
        "zero_move": (0.0, 0.0),
        "train_median_gap": (0.0, mu_med),
        "train_mean_gap": (0.0, mu_mean),
        "affine_ols": (a_ols, b_ols),
        "affine_l1": (a_l1, b_l1),
        "huber_affine": (a_huber, b_huber),
        "residual_plus_hedge_mean": (1.0, hedge_mu),
        "recency_median": (0.0, rec_med),
        "recency_affine_l1": (a_rec, b_rec),
    }.get(default_name)
    default_params = next((r["params"] for r in rows if r["name"] == default_name), {})
    payload["calibrate"] = {
        "name": default_name,
        "kind": str(default_params.get("kind") or default_name),
        "a": None if residual_affine is None else residual_affine[0],
        "b": None if residual_affine is None else residual_affine[1],
        "applies_to": "pred*sigma residual overnight log-return (generate.py --calibrate-json)",
        "params": default_params,
        **({k: v for k, v in default_params.items() if k != "params"}),
    }
    # Train quantiles of the chosen readout. Thresholds are not fit on test.
    train_default = {
        "residual_sigma": train_pred_r,
        "zero_move": np.zeros_like(train_pred_r),
        "train_median_gap": np.full_like(train_pred_r, mu_med),
        "train_mean_gap": np.full_like(train_pred_r, mu_mean),
        "affine_ols": apply_affine(train_pred_r, a_ols, b_ols),
        "affine_l1": apply_affine(train_pred_r, a_l1, b_l1),
        "residual_plus_hedge_mean": train_pred_r + hedge_mu,
        "ts_ridge_all": _ts(xtr),
        "ts_ridge_no_long_ts": _ts_cs(xtr),
        "sign_ridge_calibrated": apply_affine(_sgn(xtr), a_sgn, b_sgn_aff),
        "date_plus_residual": apply_affine(date_tr + train_pred_r, a_hy, b_hy),
        "huber_affine": apply_affine(train_pred_r, a_huber, b_huber),
        "piecewise_l1": apply_piecewise_l1(train_pred_r, a_pos, b_pos, a_neg, b_neg),
        "bin_calibrate": apply_bin_constants(train_pred_r, bin_edges, bin_vals),
        "drift_veto": apply_drift_veto(
            train_pred_r, veto["tau"], veto["a_dn"], veto["b_dn"], veto["b_up"]
        ),
        "left_tail_l1": apply_left_tail_l1(
            train_pred_r, left_tail["tau"], left_tail["a_neg"], left_tail["b_up"]
        ),
        "confidence_blend": apply_confidence_blend(
            train_pred_r, blend["tau"], blend["a"], blend["b"], blend["b_up"]
        ),
        "cond_dir_blend": apply_cond_dir_blend(
            train_pred_r,
            tau_abs=cond_blend["tau_abs"],
            lam=cond_blend["lam"],
            left_tau=cond_blend["left_tau"],
            left_a_neg=cond_blend["left_a_neg"],
            left_b_up=cond_blend["left_b_up"],
            conf_tau=cond_blend["conf_tau"],
            conf_a=cond_blend["conf_a"],
            conf_b=cond_blend["conf_b"],
            conf_b_up=cond_blend["conf_b_up"],
            b_up=cond_blend["b_up"],
        ),
        "decile_reliability": apply_decile_reliability(
            train_pred_r,
            np.asarray(decile_rel["edges"], dtype=np.float64),
            np.asarray(decile_rel["keep"], dtype=np.bool_),
            float(decile_rel["b_up"]),
        ),
        "cs_left_veto": apply_cs_left_veto(
            train_pred_r,
            tr["pred"].to_numpy(dtype=np.float64),
            tr["date"].to_numpy(dtype=np.int64),
            cs_veto["tau"],
            cs_veto["q"],
            cs_veto["a_dn"],
            cs_veto["b_dn"],
            cs_veto["b_up"],
        ),
        "logistic_up": apply_logistic_up(
            train_pred_r,
            logit_up["a"],
            logit_up["b"],
            logit_up["tau"],
            logit_up["mag"],
        ),
        "dow_gap": train_dow,
        "dow_plus_residual": train_dow + apply_affine(train_pred_r, a_dow, b_dow),
        "vol_regime_gap": apply_bin_constants(vol_tr, vol_edges, vol_vals),
        "recency_median": np.full_like(train_pred_r, rec_med),
        "recency_affine_l1": apply_affine(train_pred_r, a_rec, b_rec),
    }.get(default_name, train_pred_r)
    payload["confidence"] = _confidence_block(
        te, default_pr_test, train_abs=train_default, min_names=min_names
    )
    payload["year_slices"] = _year_direction(te)
    payload["year_slices_readout"] = _year_direction(apply_readout(te, default_pr_test))
    if log_fn:
        log_fn(
            f"accuracy default={default_name!r}  "
            f"promote_dir={promotion.get('direction')!r}  "
            f"promote_mae={promotion.get('price')!r}"
        )
    return payload
