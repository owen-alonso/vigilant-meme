"""Locked-window TRUE/FALSE overnight accuracy (not live P&L).

The promoted overnight skip predicts a *residual* in vol units. This module
converts that to an implied overnight log-return ``pred * sigma`` (same as
``generate.py``) and scores:

- direction vs realized ``r_on = log(open_{t+1}) - log(close_t)``
- excess hit rate vs the unconditional overnight-up drift (always-long)
- implied next-open vs actual next open
- long-only book up-rate on the within-date top residual names
- book-aligned sleeve overnight-up (TRAIN q / |pred| grid, VAL-gated)
- within-date relative direction vs CS median (VAL-gated vs 50%)
- long-half absolute overnight-up (pred > CS median) vs the up-floor
- H∩E stack (long-half ∩ TRAIN top-q) relative / absolute / live-IR gates
- short-sleeve overnight down-rate on the within-date bottom residual names
- book-aligned short sleeve overnight-down (TRAIN bottom-q / |pred| grid, VAL-gated)
- symmetric long-E / short-J LS (paper zero-cost + live_locate vs q20)
- two-stage next-open MAE (residual→gap, then gap→price using close_t)
- sparse MAE (calibrate only when |pred*sigma| is large; else zero-move)

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
# IDEA E: book-aligned sleeve overnight-up (pp, not fractions).
BOOK_ALIGN_QS = (0.70, 0.80, 0.90)  # top 30/20/10%
BOOK_ALIGN_ABS_QS = (0.0, 0.50, 0.70)  # 0 = no |pred| floor
BOOK_UP_FLOOR_PP = 0.50
BOOK_UP_BASE_PP = 0.20
BOOK_ALIGN_COVER = 0.05
# IDEA G: clear VAL % MAE margin vs residual×σ / zero-move / train-median (0.5 bp).
MAE_LIFT = 5e-5
SECTOR_MAE_MAPS = ("affine_l1", "piecewise_l1", "huber_affine", "bin_calibrate")
# IDEA L: two-stage residual→gap→next-open, plus residual+DOW+vol ridge.
TWO_STAGE_MAE_MAPS = (
    "two_stage_l1_pct",
    "two_stage_huber_pct",
    "two_stage_piecewise_pct",
    "two_stage_l1_usd",
    "ridge_resid_dow_vol",
)
# IDEA M: sparse MAE — map only when |pred*sigma| >= τ, else zero-move.
SPARSE_MAE_MAPS = ("sparse_l1", "sparse_huber")
SPARSE_ABS_QS = (0.0, 0.30, 0.50, 0.70, 0.80, 0.90)
SPARSE_COVER = 0.05
# IDEA H: within-date relative direction vs 50%, and long-half overnight-up.
REL_DIR_LIFT_PP = 0.50
REL_DIR_Z = 1.0
REL_COVER = 0.05
REL_ABS_QS = (0.0, 0.50, 0.70)
# IDEA I: H relative-dir ∩ E top-q / |pred| (light TRAIN re-grid).
STACK_QS = BOOK_ALIGN_QS
STACK_ABS_QS = BOOK_ALIGN_ABS_QS
STACK_IR_LIFT = 0.05
STACK_DD_TOL = 0.05
# IDEA J: short-sleeve overnight-down (bottom-q, mirror of E).
SHORT_ALIGN_QS = (0.10, 0.20, 0.30)  # bottom 10/20/30%
SHORT_ALIGN_ABS_QS = BOOK_ALIGN_ABS_QS
SHORT_IR_LIFT = 0.05
# IDEA K: symmetric long-E / short-J live_locate vs q20.
EJ_IR_LIFT = 0.05
EJ_DD_TOL = 0.05
EJ_COVER = 0.05
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


def cs_top_abs_mask(
    df: pd.DataFrame,
    *,
    q: float,
    abs_tau: float = 0.0,
    score_col: str = "pred",
    min_names: int = 3,
) -> np.ndarray:
    """Within-date top residual names, optional causal |pred| floor.

    ``q=0.80`` is the top 20%. ``abs_tau`` is a TRAIN quantile of ``|pred|``
    (0 = off). Next open is never used.
    """
    n = len(df)
    out = np.zeros(n, dtype=bool)
    if df.empty or score_col not in df.columns:
        return out
    dates = df["date"].to_numpy(dtype=np.int64)
    score = df[score_col].to_numpy(dtype=np.float64)
    mag = np.abs(score)
    tau = float(abs_tau)
    for key in np.unique(dates):
        sel = dates == key
        row = score[sel]
        finite = np.isfinite(row)
        if int(finite.sum()) < int(min_names):
            continue
        cut = float(np.nanquantile(row, float(q)))
        keep = finite & (row >= cut)
        if tau > 0.0:
            keep = keep & (mag[sel] >= tau)
        out[sel] = keep
    return out


def score_book_aligned_sleeve(
    df: pd.DataFrame,
    *,
    q: float,
    abs_tau: float = 0.0,
    min_names: int = 3,
) -> dict[str, Any]:
    """Overnight up-rate + sleeve MAE on the CS top-q residual names."""
    top_pct = float(100.0 * (1.0 - float(q)))
    name = f"book_top{int(round(top_pct))}"
    if float(abs_tau) > 0.0:
        name = f"{name}_abs"
    empty = {
        "name": name,
        "q": float(q),
        "abs_tau": float(abs_tau),
        "top_pct": top_pct,
        "up_pct": float("nan"),
        "uncond_up_pct": float("nan"),
        "excess_pp": float("nan"),
        "n": 0.0,
        "n_dates": 0.0,
        "coverage": float("nan"),
        "mae_usd": float("nan"),
        "mae_pct": float("nan"),
        "full_mae_usd": float("nan"),
        "full_mae_pct": float("nan"),
        "paper_ir": float("nan"),
    }
    if df.empty or "pred" not in df.columns or "r_on" not in df.columns:
        return empty
    mask = cs_top_abs_mask(
        df, q=float(q), abs_tau=float(abs_tau), score_col="pred", min_names=min_names
    )
    r = df["r_on"].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy(dtype=np.int64)
    moved = mask & np.isfinite(r) & (r != 0.0)
    uncond = r[np.isfinite(r) & (r != 0.0)]
    uncond_up = float((uncond > 0).mean()) if uncond.size else float("nan")
    n_dates = float(pd.Series(dates[mask]).nunique()) if int(mask.sum()) else 0.0
    cover = float(mask.mean()) if mask.size else float("nan")
    empty["n"] = float(int(moved.sum()))
    empty["n_dates"] = n_dates
    empty["coverage"] = cover
    empty["uncond_up_pct"] = (
        float(100.0 * uncond_up) if np.isfinite(uncond_up) else float("nan")
    )
    if "close" in df.columns and "next_open" in df.columns:
        close = df["close"].to_numpy(dtype=np.float64)
        nxt = df["next_open"].to_numpy(dtype=np.float64)
        if "implied_open" in df.columns:
            implied = df["implied_open"].to_numpy(dtype=np.float64)
        else:
            implied = close
        ok = np.isfinite(close) & np.isfinite(nxt) & (close > 0)
        if int(ok.sum()):
            empty["full_mae_usd"] = float(np.mean(np.abs(implied[ok] - nxt[ok])))
            empty["full_mae_pct"] = float(
                np.mean(np.abs(implied[ok] - nxt[ok]) / close[ok])
            )
        sleeve_ok = mask & ok
        if int(sleeve_ok.sum()):
            empty["mae_usd"] = float(np.mean(np.abs(implied[sleeve_ok] - nxt[sleeve_ok])))
            empty["mae_pct"] = float(
                np.mean(np.abs(implied[sleeve_ok] - nxt[sleeve_ok]) / close[sleeve_ok])
            )
    if int(moved.sum()) == 0:
        return empty
    up = float((r[moved] > 0).mean())
    daily = (
        pd.DataFrame({"date": dates[mask], "r": r[mask]})
        .groupby("date")["r"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    daily = daily[np.isfinite(daily)]
    paper_ir = float("nan")
    if daily.size >= 5:
        sd = float(daily.std(ddof=1)) if daily.size > 1 else 0.0
        if sd > 1e-12:
            paper_ir = float(daily.mean() / sd * math.sqrt(252.0))
    return {
        **empty,
        "up_pct": float(100.0 * up),
        "excess_pp": (
            float(100.0 * (up - uncond_up)) if np.isfinite(uncond_up) else float("nan")
        ),
        "paper_ir": paper_ir,
    }


def fit_book_aligned_on_train(
    df: pd.DataFrame,
    *,
    min_names: int,
) -> dict[str, Any]:
    """Select (q, |pred| floor) on TRAIN only. VAL/TEST never enter."""
    mag = np.abs(df["pred"].to_numpy(dtype=np.float64)) if not df.empty else np.array([])
    mag = mag[np.isfinite(mag)]
    rows: list[dict[str, Any]] = []
    for q in BOOK_ALIGN_QS:
        for aq in BOOK_ALIGN_ABS_QS:
            tau = 0.0 if float(aq) <= 0.0 else float(np.quantile(mag, float(aq))) if mag.size else 0.0
            row = score_book_aligned_sleeve(
                df, q=float(q), abs_tau=tau, min_names=min_names
            )
            row["abs_q"] = float(aq)
            if not np.isfinite(_as_float(row.get("up_pct"))):
                continue
            cover = _as_float(row.get("coverage"))
            if np.isfinite(cover) and cover < BOOK_ALIGN_COVER:
                continue
            rows.append(row)
    chosen: dict[str, Any] = {}
    best_key = (-1e18, -1e18, -1.0)
    for row in rows:
        xs = _as_float(row.get("excess_pp"))
        up = _as_float(row.get("up_pct"))
        q = _as_float(row.get("q"))
        if not np.isfinite(xs):
            continue
        key = (xs, up, q)
        if key > best_key:
            best_key = key
            chosen = dict(row)
    baseline = score_book_aligned_sleeve(df, q=0.80, abs_tau=0.0, min_names=min_names)
    baseline["abs_q"] = 0.0
    if not chosen:
        chosen = dict(baseline)
    return {
        "rows": rows,
        "chosen": chosen,
        "baseline": baseline,
        "fit_split": "train",
        "qs": list(BOOK_ALIGN_QS),
        "abs_qs": list(BOOK_ALIGN_ABS_QS),
        "note": (
            "Overnight up-rate of within-date top-q residual skip names "
            "(optional TRAIN |pred| floor). q=0.80 is the liquid top-20% book. "
            "Pooled TS direction is report-only."
        ),
    }


def book_aligned_grid(
    df: pd.DataFrame,
    *,
    train_pred: np.ndarray,
    min_names: int,
) -> list[dict[str, Any]]:
    """Score the q × |pred| grid with TRAIN-only magnitude thresholds."""
    mag = np.abs(np.asarray(train_pred, dtype=np.float64))
    mag = mag[np.isfinite(mag)]
    rows: list[dict[str, Any]] = []
    for q in BOOK_ALIGN_QS:
        for aq in BOOK_ALIGN_ABS_QS:
            tau = (
                0.0
                if float(aq) <= 0.0
                else (float(np.quantile(mag, float(aq))) if mag.size else 0.0)
            )
            row = score_book_aligned_sleeve(
                df, q=float(q), abs_tau=tau, min_names=min_names
            )
            row["abs_q"] = float(aq)
            rows.append(row)
    return rows


def decide_book_aligned_promote(
    *,
    val_chosen: dict[str, Any],
    val_top20: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only vs uncond overnight-up floor and the q=0.80 top-20% sleeve."""
    scored = dict(val_chosen or {})
    base = dict(val_top20 or {})
    up = _as_float(scored.get("up_pct"))
    floor = _as_float(scored.get("uncond_up_pct"))
    xs = _as_float(scored.get("excess_pp"))
    if not np.isfinite(xs) and np.isfinite(up) and np.isfinite(floor):
        xs = float(up - floor)
    up20 = _as_float(base.get("up_pct"))
    cover = _as_float(scored.get("coverage"))
    q = _as_float((chosen or {}).get("q"), default=0.80)
    abs_tau = _as_float((chosen or {}).get("abs_tau"), default=0.0)
    abs_q = _as_float((chosen or {}).get("abs_q"), default=0.0)
    same = abs(q - 0.80) < 1e-12 and abs(abs_tau) <= 1e-15
    floor_ok = bool(np.isfinite(xs) and xs >= BOOK_UP_FLOOR_PP)
    vs_base = bool(np.isfinite(up) and np.isfinite(up20) and up >= up20 + BOOK_UP_BASE_PP)
    cover_ok = bool(not np.isfinite(cover) or cover >= BOOK_ALIGN_COVER)
    promote = bool((not same) and floor_ok and vs_base and cover_ok)
    if same:
        reason = (
            "NO PROMOTE: TRAIN chose q=0.80 / no |pred| floor "
            f"(current top-20% book). VAL up {up:+.2f}% vs floor {floor:.2f}% "
            f"(xs {xs:+.2f} pp)."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE: TRAIN q={q:.2f} abs_q={abs_q:.2f} but VAL cover "
            f"{100 * cover:.1f}% < {100 * BOOK_ALIGN_COVER:.0f}%."
        )
    elif not floor_ok:
        reason = (
            f"NO PROMOTE: TRAIN q={q:.2f} VAL up {up:.2f}% vs floor {floor:.2f}% "
            f"(xs {xs:+.2f} pp < +{BOOK_UP_FLOOR_PP:.1f} pp)."
        )
    elif not vs_base:
        reason = (
            f"NO PROMOTE: TRAIN q={q:.2f} VAL up {up:.2f}% vs top-20% "
            f"{up20:.2f}% (delta {up - up20:+.2f} pp < +{BOOK_UP_BASE_PP:.1f} pp)."
        )
    else:
        reason = (
            f"PROMOTE book-aligned q={q:.2f} abs_q={abs_q:.2f}: VAL up {up:.2f}% "
            f"vs floor {floor:.2f}% (xs {xs:+.2f} pp) and vs top-20% {up20:.2f}% "
            f"(delta {up - up20:+.2f} pp)."
        )
    return {
        "promote_book_aligned": promote,
        "gated_on": "val",
        "reason": reason,
        "spec": (
            {"q": float(q), "abs_tau": float(abs_tau), "abs_q": float(abs_q)}
            if promote
            else {"q": 0.80, "abs_tau": 0.0, "abs_q": 0.0}
        ),
        "chosen": scored,
        "baseline": base,
        "train_q": float(q),
        "train_abs_q": float(abs_q),
        "val_up": up,
        "val_floor": floor,
        "val_excess_pp": xs,
        "val_top20_up": up20,
        "val_vs_top20_pp": (
            float(up - up20) if np.isfinite(up) and np.isfinite(up20) else float("nan")
        ),
        "coverage": cover,
        "floor_lift_pp": BOOK_UP_FLOOR_PP,
        "base_lift_pp": BOOK_UP_BASE_PP,
        "mae_usd": _as_float(scored.get("mae_usd")),
        "mae_pct": _as_float(scored.get("mae_pct")),
        "full_mae_usd": _as_float(scored.get("full_mae_usd")),
        "full_mae_pct": _as_float(scored.get("full_mae_pct")),
        "paper_ir": _as_float(scored.get("paper_ir")),
    }


def cs_bottom_abs_mask(
    df: pd.DataFrame,
    *,
    q: float,
    abs_tau: float = 0.0,
    score_col: str = "pred",
    min_names: int = 3,
) -> np.ndarray:
    """Within-date bottom residual names, optional causal |pred| floor.

    ``q=0.20`` is the bottom 20%. ``abs_tau`` is a TRAIN quantile of ``|pred|``
    (0 = off). Next open is never used.
    """
    n = len(df)
    out = np.zeros(n, dtype=bool)
    if df.empty or score_col not in df.columns:
        return out
    dates = df["date"].to_numpy(dtype=np.int64)
    score = df[score_col].to_numpy(dtype=np.float64)
    mag = np.abs(score)
    tau = float(abs_tau)
    for key in np.unique(dates):
        sel = dates == key
        row = score[sel]
        finite = np.isfinite(row)
        if int(finite.sum()) < int(min_names):
            continue
        cut = float(np.nanquantile(row, float(q)))
        keep = finite & (row <= cut)
        if tau > 0.0:
            keep = keep & (mag[sel] >= tau)
        out[sel] = keep
    return out


def score_short_aligned_sleeve(
    df: pd.DataFrame,
    *,
    q: float,
    abs_tau: float = 0.0,
    min_names: int = 3,
) -> dict[str, Any]:
    """Overnight down-rate + sleeve MAE on the CS bottom-q residual names."""
    bot_pct = float(100.0 * float(q))
    name = f"book_bottom{int(round(bot_pct))}"
    if float(abs_tau) > 0.0:
        name = f"{name}_abs"
    empty = {
        "name": name,
        "q": float(q),
        "abs_tau": float(abs_tau),
        "bottom_pct": bot_pct,
        "down_pct": float("nan"),
        "uncond_down_pct": float("nan"),
        "excess_pp": float("nan"),
        "n": 0.0,
        "n_dates": 0.0,
        "coverage": float("nan"),
        "mae_usd": float("nan"),
        "mae_pct": float("nan"),
        "full_mae_usd": float("nan"),
        "full_mae_pct": float("nan"),
        "paper_ir": float("nan"),
    }
    if df.empty or "pred" not in df.columns or "r_on" not in df.columns:
        return empty
    mask = cs_bottom_abs_mask(
        df, q=float(q), abs_tau=float(abs_tau), score_col="pred", min_names=min_names
    )
    r = df["r_on"].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy(dtype=np.int64)
    moved = mask & np.isfinite(r) & (r != 0.0)
    uncond = r[np.isfinite(r) & (r != 0.0)]
    uncond_down = float((uncond < 0).mean()) if uncond.size else float("nan")
    n_dates = float(pd.Series(dates[mask]).nunique()) if int(mask.sum()) else 0.0
    cover = float(mask.mean()) if mask.size else float("nan")
    empty["n"] = float(int(moved.sum()))
    empty["n_dates"] = n_dates
    empty["coverage"] = cover
    empty["uncond_down_pct"] = (
        float(100.0 * uncond_down) if np.isfinite(uncond_down) else float("nan")
    )
    if "close" in df.columns and "next_open" in df.columns:
        close = df["close"].to_numpy(dtype=np.float64)
        nxt = df["next_open"].to_numpy(dtype=np.float64)
        if "implied_open" in df.columns:
            implied = df["implied_open"].to_numpy(dtype=np.float64)
        else:
            implied = close
        ok = np.isfinite(close) & np.isfinite(nxt) & (close > 0)
        if int(ok.sum()):
            empty["full_mae_usd"] = float(np.mean(np.abs(implied[ok] - nxt[ok])))
            empty["full_mae_pct"] = float(
                np.mean(np.abs(implied[ok] - nxt[ok]) / close[ok])
            )
        sleeve_ok = mask & ok
        if int(sleeve_ok.sum()):
            empty["mae_usd"] = float(np.mean(np.abs(implied[sleeve_ok] - nxt[sleeve_ok])))
            empty["mae_pct"] = float(
                np.mean(np.abs(implied[sleeve_ok] - nxt[sleeve_ok]) / close[sleeve_ok])
            )
    if int(moved.sum()) == 0:
        return empty
    down = float((r[moved] < 0).mean())
    daily = (
        pd.DataFrame({"date": dates[mask], "r": -r[mask]})
        .groupby("date")["r"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    daily = daily[np.isfinite(daily)]
    paper_ir = float("nan")
    if daily.size >= 5:
        sd = float(daily.std(ddof=1)) if daily.size > 1 else 0.0
        if sd > 1e-12:
            paper_ir = float(daily.mean() / sd * math.sqrt(252.0))
    return {
        **empty,
        "down_pct": float(100.0 * down),
        "excess_pp": (
            float(100.0 * (down - uncond_down))
            if np.isfinite(uncond_down)
            else float("nan")
        ),
        "paper_ir": paper_ir,
    }


def fit_short_aligned_on_train(
    df: pd.DataFrame,
    *,
    min_names: int,
) -> dict[str, Any]:
    """Select (bottom-q, |pred| floor) on TRAIN only. VAL/TEST never enter."""
    mag = np.abs(df["pred"].to_numpy(dtype=np.float64)) if not df.empty else np.array([])
    mag = mag[np.isfinite(mag)]
    rows: list[dict[str, Any]] = []
    for q in SHORT_ALIGN_QS:
        for aq in SHORT_ALIGN_ABS_QS:
            tau = (
                0.0
                if float(aq) <= 0.0
                else (float(np.quantile(mag, float(aq))) if mag.size else 0.0)
            )
            row = score_short_aligned_sleeve(
                df, q=float(q), abs_tau=tau, min_names=min_names
            )
            row["abs_q"] = float(aq)
            if not np.isfinite(_as_float(row.get("down_pct"))):
                continue
            cover = _as_float(row.get("coverage"))
            if np.isfinite(cover) and cover < BOOK_ALIGN_COVER:
                continue
            rows.append(row)
    chosen: dict[str, Any] = {}
    best_key = (-1e18, -1e18, -1.0)
    for row in rows:
        xs = _as_float(row.get("excess_pp"))
        down = _as_float(row.get("down_pct"))
        q = _as_float(row.get("q"))
        if not np.isfinite(xs):
            continue
        key = (xs, down, -q)
        if key > best_key:
            best_key = key
            chosen = dict(row)
    baseline = score_short_aligned_sleeve(df, q=0.20, abs_tau=0.0, min_names=min_names)
    baseline["abs_q"] = 0.0
    if not chosen:
        chosen = dict(baseline)
    return {
        "rows": rows,
        "chosen": chosen,
        "baseline": baseline,
        "fit_split": "train",
        "qs": list(SHORT_ALIGN_QS),
        "abs_qs": list(SHORT_ALIGN_ABS_QS),
        "note": (
            "Overnight down-rate of within-date bottom-q residual skip names "
            "(optional TRAIN |pred| floor). q=0.20 is the bottom-20% sleeve. "
            "Pooled TS direction is report-only. Live q20 unchanged on hit-rate-only."
        ),
    }


def short_aligned_grid(
    df: pd.DataFrame,
    *,
    train_pred: np.ndarray,
    min_names: int,
) -> list[dict[str, Any]]:
    """Score the bottom-q × |pred| grid with TRAIN-only magnitude thresholds."""
    mag = np.abs(np.asarray(train_pred, dtype=np.float64))
    mag = mag[np.isfinite(mag)]
    rows: list[dict[str, Any]] = []
    for q in SHORT_ALIGN_QS:
        for aq in SHORT_ALIGN_ABS_QS:
            tau = (
                0.0
                if float(aq) <= 0.0
                else (float(np.quantile(mag, float(aq))) if mag.size else 0.0)
            )
            row = score_short_aligned_sleeve(
                df, q=float(q), abs_tau=tau, min_names=min_names
            )
            row["abs_q"] = float(aq)
            rows.append(row)
    return rows


def decide_short_aligned_promote(
    *,
    val_chosen: dict[str, Any],
    val_bot20: dict[str, Any],
    chosen: dict[str, Any],
    val_live_ls: dict[str, Any] | None = None,
    val_live_q20: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """VAL-only short down-rate vs down-floor / bottom-20%, plus optional live IR."""
    scored = dict(val_chosen or {})
    base = dict(val_bot20 or {})
    live = dict(val_live_ls or {})
    q20 = dict(val_live_q20 or {})
    down = _as_float(scored.get("down_pct"))
    floor = _as_float(scored.get("uncond_down_pct"))
    xs = _as_float(scored.get("excess_pp"))
    if not np.isfinite(xs) and np.isfinite(down) and np.isfinite(floor):
        xs = float(down - floor)
    down20 = _as_float(base.get("down_pct"))
    cover = _as_float(scored.get("coverage"))
    q = _as_float((chosen or {}).get("q"), default=0.20)
    abs_tau = _as_float((chosen or {}).get("abs_tau"), default=0.0)
    abs_q = _as_float((chosen or {}).get("abs_q"), default=0.0)
    same = abs(q - 0.20) < 1e-12 and abs(abs_tau) <= 1e-15
    floor_ok = bool(np.isfinite(xs) and xs >= BOOK_UP_FLOOR_PP)
    vs_base = bool(
        np.isfinite(down) and np.isfinite(down20) and down >= down20 + BOOK_UP_BASE_PP
    )
    cover_ok = bool(not np.isfinite(cover) or cover >= BOOK_ALIGN_COVER)
    promote_hit = bool((not same) and floor_ok and vs_base and cover_ok)

    ir = _as_float(live.get("unlevered_net_ir"))
    ir20 = _as_float(q20.get("unlevered_net_ir"))
    ir_delta = (
        float(ir - ir20) if np.isfinite(ir) and np.isfinite(ir20) else float("nan")
    )
    live_cover = _as_float(live.get("coverage"))
    if not np.isfinite(live_cover):
        live_cover = cover
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= SHORT_IR_LIFT)
    live_cover_ok = bool(np.isfinite(live_cover) and live_cover >= BOOK_ALIGN_COVER)
    promote_live = bool(ir_ok and live_cover_ok)

    if same:
        hit_reason = (
            "NO PROMOTE short-aligned: TRAIN chose q=0.20 / no |pred| floor "
            f"(current bottom-20% sleeve). VAL down {down:.2f}% vs floor "
            f"{floor:.2f}% (xs {xs:+.2f} pp)."
        )
    elif not cover_ok:
        hit_reason = (
            f"NO PROMOTE short-aligned: TRAIN q={q:.2f} abs_q={abs_q:.2f} but "
            f"VAL cover {100.0 * cover:.1f}% < {100.0 * BOOK_ALIGN_COVER:.0f}%."
        )
    elif not floor_ok:
        hit_reason = (
            f"NO PROMOTE short-aligned: TRAIN q={q:.2f} VAL down {down:.2f}% vs "
            f"floor {floor:.2f}% (xs {xs:+.2f} pp < +{BOOK_UP_FLOOR_PP:.1f} pp)."
        )
    elif not vs_base:
        hit_reason = (
            f"NO PROMOTE short-aligned: TRAIN q={q:.2f} VAL down {down:.2f}% vs "
            f"bottom-20% {down20:.2f}% (delta {down - down20:+.2f} pp "
            f"< +{BOOK_UP_BASE_PP:.1f} pp)."
        )
    else:
        hit_reason = (
            f"PROMOTE short-aligned q={q:.2f} abs_q={abs_q:.2f}: VAL down "
            f"{down:.2f}% vs floor {floor:.2f}% (xs {xs:+.2f} pp) and vs "
            f"bottom-20% {down20:.2f}% (delta {down - down20:+.2f} pp)."
        )

    if not live_cover_ok:
        live_reason = (
            f"NO PROMOTE short live LS: VAL cover {100.0 * live_cover:.1f}% "
            f"< {100.0 * BOOK_ALIGN_COVER:.0f}%. Keep q20 default."
        )
    elif not ir_ok:
        live_reason = (
            f"NO PROMOTE short live LS: VAL live_locate IR {ir:+.3f} vs q20 "
            f"{ir20:+.3f} (delta {ir_delta:+.3f} < +{SHORT_IR_LIFT:.2f}). "
            "Keep q20 default. Hit-rate-only."
        )
    else:
        live_reason = (
            f"PROMOTE short live LS q={q:.2f} abs_q={abs_q:.2f}: VAL "
            f"live_locate IR {ir:+.3f} vs q20 {ir20:+.3f} "
            f"(delta {ir_delta:+.3f}). Default CLI stays q20 until liquid."
        )
    return {
        "promote_short_aligned": promote_hit,
        "promote_short_live": promote_live,
        "gated_on": "val",
        "reason": hit_reason,
        "hit_reason": hit_reason,
        "live_reason": live_reason,
        "spec": (
            {"q": float(q), "abs_tau": float(abs_tau), "abs_q": float(abs_q)}
            if promote_hit
            else {"q": 0.20, "abs_tau": 0.0, "abs_q": 0.0}
        ),
        "chosen": scored,
        "baseline": base,
        "train_q": float(q),
        "train_abs_q": float(abs_q),
        "train_abs_tau": float(abs_tau),
        "val_down": down,
        "val_floor": floor,
        "val_excess_pp": xs,
        "val_bot20_down": down20,
        "val_vs_bot20_pp": (
            float(down - down20)
            if np.isfinite(down) and np.isfinite(down20)
            else float("nan")
        ),
        "coverage": cover,
        "floor_lift_pp": BOOK_UP_FLOOR_PP,
        "base_lift_pp": BOOK_UP_BASE_PP,
        "val_ir": ir,
        "val_ir_q20": ir20,
        "val_ir_delta": ir_delta,
        "ir_lift": SHORT_IR_LIFT,
        "live_book_unchanged": (not promote_live),
        "default_book_unchanged": True,
        "mae_usd": _as_float(scored.get("mae_usd")),
        "mae_pct": _as_float(scored.get("mae_pct")),
        "paper_ir": _as_float(scored.get("paper_ir")),
    }


def decide_ej_ls_promote(
    *,
    val_live: dict[str, Any],
    val_q20: dict[str, Any],
    val_paper: dict[str, Any] | None = None,
    e_chosen: dict[str, Any] | None = None,
    j_chosen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """VAL-only live_locate E+J LS vs long-only q20. Paper is report-only."""
    live = dict(val_live or {})
    q20 = dict(val_q20 or {})
    paper = dict(val_paper or {})
    e = dict(e_chosen or {})
    j = dict(j_chosen or {})
    ir = _as_float(live.get("unlevered_net_ir"))
    ir20 = _as_float(q20.get("unlevered_net_ir"))
    dd = _as_float(live.get("unlevered_max_dd"))
    dd20 = _as_float(q20.get("unlevered_max_dd"))
    cover = _as_float(live.get("coverage"))
    ir_delta = (
        float(ir - ir20) if np.isfinite(ir) and np.isfinite(ir20) else float("nan")
    )
    dd_delta = (
        float(dd - dd20) if np.isfinite(dd) and np.isfinite(dd20) else float("nan")
    )
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= EJ_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -EJ_DD_TOL)
    cover_ok = bool(np.isfinite(cover) and cover >= EJ_COVER)
    promote = bool(ir_ok and dd_ok and cover_ok)
    e_q = _as_float(e.get("q"), default=0.80)
    e_aq = _as_float(e.get("abs_q"), default=0.0)
    j_q = _as_float(j.get("q"), default=0.20)
    j_aq = _as_float(j.get("abs_q"), default=0.0)
    if not cover_ok:
        reason = (
            f"NO PROMOTE E+J live LS: VAL cover {100.0 * cover:.1f}% "
            f"< {100.0 * EJ_COVER:.0f}%. Keep q20 default. E+J remain hit-rate-only."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE E+J live LS: VAL live_locate IR {ir:+.3f} vs q20 "
            f"{ir20:+.3f} (delta {ir_delta:+.3f} < +{EJ_IR_LIFT:.2f}). "
            "Keep q20 default. E+J remain hit-rate-only."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE E+J live LS: VAL max DD {dd:+.3f} vs q20 {dd20:+.3f} "
            f"(delta {dd_delta:+.3f} < -{EJ_DD_TOL:.2f}). Keep q20 default. "
            "E+J remain hit-rate-only."
        )
    else:
        reason = (
            f"PROMOTE E+J live LS e_q={e_q:.2f} j_q={j_q:.2f}: VAL live_locate "
            f"IR {ir:+.3f} vs q20 {ir20:+.3f} (delta {ir_delta:+.3f}) and max DD "
            f"{dd:+.3f} vs {dd20:+.3f} (delta {dd_delta:+.3f}). "
            "Default CLI stays q20 until liquid."
        )
    return {
        "promote_ej_ls": promote,
        "gated_on": "val",
        "reason": reason,
        "e_q": float(e_q),
        "e_abs_q": float(e_aq),
        "e_abs_tau": _as_float(e.get("abs_tau"), default=0.0),
        "j_q": float(j_q),
        "j_abs_q": float(j_aq),
        "j_abs_tau": _as_float(j.get("abs_tau"), default=0.0),
        "val_ir": ir,
        "val_ir_q20": ir20,
        "val_ir_delta": ir_delta,
        "val_dd": dd,
        "val_dd_q20": dd20,
        "val_dd_delta": dd_delta,
        "val_coverage": cover,
        "val_paper_ir": _as_float(paper.get("unlevered_net_ir")),
        "val_paper_dd": _as_float(paper.get("unlevered_max_dd")),
        "ir_lift": EJ_IR_LIFT,
        "dd_tol": EJ_DD_TOL,
        "cover_floor": EJ_COVER,
        "live_book_unchanged": (not promote),
        "default_book_unchanged": True,
    }


def cs_relative_blocks(
    df: pd.DataFrame,
    *,
    score_col: str = "pred",
    min_names: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Within-date ``pred − CS median`` / ``r_on − CS median`` and long-half mask.

    Dates with fewer than ``min_names`` are dropped. Next open is never used.
    """
    n = len(df)
    pred_rel = np.full(n, np.nan, dtype=np.float64)
    r_rel = np.full(n, np.nan, dtype=np.float64)
    abs_dev = np.full(n, np.nan, dtype=np.float64)
    long_half = np.zeros(n, dtype=bool)
    date_ok = np.zeros(n, dtype=bool)
    if df.empty or score_col not in df.columns or "r_on" not in df.columns:
        return pred_rel, r_rel, abs_dev, long_half, date_ok
    dates = df["date"].to_numpy(dtype=np.int64)
    score = df[score_col].to_numpy(dtype=np.float64)
    r = df["r_on"].to_numpy(dtype=np.float64)
    for key in np.unique(dates):
        sel = dates == key
        row_p = score[sel]
        row_r = r[sel]
        finite_p = np.isfinite(row_p)
        if int(finite_p.sum()) < int(min_names):
            continue
        date_ok[sel] = True
        pmed = float(np.nanmedian(row_p))
        rmed = float(np.nanmedian(row_r)) if np.isfinite(row_r).any() else float("nan")
        pred_rel[sel] = row_p - pmed
        r_rel[sel] = row_r - rmed
        abs_dev[sel] = np.abs(row_p - pmed)
        long_half[sel] = finite_p & (row_p > pmed)
    return pred_rel, r_rel, abs_dev, long_half, date_ok


def score_relative_direction(
    df: pd.DataFrame,
    *,
    abs_tau: float = 0.0,
    score_col: str = "pred",
    min_names: int = 3,
) -> dict[str, Any]:
    """Relative CS-median sign hit vs 50%, plus long-half absolute overnight-up."""
    empty = {
        "abs_tau": float(abs_tau),
        "rel_hit_pct": float("nan"),
        "rel_excess_pp": float("nan"),
        "rel_z": float("nan"),
        "rel_p": float("nan"),
        "rel_n": 0.0,
        "rel_coverage": float("nan"),
        "rel_n_dates": 0.0,
        "long_up_pct": float("nan"),
        "long_uncond_up_pct": float("nan"),
        "long_excess_pp": float("nan"),
        "long_n": 0.0,
        "long_coverage": float("nan"),
        "long_n_dates": 0.0,
    }
    if df.empty:
        return empty
    pred_rel, r_rel, abs_dev, long_half, date_ok = cs_relative_blocks(
        df, score_col=score_col, min_names=min_names
    )
    rel_ok = (
        date_ok
        & np.isfinite(pred_rel)
        & np.isfinite(r_rel)
        & (pred_rel != 0.0)
        & (r_rel != 0.0)
    )
    long_ok = date_ok & long_half
    tau = float(abs_tau)
    if tau > 0.0:
        rel_ok = rel_ok & np.isfinite(abs_dev) & (abs_dev >= tau)
        long_ok = long_ok & np.isfinite(abs_dev) & (abs_dev >= tau)
    hits = (np.sign(pred_rel[rel_ok]) == np.sign(r_rel[rel_ok])).astype(np.float64)
    inf = hit_rate_inference(hits)
    dates = df["date"].to_numpy(dtype=np.int64)
    r = df["r_on"].to_numpy(dtype=np.float64)
    moved = np.isfinite(r) & (r != 0.0)
    uncond = r[moved]
    uncond_up = float((uncond > 0).mean()) if uncond.size else float("nan")
    long_moved = long_ok & moved
    up = float((r[long_moved] > 0).mean()) if int(long_moved.sum()) else float("nan")
    rel_dates = float(pd.Series(dates[rel_ok]).nunique()) if int(rel_ok.sum()) else 0.0
    long_dates = float(pd.Series(dates[long_ok]).nunique()) if int(long_ok.sum()) else 0.0
    return {
        **empty,
        "rel_hit_pct": _as_float(inf.get("hit_rate_pct")),
        "rel_excess_pp": (
            _as_float(inf.get("hit_rate_pct")) - 50.0
            if np.isfinite(_as_float(inf.get("hit_rate_pct")))
            else float("nan")
        ),
        "rel_z": _as_float(inf.get("z_vs_half")),
        "rel_p": _as_float(inf.get("p_vs_half")),
        "rel_n": _as_float(inf.get("n"), default=0.0),
        "rel_coverage": float(rel_ok.mean()) if rel_ok.size else float("nan"),
        "rel_n_dates": rel_dates,
        "long_up_pct": float(100.0 * up) if np.isfinite(up) else float("nan"),
        "long_uncond_up_pct": (
            float(100.0 * uncond_up) if np.isfinite(uncond_up) else float("nan")
        ),
        "long_excess_pp": (
            float(100.0 * (up - uncond_up))
            if np.isfinite(up) and np.isfinite(uncond_up)
            else float("nan")
        ),
        "long_n": float(int(long_moved.sum())),
        "long_coverage": float(long_ok.mean()) if long_ok.size else float("nan"),
        "long_n_dates": long_dates,
    }


def fit_relative_dir_on_train(
    df: pd.DataFrame,
    *,
    min_names: int,
) -> dict[str, Any]:
    """Select optional |pred−CS median| floor on TRAIN only."""
    pred_rel, _r_rel, abs_dev, _lh, date_ok = cs_relative_blocks(
        df, score_col="pred", min_names=min_names
    )
    del pred_rel
    mag = abs_dev[date_ok & np.isfinite(abs_dev)]
    rows: list[dict[str, Any]] = []
    for aq in REL_ABS_QS:
        tau = (
            0.0
            if float(aq) <= 0.0
            else (float(np.quantile(mag, float(aq))) if mag.size else 0.0)
        )
        row = score_relative_direction(
            df, abs_tau=tau, score_col="pred", min_names=min_names
        )
        row["abs_q"] = float(aq)
        if not np.isfinite(_as_float(row.get("rel_hit_pct"))):
            continue
        cover = _as_float(row.get("rel_coverage"))
        if np.isfinite(cover) and cover < REL_COVER:
            continue
        rows.append(row)
    chosen: dict[str, Any] = {}
    best_key = (-1e18, -1e18, -1.0)
    for row in rows:
        hit = _as_float(row.get("rel_hit_pct"))
        xs = _as_float(row.get("rel_excess_pp"))
        cover = _as_float(row.get("rel_coverage"))
        if not np.isfinite(hit):
            continue
        key = (hit, xs, cover)
        if key > best_key:
            best_key = key
            chosen = dict(row)
    full = score_relative_direction(df, abs_tau=0.0, score_col="pred", min_names=min_names)
    full["abs_q"] = 0.0
    if not chosen:
        chosen = dict(full)
    return {
        "rows": rows,
        "chosen": chosen,
        "full": full,
        "fit_split": "train",
        "abs_qs": list(REL_ABS_QS),
        "score_col": "pred",
        "note": (
            "Within-date sign(pred − CS median) vs sign(r_on − CS median). "
            "Optional TRAIN |pred−median| floor. Long-half = pred > CS median. "
            "Pooled TS direction is report-only. Live q20 unchanged."
        ),
    }


def decide_relative_dir_promote(
    *,
    val_chosen: dict[str, Any],
    val_full: dict[str, Any],
    val_top20: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only relative-dir vs 50% and long-half overnight-up vs floor/top-20%."""
    rel = dict(val_chosen or {})
    half = dict(val_full or {})
    base = dict(val_top20 or {})
    hit = _as_float(rel.get("rel_hit_pct"))
    z = _as_float(rel.get("rel_z"))
    cover = _as_float(rel.get("rel_coverage"))
    abs_tau = _as_float((chosen or {}).get("abs_tau"), default=0.0)
    abs_q = _as_float((chosen or {}).get("abs_q"), default=0.0)
    hit_ok = bool(np.isfinite(hit) and hit >= 50.0 + REL_DIR_LIFT_PP)
    z_ok = bool(np.isfinite(z) and z >= REL_DIR_Z)
    cover_ok = bool(not np.isfinite(cover) or cover >= REL_COVER)
    promote_rel = bool(hit_ok and z_ok and cover_ok)

    long_up = _as_float(half.get("long_up_pct"))
    long_floor = _as_float(half.get("long_uncond_up_pct"))
    long_xs = _as_float(half.get("long_excess_pp"))
    if not np.isfinite(long_xs) and np.isfinite(long_up) and np.isfinite(long_floor):
        long_xs = float(long_up - long_floor)
    up20 = _as_float(base.get("up_pct"))
    long_cover = _as_float(half.get("long_coverage"))
    floor_ok = bool(np.isfinite(long_xs) and long_xs >= BOOK_UP_FLOOR_PP)
    vs_base = bool(
        np.isfinite(long_up) and np.isfinite(up20) and long_up >= up20 + BOOK_UP_BASE_PP
    )
    long_cover_ok = bool(not np.isfinite(long_cover) or long_cover >= REL_COVER)
    promote_long = bool(floor_ok and vs_base and long_cover_ok)
    weaker_than_e = bool(np.isfinite(long_up) and np.isfinite(up20) and long_up < up20)

    if not cover_ok:
        rel_reason = (
            f"NO PROMOTE relative-dir: VAL cover {100.0 * cover:.1f}% "
            f"< {100.0 * REL_COVER:.0f}%."
        )
    elif not hit_ok:
        rel_reason = (
            f"NO PROMOTE relative-dir: VAL hit {hit:.2f}% "
            f"< 50%+{REL_DIR_LIFT_PP:.1f}pp."
        )
    elif not z_ok:
        rel_reason = (
            f"NO PROMOTE relative-dir: VAL hit {hit:.2f}% but z={z:.2f} "
            f"< {REL_DIR_Z:.1f} (not z-sensible)."
        )
    else:
        rel_reason = (
            f"PROMOTE relative-dir abs_q={abs_q:.2f}: VAL hit {hit:.2f}% "
            f"(xs {hit - 50.0:+.2f} pp vs 50%, z={z:.2f})."
        )

    if not long_cover_ok:
        long_reason = (
            f"NO PROMOTE long-half up: VAL cover {100.0 * long_cover:.1f}% "
            f"< {100.0 * REL_COVER:.0f}%."
        )
    elif not floor_ok:
        long_reason = (
            f"NO PROMOTE long-half up: VAL up {long_up:.2f}% vs floor "
            f"{long_floor:.2f}% (xs {long_xs:+.2f} pp < +{BOOK_UP_FLOOR_PP:.1f} pp)."
        )
    elif not vs_base:
        note = (
            " weaker than top-20% / IDEA E book sleeve."
            if weaker_than_e
            else ""
        )
        long_reason = (
            f"NO PROMOTE long-half up: VAL up {long_up:.2f}% vs top-20% "
            f"{up20:.2f}% (delta {long_up - up20:+.2f} pp < +{BOOK_UP_BASE_PP:.1f} pp)."
            + note
        )
    else:
        long_reason = (
            f"PROMOTE long-half up: VAL up {long_up:.2f}% vs floor "
            f"{long_floor:.2f}% (xs {long_xs:+.2f} pp) and vs top-20% "
            f"{up20:.2f}% (delta {long_up - up20:+.2f} pp)."
        )
    return {
        "promote_relative_dir": promote_rel,
        "promote_long_half": promote_long,
        "gated_on": "val",
        "rel_reason": rel_reason,
        "long_reason": long_reason,
        "reason": f"{rel_reason} {long_reason}",
        "spec": (
            {"abs_tau": float(abs_tau), "abs_q": float(abs_q), "score_col": "pred"}
            if promote_rel
            else {"abs_tau": 0.0, "abs_q": 0.0, "score_col": "pred"}
        ),
        "chosen": rel,
        "full": half,
        "baseline": base,
        "train_abs_q": float(abs_q),
        "train_abs_tau": float(abs_tau),
        "val_rel_hit": hit,
        "val_rel_z": z,
        "val_rel_coverage": cover,
        "val_long_up": long_up,
        "val_long_floor": long_floor,
        "val_long_excess_pp": long_xs,
        "val_top20_up": up20,
        "val_vs_top20_pp": (
            float(long_up - up20)
            if np.isfinite(long_up) and np.isfinite(up20)
            else float("nan")
        ),
        "weaker_than_top20": weaker_than_e,
        "rel_lift_pp": REL_DIR_LIFT_PP,
        "rel_z_floor": REL_DIR_Z,
        "cover_floor": REL_COVER,
        "live_book_unchanged": True,
    }


def cs_stack_mask(
    df: pd.DataFrame,
    *,
    q: float,
    abs_tau: float = 0.0,
    score_col: str = "pred",
    min_names: int = 3,
) -> np.ndarray:
    """E top-q ∩ |pred| floor ∩ within-date ``pred > CS median``.

    Next open is never used.
    """
    top = cs_top_abs_mask(
        df, q=float(q), abs_tau=float(abs_tau), score_col=score_col, min_names=min_names
    )
    _pr, _rr, _ad, long_half, date_ok = cs_relative_blocks(
        df, score_col=score_col, min_names=min_names
    )
    del _pr, _rr, _ad
    return date_ok & long_half & top


def score_rel_e_stack(
    df: pd.DataFrame,
    *,
    q: float,
    abs_tau: float = 0.0,
    min_names: int = 3,
) -> dict[str, Any]:
    """Relative CS-median sign hit and absolute overnight-up on the H∩E stack."""
    empty = {
        "q": float(q),
        "abs_tau": float(abs_tau),
        "rel_hit_pct": float("nan"),
        "rel_excess_pp": float("nan"),
        "rel_z": float("nan"),
        "rel_p": float("nan"),
        "rel_n": 0.0,
        "rel_coverage": float("nan"),
        "rel_n_dates": 0.0,
        "long_up_pct": float("nan"),
        "long_uncond_up_pct": float("nan"),
        "long_excess_pp": float("nan"),
        "long_n": 0.0,
        "long_coverage": float("nan"),
        "long_n_dates": 0.0,
    }
    if df.empty:
        return empty
    pred_rel, r_rel, _abs_dev, _lh, date_ok = cs_relative_blocks(
        df, score_col="pred", min_names=min_names
    )
    mask = cs_stack_mask(
        df, q=float(q), abs_tau=float(abs_tau), score_col="pred", min_names=min_names
    )
    rel_ok = (
        mask
        & date_ok
        & np.isfinite(pred_rel)
        & np.isfinite(r_rel)
        & (pred_rel != 0.0)
        & (r_rel != 0.0)
    )
    long_ok = mask
    hits = (np.sign(pred_rel[rel_ok]) == np.sign(r_rel[rel_ok])).astype(np.float64)
    inf = hit_rate_inference(hits)
    dates = df["date"].to_numpy(dtype=np.int64)
    r = df["r_on"].to_numpy(dtype=np.float64)
    moved = np.isfinite(r) & (r != 0.0)
    uncond = r[moved]
    uncond_up = float((uncond > 0).mean()) if uncond.size else float("nan")
    long_moved = long_ok & moved
    up = float((r[long_moved] > 0).mean()) if int(long_moved.sum()) else float("nan")
    rel_dates = float(pd.Series(dates[rel_ok]).nunique()) if int(rel_ok.sum()) else 0.0
    long_dates = float(pd.Series(dates[long_ok]).nunique()) if int(long_ok.sum()) else 0.0
    return {
        **empty,
        "rel_hit_pct": _as_float(inf.get("hit_rate_pct")),
        "rel_excess_pp": (
            _as_float(inf.get("hit_rate_pct")) - 50.0
            if np.isfinite(_as_float(inf.get("hit_rate_pct")))
            else float("nan")
        ),
        "rel_z": _as_float(inf.get("z_vs_half")),
        "rel_p": _as_float(inf.get("p_vs_half")),
        "rel_n": _as_float(inf.get("n"), default=0.0),
        "rel_coverage": float(rel_ok.mean()) if rel_ok.size else float("nan"),
        "rel_n_dates": rel_dates,
        "long_up_pct": float(100.0 * up) if np.isfinite(up) else float("nan"),
        "long_uncond_up_pct": (
            float(100.0 * uncond_up) if np.isfinite(uncond_up) else float("nan")
        ),
        "long_excess_pp": (
            float(100.0 * (up - uncond_up))
            if np.isfinite(up) and np.isfinite(uncond_up)
            else float("nan")
        ),
        "long_n": float(int(long_moved.sum())),
        "long_coverage": float(long_ok.mean()) if long_ok.size else float("nan"),
        "long_n_dates": long_dates,
    }


def fit_rel_e_stack_on_train(
    df: pd.DataFrame,
    *,
    min_names: int,
    e_chosen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select (q, |pred| floor) ∩ long-half on TRAIN to max relative hit."""
    mag = np.abs(df["pred"].to_numpy(dtype=np.float64)) if not df.empty else np.array([])
    mag = mag[np.isfinite(mag)]
    candidates: list[tuple[float, float, float]] = []
    for q in STACK_QS:
        for aq in STACK_ABS_QS:
            tau = (
                0.0
                if float(aq) <= 0.0
                else (float(np.quantile(mag, float(aq))) if mag.size else 0.0)
            )
            candidates.append((float(q), float(aq), float(tau)))
    e_spec = dict(e_chosen or {})
    if e_spec:
        candidates.append(
            (
                float(e_spec.get("q") or 0.80),
                float(e_spec.get("abs_q") or 0.0),
                float(e_spec.get("abs_tau") or 0.0),
            )
        )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for q, aq, tau in candidates:
        key = (round(float(q), 6), round(float(tau), 8))
        if key in seen:
            continue
        seen.add(key)
        row = score_rel_e_stack(df, q=float(q), abs_tau=float(tau), min_names=min_names)
        row["abs_q"] = float(aq)
        if not np.isfinite(_as_float(row.get("rel_hit_pct"))):
            continue
        cover = _as_float(row.get("rel_coverage"))
        if np.isfinite(cover) and cover < REL_COVER:
            continue
        rows.append(row)
    chosen: dict[str, Any] = {}
    best_key = (-1e18, -1e18, -1.0)
    for row in rows:
        hit = _as_float(row.get("rel_hit_pct"))
        xs = _as_float(row.get("rel_excess_pp"))
        cover = _as_float(row.get("rel_coverage"))
        if not np.isfinite(hit):
            continue
        key = (hit, xs, cover)
        if key > best_key:
            best_key = key
            chosen = dict(row)
    e_q = float(e_spec.get("q") or 0.80)
    e_tau = float(e_spec.get("abs_tau") or 0.0)
    e_aq = float(e_spec.get("abs_q") or 0.0)
    e_ref = score_rel_e_stack(df, q=e_q, abs_tau=e_tau, min_names=min_names)
    e_ref["abs_q"] = e_aq
    if not chosen:
        chosen = dict(e_ref)
    return {
        "rows": rows,
        "chosen": chosen,
        "e_ref": e_ref,
        "e_chosen": {
            "q": e_q,
            "abs_q": e_aq,
            "abs_tau": e_tau,
        },
        "fit_split": "train",
        "qs": list(STACK_QS),
        "abs_qs": list(STACK_ABS_QS),
        "score_col": "pred",
        "note": (
            "H∩E stack: pred > CS median and within-date top-q residual "
            "(optional TRAIN |pred| floor). TRAIN picks max relative hit vs 50% "
            "with cover ≥ 5%. Absolute up vs E's sleeve. Live IR vs q20. "
            "Live q20 unchanged unless the F IR gate clears."
        ),
    }


def decide_rel_e_stack_promote(
    *,
    val_chosen: dict[str, Any],
    val_e: dict[str, Any],
    val_live_stack: dict[str, Any] | None = None,
    val_live_q20: dict[str, Any] | None = None,
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only relative hit, absolute up vs E, and optional live IR vs q20."""
    rel = dict(val_chosen or {})
    e_row = dict(val_e or {})
    live = dict(val_live_stack or {})
    q20 = dict(val_live_q20 or {})
    hit = _as_float(rel.get("rel_hit_pct"))
    z = _as_float(rel.get("rel_z"))
    cover = _as_float(rel.get("rel_coverage"))
    q = _as_float((chosen or {}).get("q"), default=0.80)
    abs_tau = _as_float((chosen or {}).get("abs_tau"), default=0.0)
    abs_q = _as_float((chosen or {}).get("abs_q"), default=0.0)
    hit_ok = bool(np.isfinite(hit) and hit >= 50.0 + REL_DIR_LIFT_PP)
    z_ok = bool(np.isfinite(z) and z >= REL_DIR_Z)
    cover_ok = bool(np.isfinite(cover) and cover >= REL_COVER)
    promote_rel = bool(hit_ok and z_ok and cover_ok)

    long_up = _as_float(rel.get("long_up_pct"))
    long_floor = _as_float(rel.get("long_uncond_up_pct"))
    long_xs = _as_float(rel.get("long_excess_pp"))
    if not np.isfinite(long_xs) and np.isfinite(long_up) and np.isfinite(long_floor):
        long_xs = float(long_up - long_floor)
    e_up = _as_float(e_row.get("up_pct"))
    if not np.isfinite(e_up):
        e_up = _as_float(e_row.get("long_up_pct"))
    long_cover = _as_float(rel.get("long_coverage"))
    floor_ok = bool(np.isfinite(long_xs) and long_xs >= BOOK_UP_FLOOR_PP)
    vs_e = bool(
        np.isfinite(long_up) and np.isfinite(e_up) and long_up >= e_up + BOOK_UP_BASE_PP
    )
    long_cover_ok = bool(not np.isfinite(long_cover) or long_cover >= REL_COVER)
    promote_abs = bool(floor_ok and vs_e and long_cover_ok)
    weaker_than_e = bool(np.isfinite(long_up) and np.isfinite(e_up) and long_up < e_up)

    ir = _as_float(live.get("unlevered_net_ir"))
    ir20 = _as_float(q20.get("unlevered_net_ir"))
    dd = _as_float(live.get("unlevered_max_dd"))
    dd20 = _as_float(q20.get("unlevered_max_dd"))
    live_cover = _as_float(live.get("coverage"))
    if not np.isfinite(live_cover):
        live_cover = cover
    ir_delta = (
        float(ir - ir20) if np.isfinite(ir) and np.isfinite(ir20) else float("nan")
    )
    dd_delta = (
        float(dd - dd20) if np.isfinite(dd) and np.isfinite(dd20) else float("nan")
    )
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= STACK_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -STACK_DD_TOL)
    live_cover_ok = bool(np.isfinite(live_cover) and live_cover >= REL_COVER)
    promote_live = bool(ir_ok and dd_ok and live_cover_ok)

    if not cover_ok:
        rel_reason = (
            f"NO PROMOTE stack relative-dir: VAL cover {100.0 * cover:.1f}% "
            f"< {100.0 * REL_COVER:.0f}%."
        )
    elif not hit_ok:
        rel_reason = (
            f"NO PROMOTE stack relative-dir: VAL hit {hit:.2f}% "
            f"< 50%+{REL_DIR_LIFT_PP:.1f}pp."
        )
    elif not z_ok:
        rel_reason = (
            f"NO PROMOTE stack relative-dir: VAL hit {hit:.2f}% but z={z:.2f} "
            f"< {REL_DIR_Z:.1f} (not z-sensible)."
        )
    else:
        rel_reason = (
            f"PROMOTE stack relative-dir q={q:.2f} abs_q={abs_q:.2f}: VAL hit "
            f"{hit:.2f}% (xs {hit - 50.0:+.2f} pp vs 50%, z={z:.2f})."
        )

    if not long_cover_ok:
        abs_reason = (
            f"NO PROMOTE stack absolute-up: VAL cover {100.0 * long_cover:.1f}% "
            f"< {100.0 * REL_COVER:.0f}%."
        )
    elif not floor_ok:
        abs_reason = (
            f"NO PROMOTE stack absolute-up: VAL up {long_up:.2f}% vs floor "
            f"{long_floor:.2f}% (xs {long_xs:+.2f} pp < +{BOOK_UP_FLOOR_PP:.1f} pp)."
        )
    elif not vs_e:
        note = " weaker than IDEA E sleeve." if weaker_than_e else ""
        abs_reason = (
            f"NO PROMOTE stack absolute-up: VAL up {long_up:.2f}% vs E "
            f"{e_up:.2f}% (delta {long_up - e_up:+.2f} pp < +{BOOK_UP_BASE_PP:.1f} pp)."
            + note
        )
    else:
        abs_reason = (
            f"PROMOTE stack absolute-up: VAL up {long_up:.2f}% vs floor "
            f"{long_floor:.2f}% (xs {long_xs:+.2f} pp) and vs E {e_up:.2f}% "
            f"(delta {long_up - e_up:+.2f} pp)."
        )

    if not live_cover_ok:
        live_reason = (
            f"NO PROMOTE stack live IR: VAL cover {100.0 * live_cover:.1f}% "
            f"< {100.0 * REL_COVER:.0f}%. Keep q20 default."
        )
    elif not ir_ok:
        live_reason = (
            f"NO PROMOTE stack live IR: VAL unlev net IR {ir:+.3f} vs q20 "
            f"{ir20:+.3f} (delta {ir_delta:+.3f} < +{STACK_IR_LIFT:.2f}). "
            "Keep q20 default. Hit-rate-only."
        )
    elif not dd_ok:
        live_reason = (
            f"NO PROMOTE stack live IR: VAL max DD {dd:+.3f} vs q20 "
            f"{dd20:+.3f} (delta {dd_delta:+.3f} < -{STACK_DD_TOL:.2f}). "
            "Keep q20 default."
        )
    else:
        live_reason = (
            f"PROMOTE stack live IR q={q:.2f} abs_q={abs_q:.2f}: VAL unlev "
            f"net IR {ir:+.3f} vs q20 {ir20:+.3f} (delta {ir_delta:+.3f}). "
            "Default CLI stays q20 until liquid."
        )
    return {
        "promote_stack_rel": promote_rel,
        "promote_stack_abs": promote_abs,
        "promote_stack_live": promote_live,
        "gated_on": "val",
        "rel_reason": rel_reason,
        "abs_reason": abs_reason,
        "live_reason": live_reason,
        "reason": f"{rel_reason} {abs_reason} {live_reason}",
        "spec": {
            "q": float(q),
            "abs_q": float(abs_q),
            "abs_tau": float(abs_tau),
            "score_col": "pred",
            "stack_long_half": True,
        },
        "chosen": rel,
        "e_sleeve": e_row,
        "train_q": float(q),
        "train_abs_q": float(abs_q),
        "train_abs_tau": float(abs_tau),
        "val_rel_hit": hit,
        "val_rel_z": z,
        "val_rel_coverage": cover,
        "val_long_up": long_up,
        "val_long_floor": long_floor,
        "val_long_excess_pp": long_xs,
        "val_e_up": e_up,
        "val_vs_e_pp": (
            float(long_up - e_up)
            if np.isfinite(long_up) and np.isfinite(e_up)
            else float("nan")
        ),
        "weaker_than_e": weaker_than_e,
        "val_ir": ir,
        "val_ir_q20": ir20,
        "val_ir_delta": ir_delta,
        "val_dd": dd,
        "val_dd_q20": dd20,
        "val_dd_delta": dd_delta,
        "rel_lift_pp": REL_DIR_LIFT_PP,
        "rel_z_floor": REL_DIR_Z,
        "cover_floor": REL_COVER,
        "abs_vs_e_pp": BOOK_UP_BASE_PP,
        "ir_lift": STACK_IR_LIFT,
        "dd_tol": STACK_DD_TOL,
        "live_book_unchanged": (not promote_live),
        "default_book_unchanged": True,
    }


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
        book_txt = format_book_aligned_block(payload)
        if book_txt:
            lines.extend(["", book_txt])
        live_txt = format_conviction_live_block(payload)
        if live_txt:
            lines.extend(["", live_txt])
        mae_txt = format_sector_mae_block(payload)
        if mae_txt:
            lines.extend(["", mae_txt])
        two_txt = format_two_stage_mae_block(payload)
        if two_txt:
            lines.extend(["", two_txt])
        sparse_txt = format_sparse_mae_block(payload)
        if sparse_txt:
            lines.extend(["", sparse_txt])
        rel_txt = format_relative_dir_block(payload)
        if rel_txt:
            lines.extend(["", rel_txt])
        stack_txt = format_rel_e_stack_block(payload)
        if stack_txt:
            lines.extend(["", stack_txt])
        short_txt = format_short_aligned_block(payload)
        if short_txt:
            lines.extend(["", short_txt])
        ej_txt = format_ej_ls_block(payload)
        if ej_txt:
            lines.extend(["", ej_txt])
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


def fit_residual_mae_maps(
    pred_r: np.ndarray,
    r_on: np.ndarray,
) -> dict[str, Any]:
    """TRAIN-only residual→overnight maps on sector-overnight skip pred*sigma."""
    a_l1, b_l1 = fit_affine_l1(pred_r, r_on)
    a_h, b_h = fit_affine_huber(pred_r, r_on)
    a_pos, b_pos, a_neg, b_neg = fit_piecewise_l1(pred_r, r_on)
    edges, values = fit_bin_constants(pred_r, r_on, n_bins=7)
    return {
        "fit_split": "train",
        "hedge": "sector_overnight",
        "maps": {
            "affine_l1": {"kind": "affine_l1", "a": float(a_l1), "b": float(b_l1)},
            "huber_affine": {"kind": "huber_affine", "a": float(a_h), "b": float(b_h)},
            "piecewise_l1": {
                "kind": "piecewise_l1",
                "a_pos": float(a_pos),
                "b_pos": float(b_pos),
                "a_neg": float(a_neg),
                "b_neg": float(b_neg),
            },
            "bin_calibrate": {
                "kind": "bin_calibrate",
                "edges": np.asarray(edges, dtype=np.float64).tolist(),
                "values": np.asarray(values, dtype=np.float64).tolist(),
            },
        },
        "note": (
            "Maps residual*sigma → raw overnight gap. Fit on sector-overnight "
            "skip preds (IDEA 3), not SPY-only residual. Next open is never a feature."
        ),
    }


def score_residual_mae_map(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    min_names: int,
) -> dict[str, float]:
    if df.empty:
        return slim_accuracy({"empty": True})
    hat = apply_calibrate_spec(
        df["pred_r"].to_numpy(dtype=np.float64),
        spec,
    )
    return slim_accuracy(_score_pred_r(df, hat, min_names))


def decide_sector_mae_promote(
    *,
    val_maps: dict[str, dict[str, Any]],
    val_residual: dict[str, Any],
    val_zero: dict[str, Any],
    val_median: dict[str, Any],
    val_current: dict[str, Any] | None,
    current_name: str,
    maps: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only. Promote a sector-overnight MAE map by a clear % MAE margin."""
    scored: list[tuple[str, dict[str, Any]]] = []
    for name in SECTOR_MAE_MAPS:
        row = dict(val_maps.get(name) or {})
        mae = _as_float(row.get("mae_pct"))
        if np.isfinite(mae):
            scored.append((name, row))
    resid = _as_float((val_residual or {}).get("mae_pct"))
    zero = _as_float((val_zero or {}).get("mae_pct"))
    median = _as_float((val_median or {}).get("mae_pct"))
    floor = min(
        [x for x in (resid, zero, median) if np.isfinite(x)],
        default=float("nan"),
    )
    best_name = ""
    best_row: dict[str, Any] = {}
    best_mae = float("nan")
    for name, row in scored:
        mae = _as_float(row.get("mae_pct"))
        if not np.isfinite(best_mae) or mae < best_mae:
            best_mae = mae
            best_name = name
            best_row = row
    margin = (
        float(floor - best_mae)
        if np.isfinite(floor) and np.isfinite(best_mae)
        else float("nan")
    )
    floors_ok = bool(np.isfinite(margin) and margin >= MAE_LIFT)
    cur_name = str(current_name or "residual_sigma")
    cur_mae = _as_float((val_current or {}).get("mae_pct"))
    if not np.isfinite(cur_mae):
        cur_mae = resid
    vs_current = (
        float(cur_mae - best_mae)
        if np.isfinite(cur_mae) and np.isfinite(best_mae)
        else float("nan")
    )
    current_is_family = cur_name in SECTOR_MAE_MAPS
    if current_is_family:
        current_ok = True
    else:
        current_ok = bool(np.isfinite(vs_current) and vs_current >= MAE_LIFT)
    same = bool(best_name and best_name == cur_name)
    promote = bool(best_name and floors_ok and current_ok and not same)
    dir_pct = _as_float(best_row.get("dir_pct"))
    resid_dir = _as_float((val_residual or {}).get("dir_pct"))
    med_dir = _as_float((val_median or {}).get("dir_pct"))
    dir_ok = bool(
        np.isfinite(dir_pct)
        and np.isfinite(resid_dir)
        and np.isfinite(med_dir)
        and dir_pct >= resid_dir + 100.0 * DIR_LIFT
        and dir_pct >= med_dir + 100.0 * DIR_LIFT
    )
    if not best_name:
        reason = "NO PROMOTE: no finite VAL MAE among sector-overnight maps."
    elif same and floors_ok:
        reason = (
            f"NO NEW MAE DEFAULT: {best_name} is already the accuracy default "
            f"(VAL MAE% {100.0 * best_mae:.4f}, margin vs floors {1e4 * margin:+.2f} bp)."
        )
    elif not floors_ok:
        reason = (
            f"NO PROMOTE: best {best_name} VAL MAE% {100.0 * best_mae:.4f} vs "
            f"residual {100.0 * resid:.4f} / zero {100.0 * zero:.4f} / "
            f"median {100.0 * median:.4f} (margin {1e4 * margin:+.2f} bp "
            f"< +{1e4 * MAE_LIFT:.1f} bp). Keep {cur_name}."
        )
    elif not current_ok:
        reason = (
            f"NO PROMOTE: {best_name} VAL MAE% {100.0 * best_mae:.4f} does not beat "
            f"current default {cur_name} {100.0 * cur_mae:.4f} by "
            f"+{1e4 * MAE_LIFT:.1f} bp (delta {1e4 * vs_current:+.2f} bp). "
            f"Keep {cur_name}."
        )
    else:
        reason = (
            f"PROMOTE sector-overnight MAE default {best_name}: VAL MAE% "
            f"{100.0 * best_mae:.4f} beats residual/zero/median by "
            f"{1e4 * margin:+.2f} bp and current {cur_name} by "
            f"{1e4 * vs_current:+.2f} bp. Live q20 book unchanged."
        )
    return {
        "promote_sector_mae": promote,
        "gated_on": "val",
        "reason": reason,
        "name": best_name if promote else cur_name,
        "best_name": best_name,
        "spec": dict((maps or {}).get(best_name) or {}) if promote else {},
        "val_best": best_row,
        "val_residual": dict(val_residual or {}),
        "val_zero": dict(val_zero or {}),
        "val_median": dict(val_median or {}),
        "val_current": dict(val_current or {}),
        "current_name": cur_name,
        "val_mae_pct": best_mae,
        "val_floor_mae_pct": floor,
        "val_margin_bp": 1e4 * margin if np.isfinite(margin) else float("nan"),
        "val_vs_current_bp": 1e4 * vs_current if np.isfinite(vs_current) else float("nan"),
        "mae_lift": MAE_LIFT,
        "dir_report_only": (not dir_ok),
        "dir_clears_gates": dir_ok,
        "hedge": "sector_overnight",
        "live_book_unchanged": True,
    }


def _price_mult_from_log(hat_r: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.asarray(hat_r, dtype=np.float64), -20.0, 20.0))


def _log_from_mult(mult: np.ndarray) -> np.ndarray:
    m = np.asarray(mult, dtype=np.float64)
    out = np.full(m.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(m) & (m > 1e-12)
    out[ok] = np.log(m[ok])
    return out


def _dow_dummies(weekdays: np.ndarray, n_days: int = 5) -> np.ndarray:
    wd = np.asarray(weekdays, dtype=np.int64)
    out = np.zeros((wd.size, int(n_days)), dtype=np.float64)
    for i, day in enumerate(wd):
        if 0 <= int(day) < int(n_days):
            out[i, int(day)] = 1.0
    return out


def fit_resid_dow_vol_ridge(
    pred_r: np.ndarray,
    r_on: np.ndarray,
    dates: np.ndarray,
    vol_level: np.ndarray | None,
    *,
    ridge: float = 10.0,
) -> dict[str, Any]:
    """TRAIN-only ridge: sector residual*sigma + causal DOW + trailing vol → r_on."""
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    if vol_level is None:
        v = np.zeros_like(p)
    else:
        v = np.asarray(vol_level, dtype=np.float64)
        if v.shape[0] != p.shape[0]:
            v = np.zeros_like(p)
    v = np.where(np.isfinite(v), v, 0.0)
    wd = weekday_of_dates(np.asarray(dates, dtype=np.int64))
    x = np.column_stack([p, v, _dow_dummies(wd)])
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(x).all(axis=1)
    n_ok = int(ok.sum())
    k = int(x.shape[1])
    if n_ok < max(12, k + 4):
        return {
            "kind": "ridge_resid_dow_vol",
            "weights": [0.0] * k,
            "bias": float(np.median(y[ok]) if n_ok else 0.0),
            "feat_mean": [0.0] * k,
            "feat_std": [1.0] * k,
            "ridge": float(ridge),
        }
    mu = x[ok].mean(axis=0)
    sd = x[ok].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x[ok] - mu) / sd
    design = np.column_stack([z, np.ones(n_ok, dtype=np.float64)])
    xtx = design.T @ design
    xtx[:k, :k] = xtx[:k, :k] + float(ridge) * np.eye(k)
    coef, *_ = np.linalg.lstsq(xtx, design.T @ y[ok], rcond=None)
    return {
        "kind": "ridge_resid_dow_vol",
        "weights": [float(v) for v in coef[:k]],
        "bias": float(coef[k]),
        "feat_mean": [float(v) for v in mu],
        "feat_std": [float(v) for v in sd],
        "ridge": float(ridge),
    }


def apply_resid_dow_vol_ridge(
    pred_r: np.ndarray,
    spec: Mapping[str, Any],
    *,
    dates: np.ndarray | None,
    vol_level: np.ndarray | None,
) -> np.ndarray:
    p = np.asarray(pred_r, dtype=np.float64)
    if vol_level is None:
        v = np.zeros_like(p)
    else:
        v = np.asarray(vol_level, dtype=np.float64)
        if v.shape[0] != p.shape[0]:
            v = np.zeros_like(p)
    v = np.where(np.isfinite(v), v, 0.0)
    if dates is None:
        wd = np.zeros(p.shape[0], dtype=np.int64)
    else:
        wd = weekday_of_dates(np.asarray(dates, dtype=np.int64))
    x = np.column_stack([p, v, _dow_dummies(wd)])
    w = np.asarray(spec.get("weights") or [], dtype=np.float64)
    mu = np.asarray(spec.get("feat_mean") or [], dtype=np.float64)
    sd = np.asarray(spec.get("feat_std") or [], dtype=np.float64)
    if w.size != x.shape[1] or mu.size != x.shape[1] or sd.size != x.shape[1]:
        return np.full_like(p, float(spec.get("bias") or 0.0))
    sd = np.where(np.abs(sd) < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    return z @ w + float(spec.get("bias") or 0.0)


def apply_two_stage_map(
    pred_r: np.ndarray,
    spec: Mapping[str, Any] | None,
    *,
    close: np.ndarray | None = None,
    dates: np.ndarray | None = None,
    vol_level: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a TRAIN-only two-stage or residual+DOW+vol MAE readout.

    Next open is never an input. ``close`` is the prior close known at t.
    """
    params = dict(spec or {})
    kind = str(params.get("kind") or params.get("name") or "").strip().lower()
    if kind in ("ridge_resid_dow_vol", "resid_dow_vol"):
        return apply_resid_dow_vol_ridge(
            pred_r, params, dates=dates, vol_level=vol_level
        )
    hat = apply_calibrate_spec(pred_r, params.get("stage1") or {})
    a2 = 1.0 if params.get("a2") is None else float(params["a2"])
    b2 = 0.0 if params.get("b2") is None else float(params["b2"])
    g = _price_mult_from_log(hat)
    space = str(params.get("stage2") or params.get("space") or "pct").lower()
    if space in ("usd", "dollar", "$"):
        if close is None:
            return _log_from_mult(a2 * g + b2)
        c = np.asarray(close, dtype=np.float64)
        implied = a2 * c * g + b2
        return _log_from_mult(implied / np.clip(c, 1e-12, None))
    return _log_from_mult(a2 * g + b2)


def fit_two_stage_mae_maps(df: pd.DataFrame) -> dict[str, Any]:
    """TRAIN-only residual→gap then gap→next-open $/%, plus residual+DOW+vol ridge.

    Stage 2 uses ``close_t`` (known) and ``next_open`` as the *label* only.
    """
    empty = {
        "fit_split": "train",
        "hedge": "sector_overnight",
        "maps": {},
        "note": (
            "Two-stage next-open MAE. Stage 1 maps residual*sigma → overnight "
            "gap. Stage 2 maps exp(gap) → next-open $ or % using close_t. "
            "ridge_resid_dow_vol is pred_r + causal DOW + trailing vol → r_on. "
            "Next open is never a feature."
        ),
    }
    if df.empty or "pred_r" not in df.columns:
        return empty
    pred_r = df["pred_r"].to_numpy(dtype=np.float64)
    r_on = df["r_on"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    nxt = df["next_open"].to_numpy(dtype=np.float64)
    stage1 = (fit_residual_mae_maps(pred_r, r_on).get("maps") or {})
    maps: dict[str, Any] = {}
    pairs = (
        ("affine_l1", "two_stage_l1_pct"),
        ("huber_affine", "two_stage_huber_pct"),
        ("piecewise_l1", "two_stage_piecewise_pct"),
    )
    y_pct = nxt / np.clip(close, 1e-12, None)
    for s1_name, out_name in pairs:
        s1 = dict(stage1.get(s1_name) or {})
        hat = apply_calibrate_spec(pred_r, s1)
        a2, b2 = fit_affine_l1(_price_mult_from_log(hat), y_pct)
        maps[out_name] = {
            "kind": out_name,
            "stage1": s1,
            "stage2": "pct",
            "a2": float(a2),
            "b2": float(b2),
        }
    s1_l1 = dict(stage1.get("affine_l1") or {})
    hat_l1 = apply_calibrate_spec(pred_r, s1_l1)
    a2u, b2u = fit_affine_l1(close * _price_mult_from_log(hat_l1), nxt)
    maps["two_stage_l1_usd"] = {
        "kind": "two_stage_l1_usd",
        "stage1": s1_l1,
        "stage2": "usd",
        "a2": float(a2u),
        "b2": float(b2u),
    }
    vol = (
        df["vol_level"].to_numpy(dtype=np.float64)
        if "vol_level" in df.columns
        else None
    )
    maps["ridge_resid_dow_vol"] = fit_resid_dow_vol_ridge(
        pred_r,
        r_on,
        df["date"].to_numpy(dtype=np.int64),
        vol,
        ridge=10.0,
    )
    empty["maps"] = maps
    return empty


def score_two_stage_mae_map(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    min_names: int,
) -> dict[str, float]:
    if df.empty:
        return slim_accuracy({"empty": True})
    vol = (
        df["vol_level"].to_numpy(dtype=np.float64)
        if "vol_level" in df.columns
        else None
    )
    hat = apply_two_stage_map(
        df["pred_r"].to_numpy(dtype=np.float64),
        spec,
        close=df["close"].to_numpy(dtype=np.float64),
        dates=df["date"].to_numpy(dtype=np.int64),
        vol_level=vol,
    )
    return slim_accuracy(_score_pred_r(df, hat, min_names))


def decide_two_stage_mae_promote(
    *,
    val_maps: dict[str, dict[str, Any]],
    val_residual: dict[str, Any],
    val_zero: dict[str, Any],
    val_median: dict[str, Any],
    val_current: dict[str, Any] | None,
    current_name: str,
    maps: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only. Promote a two-stage / ridge MAE map vs floors and current default."""
    scored: list[tuple[str, dict[str, Any]]] = []
    for name in TWO_STAGE_MAE_MAPS:
        row = dict(val_maps.get(name) or {})
        mae = _as_float(row.get("mae_pct"))
        if np.isfinite(mae):
            scored.append((name, row))
    resid = _as_float((val_residual or {}).get("mae_pct"))
    zero = _as_float((val_zero or {}).get("mae_pct"))
    median = _as_float((val_median or {}).get("mae_pct"))
    floor = min(
        [x for x in (resid, zero, median) if np.isfinite(x)],
        default=float("nan"),
    )
    best_name = ""
    best_row: dict[str, Any] = {}
    best_mae = float("nan")
    for name, row in scored:
        mae = _as_float(row.get("mae_pct"))
        if not np.isfinite(best_mae) or mae < best_mae:
            best_mae = mae
            best_name = name
            best_row = row
    margin = (
        float(floor - best_mae)
        if np.isfinite(floor) and np.isfinite(best_mae)
        else float("nan")
    )
    floors_ok = bool(np.isfinite(margin) and margin >= MAE_LIFT)
    cur_name = str(current_name or "residual_sigma")
    cur_mae = _as_float((val_current or {}).get("mae_pct"))
    if not np.isfinite(cur_mae):
        cur_mae = resid
    vs_current = (
        float(cur_mae - best_mae)
        if np.isfinite(cur_mae) and np.isfinite(best_mae)
        else float("nan")
    )
    current_ok = bool(np.isfinite(vs_current) and vs_current >= -1e-15)
    same = bool(best_name and best_name == cur_name)
    promote = bool(best_name and floors_ok and current_ok and not same)
    dir_pct = _as_float(best_row.get("dir_pct"))
    resid_dir = _as_float((val_residual or {}).get("dir_pct"))
    med_dir = _as_float((val_median or {}).get("dir_pct"))
    dir_ok = bool(
        np.isfinite(dir_pct)
        and np.isfinite(resid_dir)
        and np.isfinite(med_dir)
        and dir_pct >= resid_dir + 100.0 * DIR_LIFT
        and dir_pct >= med_dir + 100.0 * DIR_LIFT
    )
    if not best_name:
        reason = "NO PROMOTE: no finite VAL MAE among two-stage next-open maps."
    elif same and floors_ok:
        reason = (
            f"NO NEW MAE DEFAULT: {best_name} is already the accuracy default "
            f"(VAL MAE% {100.0 * best_mae:.4f}, margin vs floors {1e4 * margin:+.2f} bp)."
        )
    elif not floors_ok:
        reason = (
            f"NO PROMOTE: best {best_name} VAL MAE% {100.0 * best_mae:.4f} vs "
            f"residual {100.0 * resid:.4f} / zero {100.0 * zero:.4f} / "
            f"median {100.0 * median:.4f} (margin {1e4 * margin:+.2f} bp "
            f"< +{1e4 * MAE_LIFT:.1f} bp). Keep {cur_name}."
        )
    elif not current_ok:
        reason = (
            f"NO PROMOTE: {best_name} VAL MAE% {100.0 * best_mae:.4f} is worse "
            f"than current default {cur_name} {100.0 * cur_mae:.4f} "
            f"(delta {1e4 * vs_current:+.2f} bp). Keep {cur_name}."
        )
    else:
        reason = (
            f"PROMOTE two-stage next-open MAE default {best_name}: VAL MAE% "
            f"{100.0 * best_mae:.4f} beats residual/zero/median by "
            f"{1e4 * margin:+.2f} bp and is not worse than {cur_name} "
            f"({1e4 * vs_current:+.2f} bp). Live q20 book unchanged."
        )
    return {
        "promote_two_stage_mae": promote,
        "gated_on": "val",
        "reason": reason,
        "name": best_name if promote else cur_name,
        "best_name": best_name,
        "spec": dict((maps or {}).get(best_name) or {}) if promote else {},
        "val_best": best_row,
        "val_residual": dict(val_residual or {}),
        "val_zero": dict(val_zero or {}),
        "val_median": dict(val_median or {}),
        "val_current": dict(val_current or {}),
        "current_name": cur_name,
        "val_mae_pct": best_mae,
        "val_floor_mae_pct": floor,
        "val_margin_bp": 1e4 * margin if np.isfinite(margin) else float("nan"),
        "val_vs_current_bp": 1e4 * vs_current if np.isfinite(vs_current) else float("nan"),
        "mae_lift": MAE_LIFT,
        "dir_report_only": (not dir_ok),
        "dir_clears_gates": dir_ok,
        "hedge": "sector_overnight",
        "live_book_unchanged": True,
    }


def apply_sparse_mae(
    pred_r: np.ndarray,
    a: float,
    b: float,
    tau: float,
) -> np.ndarray:
    """``a * pred_r + b`` when ``|pred_r| >= τ``, else 0 (zero-move)."""
    p = np.asarray(pred_r, dtype=np.float64)
    hat = float(a) * p + float(b)
    small = np.isfinite(p) & (np.abs(p) < float(tau))
    hat[small] = 0.0
    hat[~np.isfinite(p)] = np.nan
    return hat


def fit_sparse_mae_maps(
    df: pd.DataFrame,
    *,
    min_names: int,
) -> dict[str, Any]:
    """TRAIN-only affine_l1/huber + |pred*sigma| τ. Small gaps predict 0."""
    empty = {
        "fit_split": "train",
        "hedge": "sector_overnight",
        "maps": {},
        "grid": [],
        "note": (
            "Fit residual*sigma → overnight gap on TRAIN (affine_l1 / huber). "
            "Choose |pred| τ on TRAIN % MAE. Below τ predict 0 (zero-move). "
            "Next open is never a feature."
        ),
    }
    if df.empty or "pred_r" not in df.columns:
        return empty
    pred_r = df["pred_r"].to_numpy(dtype=np.float64)
    r_on = df["r_on"].to_numpy(dtype=np.float64)
    a_l1, b_l1 = fit_affine_l1(pred_r, r_on)
    a_h, b_h = fit_affine_huber(pred_r, r_on)
    abs_ok = pred_r[np.isfinite(pred_r)]
    families = (
        ("sparse_l1", float(a_l1), float(b_l1)),
        ("sparse_huber", float(a_h), float(b_h)),
    )
    grid: list[dict[str, Any]] = []
    maps: dict[str, Any] = {}
    for name, a, b in families:
        best_mae = float("inf")
        best: dict[str, Any] | None = None
        for q in SPARSE_ABS_QS:
            if abs_ok.size == 0 or float(q) <= 0.0:
                tau = 0.0
            else:
                tau = float(np.quantile(np.abs(abs_ok), float(q)))
            keep = np.isfinite(pred_r) & (np.abs(pred_r) >= tau)
            cover = float(keep.mean()) if pred_r.size else 0.0
            if float(q) > 0.0 and cover < SPARSE_COVER:
                continue
            hat = apply_sparse_mae(pred_r, a, b, tau)
            scored = slim_accuracy(_score_pred_r(df, hat, min_names))
            row = {
                "name": name,
                "abs_q": float(q),
                "tau": tau,
                "cover": cover,
                "a": a,
                "b": b,
                **scored,
            }
            grid.append(row)
            mae = _as_float(scored.get("mae_pct"))
            if np.isfinite(mae) and mae < best_mae:
                best_mae = mae
                best = {
                    "kind": name,
                    "a": a,
                    "b": b,
                    "tau": tau,
                    "abs_q": float(q),
                    "cover": cover,
                    "train_mae_pct": mae,
                }
        maps[name] = best or {
            "kind": name,
            "a": a,
            "b": b,
            "tau": 0.0,
            "abs_q": 0.0,
            "cover": 1.0,
            "train_mae_pct": float("nan"),
        }
    empty["maps"] = maps
    empty["grid"] = grid
    return empty


def score_sparse_mae_map(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    min_names: int,
) -> dict[str, float]:
    if df.empty:
        return slim_accuracy({"empty": True})
    hat = apply_sparse_mae(
        df["pred_r"].to_numpy(dtype=np.float64),
        float(spec.get("a") or 0.0),
        float(spec.get("b") or 0.0),
        float(spec.get("tau") or 0.0),
    )
    out = slim_accuracy(_score_pred_r(df, hat, min_names))
    keep = np.isfinite(df["pred_r"].to_numpy(dtype=np.float64)) & (
        np.abs(df["pred_r"].to_numpy(dtype=np.float64)) >= float(spec.get("tau") or 0.0)
    )
    out["cover"] = float(keep.mean()) if len(df) else float("nan")
    out["tau"] = float(spec.get("tau") or 0.0)
    out["abs_q"] = float(spec.get("abs_q") or 0.0)
    return out


def decide_sparse_mae_promote(
    *,
    val_maps: dict[str, dict[str, Any]],
    val_residual: dict[str, Any],
    val_zero: dict[str, Any],
    val_median: dict[str, Any],
    val_current: dict[str, Any] | None,
    current_name: str,
    maps: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only. Promote sparse MAE vs floors and current default."""
    scored: list[tuple[str, dict[str, Any]]] = []
    for name in SPARSE_MAE_MAPS:
        row = dict(val_maps.get(name) or {})
        mae = _as_float(row.get("mae_pct"))
        if np.isfinite(mae):
            scored.append((name, row))
    resid = _as_float((val_residual or {}).get("mae_pct"))
    zero = _as_float((val_zero or {}).get("mae_pct"))
    median = _as_float((val_median or {}).get("mae_pct"))
    floor = min(
        [x for x in (resid, zero, median) if np.isfinite(x)],
        default=float("nan"),
    )
    best_name = ""
    best_row: dict[str, Any] = {}
    best_mae = float("nan")
    for name, row in scored:
        mae = _as_float(row.get("mae_pct"))
        if not np.isfinite(best_mae) or mae < best_mae:
            best_mae = mae
            best_name = name
            best_row = row
    margin = (
        float(floor - best_mae)
        if np.isfinite(floor) and np.isfinite(best_mae)
        else float("nan")
    )
    floors_ok = bool(np.isfinite(margin) and margin >= MAE_LIFT)
    cur_name = str(current_name or "residual_sigma")
    cur_mae = _as_float((val_current or {}).get("mae_pct"))
    if not np.isfinite(cur_mae):
        cur_mae = resid
    vs_current = (
        float(cur_mae - best_mae)
        if np.isfinite(cur_mae) and np.isfinite(best_mae)
        else float("nan")
    )
    current_ok = bool(np.isfinite(vs_current) and vs_current >= -1e-15)
    same = bool(best_name and best_name == cur_name)
    promote = bool(best_name and floors_ok and current_ok and not same)
    dir_pct = _as_float(best_row.get("dir_pct"))
    resid_dir = _as_float((val_residual or {}).get("dir_pct"))
    med_dir = _as_float((val_median or {}).get("dir_pct"))
    dir_ok = bool(
        np.isfinite(dir_pct)
        and np.isfinite(resid_dir)
        and np.isfinite(med_dir)
        and dir_pct >= resid_dir + 100.0 * DIR_LIFT
        and dir_pct >= med_dir + 100.0 * DIR_LIFT
    )
    spec = dict((maps or {}).get(best_name) or {}) if promote else {}
    tau = _as_float((spec or best_row).get("tau"))
    if not best_name:
        reason = "NO PROMOTE: no finite VAL MAE among sparse MAE maps."
    elif same and floors_ok:
        reason = (
            f"NO NEW MAE DEFAULT: {best_name} is already the accuracy default "
            f"(VAL MAE% {100.0 * best_mae:.4f}, margin vs floors {1e4 * margin:+.2f} bp)."
        )
    elif not floors_ok:
        reason = (
            f"NO PROMOTE: best {best_name} VAL MAE% {100.0 * best_mae:.4f} vs "
            f"residual {100.0 * resid:.4f} / zero {100.0 * zero:.4f} / "
            f"median {100.0 * median:.4f} (margin {1e4 * margin:+.2f} bp "
            f"< +{1e4 * MAE_LIFT:.1f} bp). Keep {cur_name}."
        )
    elif not current_ok:
        reason = (
            f"NO PROMOTE: {best_name} VAL MAE% {100.0 * best_mae:.4f} is worse "
            f"than current default {cur_name} {100.0 * cur_mae:.4f} "
            f"(delta {1e4 * vs_current:+.2f} bp). Keep {cur_name}."
        )
    else:
        reason = (
            f"PROMOTE sparse MAE default {best_name} τ={tau:.6f}: VAL MAE% "
            f"{100.0 * best_mae:.4f} beats residual/zero/median by "
            f"{1e4 * margin:+.2f} bp and is not worse than {cur_name} "
            f"({1e4 * vs_current:+.2f} bp). Live q20 book unchanged."
        )
    return {
        "promote_sparse_mae": promote,
        "gated_on": "val",
        "reason": reason,
        "name": best_name if promote else cur_name,
        "best_name": best_name,
        "spec": spec,
        "val_best": best_row,
        "val_residual": dict(val_residual or {}),
        "val_zero": dict(val_zero or {}),
        "val_median": dict(val_median or {}),
        "val_current": dict(val_current or {}),
        "current_name": cur_name,
        "val_mae_pct": best_mae,
        "val_floor_mae_pct": floor,
        "val_margin_bp": 1e4 * margin if np.isfinite(margin) else float("nan"),
        "val_vs_current_bp": 1e4 * vs_current if np.isfinite(vs_current) else float("nan"),
        "tau": tau,
        "mae_lift": MAE_LIFT,
        "dir_report_only": (not dir_ok),
        "dir_clears_gates": dir_ok,
        "hedge": "sector_overnight",
        "live_book_unchanged": True,
    }


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
    close: np.ndarray | None = None,
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
    if kind in SPARSE_MAE_MAPS or kind in ("sparse_mae",):
        return apply_sparse_mae(
            p,
            1.0 if params.get("a") is None else float(params["a"]),
            0.0 if params.get("b") is None else float(params["b"]),
            float(params.get("tau") or 0.0),
        )
    if kind in TWO_STAGE_MAE_MAPS or kind in (
        "two_stage_pct",
        "two_stage_usd",
        "resid_dow_vol",
    ):
        return apply_two_stage_map(
            p, params, close=close, dates=dates, vol_level=vol_level
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


def _fmt_book_sleeve(row: Mapping[str, Any] | None) -> str:
    r = dict(row or {})
    return (
        f"up {_as_float(r.get('up_pct')):.2f}%  "
        f"floor {_as_float(r.get('uncond_up_pct')):.2f}%  "
        f"xs {_as_float(r.get('excess_pp')):+.2f}pp  "
        f"cover {100.0 * _as_float(r.get('coverage')):.1f}%  "
        f"n={int(_as_float(r.get('n'), 0.0))}  "
        f"dates={int(_as_float(r.get('n_dates'), 0.0))}  "
        f"MAE ${_as_float(r.get('mae_usd')):.4f} / "
        f"{100.0 * _as_float(r.get('mae_pct')):.4f}%  "
        f"(full ${_as_float(r.get('full_mae_usd')):.4f} / "
        f"{100.0 * _as_float(r.get('full_mae_pct')):.4f}%)  "
        f"paperIR {_as_float(r.get('paper_ir')):+.3f}"
    )


def format_book_aligned_block(payload: dict[str, Any]) -> str:
    """VAL-gated overnight-up of TRAIN-chosen top-q residual names."""
    promo = payload.get("book_aligned_promotion") or {}
    fit = payload.get("book_aligned_fit") or {}
    cmp = payload.get("book_aligned_compare") or {}
    if not promo and not fit:
        return ""
    yes = bool(promo.get("promote_book_aligned"))
    chosen = fit.get("chosen") or {}
    tr = (cmp.get("train") or {}).get("chosen") or chosen
    va = (cmp.get("val") or {}).get("chosen") or {}
    te = (cmp.get("test") or {}).get("chosen") or {}
    va20 = (cmp.get("val") or {}).get("top20") or {}
    te20 = (cmp.get("test") or {}).get("top20") or {}
    q = _as_float(chosen.get("q"), default=0.80)
    abs_q = _as_float(chosen.get("abs_q"), default=0.0)
    abs_tau = _as_float(chosen.get("abs_tau"), default=0.0)
    grid = (cmp.get("val") or {}).get("grid") or []
    grid_lines = []
    for row in grid:
        grid_lines.append(
            f"    q={_as_float(row.get('q')):.2f} abs_q={_as_float(row.get('abs_q')):.2f}  "
            f"{_fmt_book_sleeve(row)}"
        )
    if not grid_lines:
        grid_lines = ["    (empty)"]
    return "\n".join(
        [
            f"PROMOTE BOOK-ALIGNED? {'YES' if yes else 'NO'}",
            "  Primary object = overnight up-rate of within-date top-q residual "
            "skip names (sector-overnight skip pred) vs unconditional overnight-up. "
            "Pooled TS direction stays report-only. TEST is report-only. "
            "MAE on the sleeve vs full tape is secondary. Live book CLI unchanged.",
            f"  TRAIN pick q={q:.2f} (top {100.0 * (1.0 - q):.0f}%)  "
            f"abs_q={abs_q:.2f}  |pred|>={abs_tau:.5f}  "
            f"fit_split={fit.get('fit_split')!r}",
            f"  TRAIN chosen  {_fmt_book_sleeve(tr)}",
            f"  VAL   chosen  {_fmt_book_sleeve(va)}",
            f"  VAL   top-20% {_fmt_book_sleeve(va20)}",
            f"  VAL   vs floor {_as_float(promo.get('val_excess_pp')):+.2f}pp  "
            f"(need ≥+{BOOK_UP_FLOOR_PP:.1f}pp)  vs top-20% "
            f"{_as_float(promo.get('val_vs_top20_pp')):+.2f}pp  "
            f"(need ≥+{BOOK_UP_BASE_PP:.1f}pp)",
            f"  TEST  chosen  {_fmt_book_sleeve(te)}  (report-only)",
            f"  TEST  top-20% {_fmt_book_sleeve(te20)}  (report-only)",
            f"  {promo.get('reason') or 'no decision'}",
            "  VAL grid (report-only; |pred| τ from TRAIN):",
            *grid_lines,
        ]
    )


def _fmt_short_sleeve(row: Mapping[str, Any] | None) -> str:
    r = dict(row or {})
    return (
        f"down {_as_float(r.get('down_pct')):.2f}%  "
        f"floor {_as_float(r.get('uncond_down_pct')):.2f}%  "
        f"xs {_as_float(r.get('excess_pp')):+.2f}pp  "
        f"cover {100.0 * _as_float(r.get('coverage')):.1f}%  "
        f"n={int(_as_float(r.get('n'), 0.0))}  "
        f"dates={int(_as_float(r.get('n_dates'), 0.0))}  "
        f"MAE ${_as_float(r.get('mae_usd')):.4f} / "
        f"{100.0 * _as_float(r.get('mae_pct')):.4f}%  "
        f"paperIR {_as_float(r.get('paper_ir')):+.3f}"
    )


def format_short_aligned_block(payload: dict[str, Any]) -> str:
    """VAL-gated overnight-down of TRAIN-chosen bottom-q residual names."""
    promo = payload.get("short_aligned_promotion") or {}
    fit = payload.get("short_aligned_fit") or {}
    cmp = payload.get("short_aligned_compare") or {}
    live = payload.get("short_aligned_live") or {}
    if not promo and not fit:
        return ""
    yes = bool(promo.get("promote_short_aligned"))
    yes_live = bool(promo.get("promote_short_live"))
    chosen = fit.get("chosen") or {}
    tr = (cmp.get("train") or {}).get("chosen") or chosen
    va = (cmp.get("val") or {}).get("chosen") or {}
    te = (cmp.get("test") or {}).get("chosen") or {}
    va20 = (cmp.get("val") or {}).get("bot20") or {}
    te20 = (cmp.get("test") or {}).get("bot20") or {}
    q = _as_float(chosen.get("q"), default=0.20)
    abs_q = _as_float(chosen.get("abs_q"), default=0.0)
    abs_tau = _as_float(chosen.get("abs_tau"), default=0.0)
    grid = (cmp.get("val") or {}).get("grid") or []
    grid_lines = []
    for row in grid:
        grid_lines.append(
            f"    q={_as_float(row.get('q')):.2f} abs_q={_as_float(row.get('abs_q')):.2f}  "
            f"{_fmt_short_sleeve(row)}"
        )
    if not grid_lines:
        grid_lines = ["    (empty)"]
    va_live = live.get("val") or {}
    te_live = live.get("test") or {}
    return "\n".join(
        [
            f"PROMOTE SHORT-ALIGNED? {'YES' if yes else 'NO'}  "
            f"PROMOTE SHORT LIVE LS? {'YES' if yes_live else 'NO'}",
            "  Primary object = overnight down-rate of within-date bottom-q "
            "residual skip names vs unconditional overnight-down. Optional TRAIN "
            "|pred| floor. TEST report-only. live_locate LS (long q20, short "
            "TRAIN sleeve) vs long-only q20 is optional. Live q20 unchanged "
            "on hit-rate-only.",
            f"  TRAIN pick q={q:.2f} (bottom {100.0 * q:.0f}%)  "
            f"abs_q={abs_q:.2f}  |pred|>={abs_tau:.5f}  "
            f"fit_split={fit.get('fit_split')!r}",
            f"  TRAIN chosen  {_fmt_short_sleeve(tr)}",
            f"  VAL   chosen  {_fmt_short_sleeve(va)}",
            f"  VAL   bot-20% {_fmt_short_sleeve(va20)}",
            f"  VAL   vs floor {_as_float(promo.get('val_excess_pp')):+.2f}pp  "
            f"(need ≥+{BOOK_UP_FLOOR_PP:.1f}pp)  vs bottom-20% "
            f"{_as_float(promo.get('val_vs_bot20_pp')):+.2f}pp  "
            f"(need ≥+{BOOK_UP_BASE_PP:.1f}pp)",
            f"  TEST  chosen  {_fmt_short_sleeve(te)}  (report-only)",
            f"  TEST  bot-20% {_fmt_short_sleeve(te20)}  (report-only)",
            _fmt_stack_live("VAL q20", va_live.get("q20")),
            _fmt_stack_live("VAL LS", va_live.get("ls")),
            _fmt_stack_live("TEST q20", te_live.get("q20")),
            _fmt_stack_live("TEST LS", te_live.get("ls")),
            f"  {promo.get('hit_reason') or promo.get('reason') or 'no hit decision'}",
            f"  {promo.get('live_reason') or 'no live decision'}",
            "  VAL grid (report-only; |pred| τ from TRAIN):",
            *grid_lines,
        ]
    )


def format_ej_ls_block(payload: dict[str, Any]) -> str:
    promo = payload.get("ej_ls_promotion") or {}
    cmp = payload.get("ej_ls_compare") or {}
    if not promo and not cmp:
        return ""
    yes = bool(promo.get("promote_ej_ls"))
    va = cmp.get("val") or {}
    te = cmp.get("test") or {}
    return "\n".join(
        [
            f"PROMOTE E+J LIVE LS? {'YES' if yes else 'NO'}",
            "  Symmetric LS: long IDEA E TRAIN top-q ∩ |pred|, short IDEA J "
            "TRAIN bottom-q ∩ |pred|. Paper is zero-cost. live_locate is the "
            "gate vs long-only q20 (IR ≥ q20+0.05, DD not worse by >0.05, "
            "cover ≥ 5%). TEST report-only. Live q20 unchanged unless the "
            "IR/DD gate clears. E+J hit-rate promotes stay hit-rate-only "
            "if live does not clear.",
            f"  TRAIN E q={_as_float(promo.get('e_q')):.2f} abs_q="
            f"{_as_float(promo.get('e_abs_q')):.2f}  "
            f"J q={_as_float(promo.get('j_q')):.2f} abs_q="
            f"{_as_float(promo.get('j_abs_q')):.2f}  gated_on={promo.get('gated_on')!r}",
            "  VAL (gate):",
            _fmt_stack_live("q20", va.get("q20")),
            _fmt_stack_live("paper 0c", va.get("paper")),
            _fmt_stack_live("live_locate", va.get("live")),
            f"  VAL IR delta {_as_float(promo.get('val_ir_delta')):+.3f}  "
            f"(need ≥+{EJ_IR_LIFT:.2f})  DD delta "
            f"{_as_float(promo.get('val_dd_delta')):+.3f}  "
            f"(need ≥-{EJ_DD_TOL:.2f})  cover "
            f"{100.0 * _as_float(promo.get('val_coverage')):.1f}%",
            "  TEST (report-only):",
            _fmt_stack_live("q20", te.get("q20")),
            _fmt_stack_live("paper 0c", te.get("paper")),
            _fmt_stack_live("live_locate", te.get("live")),
            f"  {promo.get('reason') or 'no E+J live decision'}",
        ]
    )


def format_conviction_live_block(payload: dict[str, Any]) -> str:
    """IDEA F live-cost IR gate; implemented in forecast.shorting."""
    if not payload.get("conviction_live_promotion") and not payload.get("conviction_live"):
        return ""
    from forecast.shorting import _conviction_live_block

    return _conviction_live_block(payload)


def _fmt_mae_row(label: str, row: Mapping[str, Any] | None) -> str:
    r = dict(row or {})
    return (
        f"  {label:<16}  "
        f"MAE% {100.0 * _as_float(r.get('mae_pct')):.4f}  "
        f"MAE$ {_as_float(r.get('mae_usd')):.4f}  "
        f"dir {_as_float(r.get('dir_pct')):.2f}%  "
        f"xs {_as_float(r.get('excess_pp')):+.2f}pp"
    )


def format_sector_mae_block(payload: dict[str, Any]) -> str:
    promo = payload.get("sector_mae_promotion") or {}
    cmp = payload.get("sector_mae_compare") or {}
    fit = payload.get("sector_mae_fit") or {}
    if not promo and not cmp:
        return ""
    yes = bool(promo.get("promote_sector_mae"))
    val_maps = cmp.get("val") or {}
    test_maps = cmp.get("test") or {}
    lines = [
        f"PROMOTE SECTOR-MAE? {'YES' if yes else 'NO'}",
        "  TRAIN residual→overnight maps on sector-overnight skip pred*sigma "
        "(affine_l1 / piecewise_l1 / huber_affine / bin_calibrate). "
        "VAL % MAE must beat residual×σ AND zero-move AND train-median by "
        f"≥{1e4 * MAE_LIFT:.1f} bp, and must not lose to the current MAE default. "
        "Dir excess is report-only unless it also clears dir gates. "
        "Live q20 book unchanged.",
        f"  hedge={fit.get('hedge') or promo.get('hedge')!r}  "
        f"fit_split={fit.get('fit_split')!r}  "
        f"current_default={promo.get('current_name')!r}",
        "  VAL (gate):",
        _fmt_mae_row("residual×σ", cmp.get("val_residual") or {}),
        _fmt_mae_row("zero-move", cmp.get("val_zero") or {}),
        _fmt_mae_row("train-median", cmp.get("val_median") or {}),
    ]
    for name in SECTOR_MAE_MAPS:
        mark = " *" if name == promo.get("best_name") else ""
        lines.append(_fmt_mae_row(name + mark, val_maps.get(name) or {}))
    if cmp.get("val_current") and str(promo.get("current_name") or "") not in SECTOR_MAE_MAPS:
        lines.append(
            _fmt_mae_row(
                f"current {promo.get('current_name')}",
                cmp.get("val_current") or {},
            )
        )
    lines.extend(
        [
            f"  VAL best={promo.get('best_name')!r}  "
            f"margin vs floors {_as_float(promo.get('val_margin_bp')):+.2f} bp  "
            f"vs current {_as_float(promo.get('val_vs_current_bp')):+.2f} bp  "
            f"(need ≥+{1e4 * MAE_LIFT:.1f} bp)  "
            f"dir_clears_gates={bool(promo.get('dir_clears_gates'))}",
            "  TEST (report-only):",
            _fmt_mae_row("residual×σ", cmp.get("test_residual") or {}),
        ]
    )
    for name in SECTOR_MAE_MAPS:
        lines.append(_fmt_mae_row(name, test_maps.get(name) or {}))
    lines.append(f"  {promo.get('reason') or 'no decision'}")
    return "\n".join(lines)


def format_two_stage_mae_block(payload: dict[str, Any]) -> str:
    promo = payload.get("two_stage_mae_promotion") or {}
    cmp = payload.get("two_stage_mae_compare") or {}
    fit = payload.get("two_stage_mae_fit") or {}
    if not promo and not cmp:
        return ""
    yes = bool(promo.get("promote_two_stage_mae"))
    val_maps = cmp.get("val") or {}
    test_maps = cmp.get("test") or {}
    lines = [
        f"PROMOTE TWO-STAGE MAE? {'YES' if yes else 'NO'}",
        "  TRAIN two-stage next-open MAE: residual*sigma → overnight gap "
        "(affine_l1 / huber / piecewise), then gap → next-open $/% using "
        "close_t. Also residual + causal DOW + trailing vol ridge. "
        f"VAL % MAE must beat residual×σ AND zero-move AND train-median by "
        f"≥{1e4 * MAE_LIFT:.1f} bp, and must not be worse than the current "
        "MAE default. Dir report-only unless it also clears dir gates. "
        "Live q20 book unchanged. Next open is never a feature.",
        f"  hedge={fit.get('hedge') or promo.get('hedge')!r}  "
        f"fit_split={fit.get('fit_split')!r}  "
        f"current_default={promo.get('current_name')!r}",
        "  VAL (gate):",
        _fmt_mae_row("residual×σ", cmp.get("val_residual") or {}),
        _fmt_mae_row("zero-move", cmp.get("val_zero") or {}),
        _fmt_mae_row("train-median", cmp.get("val_median") or {}),
    ]
    for name in TWO_STAGE_MAE_MAPS:
        mark = " *" if name == promo.get("best_name") else ""
        lines.append(_fmt_mae_row(name + mark, val_maps.get(name) or {}))
    if cmp.get("val_current") and str(promo.get("current_name") or "") not in TWO_STAGE_MAE_MAPS:
        lines.append(
            _fmt_mae_row(
                f"current {promo.get('current_name')}",
                cmp.get("val_current") or {},
            )
        )
    lines.extend(
        [
            f"  VAL best={promo.get('best_name')!r}  "
            f"margin vs floors {_as_float(promo.get('val_margin_bp')):+.2f} bp  "
            f"vs current {_as_float(promo.get('val_vs_current_bp')):+.2f} bp  "
            f"(floors need ≥+{1e4 * MAE_LIFT:.1f} bp; current must not be worse)  "
            f"dir_clears_gates={bool(promo.get('dir_clears_gates'))}",
            "  TEST (report-only):",
            _fmt_mae_row("residual×σ", cmp.get("test_residual") or {}),
        ]
    )
    for name in TWO_STAGE_MAE_MAPS:
        lines.append(_fmt_mae_row(name, test_maps.get(name) or {}))
    lines.append(f"  {promo.get('reason') or 'no decision'}")
    return "\n".join(lines)


def format_sparse_mae_block(payload: dict[str, Any]) -> str:
    promo = payload.get("sparse_mae_promotion") or {}
    cmp = payload.get("sparse_mae_compare") or {}
    fit = payload.get("sparse_mae_fit") or {}
    if not promo and not cmp:
        return ""
    yes = bool(promo.get("promote_sparse_mae"))
    val_maps = cmp.get("val") or {}
    test_maps = cmp.get("test") or {}
    lines = [
        f"PROMOTE SPARSE MAE? {'YES' if yes else 'NO'}",
        "  TRAIN affine_l1 / huber on residual*sigma → overnight gap. "
        "TRAIN |pred| τ: below τ predict 0 (zero-move), else use the map. "
        f"VAL % MAE must beat residual×σ AND zero-move AND train-median by "
        f"≥{1e4 * MAE_LIFT:.1f} bp, and must not be worse than the current "
        "MAE default. Dir report-only unless it also clears dir gates. "
        "Live q20 book unchanged. Next open is never a feature.",
        f"  hedge={fit.get('hedge') or promo.get('hedge')!r}  "
        f"fit_split={fit.get('fit_split')!r}  "
        f"current_default={promo.get('current_name')!r}",
        "  VAL (gate):",
        _fmt_mae_row("residual×σ", cmp.get("val_residual") or {}),
        _fmt_mae_row("zero-move", cmp.get("val_zero") or {}),
        _fmt_mae_row("train-median", cmp.get("val_median") or {}),
    ]
    for name in SPARSE_MAE_MAPS:
        mark = " *" if name == promo.get("best_name") else ""
        row = dict(val_maps.get(name) or {})
        spec = ((fit.get("maps") or {}).get(name) or {})
        lines.append(
            _fmt_mae_row(name + mark, row)
            + f"  τ={_as_float(spec.get('tau') if spec.get('tau') is not None else row.get('tau')):.6f}"
            f"  abs_q={_as_float(spec.get('abs_q') if spec.get('abs_q') is not None else row.get('abs_q')):.2f}"
            f"  cover {100.0 * _as_float(row.get('cover') if row.get('cover') is not None else spec.get('cover')):.1f}%"
        )
    if cmp.get("val_current") and str(promo.get("current_name") or "") not in SPARSE_MAE_MAPS:
        lines.append(
            _fmt_mae_row(
                f"current {promo.get('current_name')}",
                cmp.get("val_current") or {},
            )
        )
    lines.extend(
        [
            f"  VAL best={promo.get('best_name')!r}  "
            f"margin vs floors {_as_float(promo.get('val_margin_bp')):+.2f} bp  "
            f"vs current {_as_float(promo.get('val_vs_current_bp')):+.2f} bp  "
            f"(floors need ≥+{1e4 * MAE_LIFT:.1f} bp; current must not be worse)  "
            f"dir_clears_gates={bool(promo.get('dir_clears_gates'))}",
            "  TEST (report-only):",
            _fmt_mae_row("residual×σ", cmp.get("test_residual") or {}),
        ]
    )
    for name in SPARSE_MAE_MAPS:
        lines.append(_fmt_mae_row(name, test_maps.get(name) or {}))
    lines.append(f"  {promo.get('reason') or 'no decision'}")
    return "\n".join(lines)


def _fmt_rel_row(label: str, row: Mapping[str, Any] | None) -> str:
    r = dict(row or {})
    return (
        f"  {label:<16}  "
        f"rel {_as_float(r.get('rel_hit_pct')):.2f}%  "
        f"xs {_as_float(r.get('rel_excess_pp')):+.2f}pp vs 50%  "
        f"z={_as_float(r.get('rel_z')):.2f}  "
        f"cover {100.0 * _as_float(r.get('rel_coverage')):.1f}%  "
        f"n={int(_as_float(r.get('rel_n'), 0.0))}  |  "
        f"long-half up {_as_float(r.get('long_up_pct')):.2f}%  "
        f"xs {_as_float(r.get('long_excess_pp')):+.2f}pp vs "
        f"{_as_float(r.get('long_uncond_up_pct')):.2f}%  "
        f"n={int(_as_float(r.get('long_n'), 0.0))}"
    )


def format_relative_dir_block(payload: dict[str, Any]) -> str:
    promo = payload.get("relative_dir_promotion") or {}
    fit = payload.get("relative_dir_fit") or {}
    cmp = payload.get("relative_dir_compare") or {}
    if not promo and not fit:
        return ""
    yes_rel = bool(promo.get("promote_relative_dir"))
    yes_long = bool(promo.get("promote_long_half"))
    chosen = fit.get("chosen") or {}
    va_ch = (cmp.get("val") or {}).get("chosen") or {}
    va_full = (cmp.get("val") or {}).get("full") or {}
    va20 = (cmp.get("val") or {}).get("top20") or {}
    te_ch = (cmp.get("test") or {}).get("chosen") or {}
    te_full = (cmp.get("test") or {}).get("full") or {}
    te20 = (cmp.get("test") or {}).get("top20") or {}
    return "\n".join(
        [
            f"PROMOTE RELATIVE-DIR? {'YES' if yes_rel else 'NO'}  "
            f"PROMOTE LONG-HALF UP? {'YES' if yes_long else 'NO'}",
            "  Within-date sign(pred − CS median) vs sign(r_on − CS median) "
            "on sector-overnight skip pred. Hit vs 50%. Long-half = pred > CS "
            "median, absolute overnight-up vs uncond floor and vs top-20%. "
            "Optional TRAIN |pred−median| floor. TEST report-only. "
            "Live q20 unchanged.",
            f"  TRAIN pick abs_q={_as_float(chosen.get('abs_q')):.2f}  "
            f"|pred−med|>={_as_float(chosen.get('abs_tau')):.5f}  "
            f"fit_split={fit.get('fit_split')!r}",
            "  VAL (gate):",
            _fmt_rel_row("full", va_full),
            _fmt_rel_row("TRAIN sleeve", va_ch),
            f"  VAL top-20% up {_as_float(va20.get('up_pct')):.2f}%  "
            f"xs {_as_float(va20.get('excess_pp')):+.2f}pp",
            f"  VAL rel vs 50% {_as_float(promo.get('val_rel_hit')):.2f}%  "
            f"z={_as_float(promo.get('val_rel_z')):.2f}  "
            f"(need ≥{50.0 + REL_DIR_LIFT_PP:.1f}% and z≥{REL_DIR_Z:.1f})  "
            f"long-half vs floor {_as_float(promo.get('val_long_excess_pp')):+.2f}pp  "
            f"vs top-20% {_as_float(promo.get('val_vs_top20_pp')):+.2f}pp",
            "  TEST (report-only):",
            _fmt_rel_row("full", te_full),
            _fmt_rel_row("TRAIN sleeve", te_ch),
            f"  TEST top-20% up {_as_float(te20.get('up_pct')):.2f}%  "
            f"xs {_as_float(te20.get('excess_pp')):+.2f}pp",
            f"  {promo.get('rel_reason') or 'no relative decision'}",
            f"  {promo.get('long_reason') or 'no long-half decision'}",
        ]
    )


def _fmt_stack_live(label: str, row: Mapping[str, Any] | None) -> str:
    r = dict(row or {})
    return (
        f"  {label:<16}  "
        f"IR {_as_float(r.get('unlevered_net_ir')):+.3f}  "
        f"maxDD {_as_float(r.get('unlevered_max_dd')):+.3f}  "
        f"to {_as_float(r.get('mean_turnover')):.3f}  "
        f"cost {_as_float(r.get('mean_cost_unlev_bp')):.1f}bp  "
        f"cover {100.0 * _as_float(r.get('coverage')):.1f}%  "
        f"n={int(_as_float(r.get('n'), 0.0))}"
    )


def format_rel_e_stack_block(payload: dict[str, Any]) -> str:
    promo = payload.get("rel_e_stack_promotion") or {}
    fit = payload.get("rel_e_stack_fit") or {}
    cmp = payload.get("rel_e_stack_compare") or {}
    live = payload.get("rel_e_stack_live") or {}
    if not promo and not fit:
        return ""
    yes_rel = bool(promo.get("promote_stack_rel"))
    yes_abs = bool(promo.get("promote_stack_abs"))
    yes_live = bool(promo.get("promote_stack_live"))
    chosen = fit.get("chosen") or {}
    va = (cmp.get("val") or {}).get("chosen") or {}
    va_e = (cmp.get("val") or {}).get("e_sleeve") or {}
    va_h = (cmp.get("val") or {}).get("h_chosen") or {}
    te = (cmp.get("test") or {}).get("chosen") or {}
    te_e = (cmp.get("test") or {}).get("e_sleeve") or {}
    va_live = (live.get("val") or {})
    te_live = (live.get("test") or {})
    return "\n".join(
        [
            f"PROMOTE STACK RELATIVE-DIR? {'YES' if yes_rel else 'NO'}  "
            f"PROMOTE STACK ABSOLUTE-UP? {'YES' if yes_abs else 'NO'}  "
            f"PROMOTE STACK LIVE IR? {'YES' if yes_live else 'NO'}",
            "  H∩E stack: pred > CS median and TRAIN top-q / |pred| floor "
            "on sector-overnight skip pred. Relative hit vs 50% (cover ≥ 5%). "
            "Absolute overnight-up vs E's sleeve (+0.2pp) and the up-floor. "
            "Live IR vs q20 (F gate +0.05). TEST report-only. "
            "Live q20 unchanged unless the IR gate clears.",
            f"  TRAIN pick q={_as_float(chosen.get('q')):.2f}  "
            f"abs_q={_as_float(chosen.get('abs_q')):.2f}  "
            f"|pred|>={_as_float(chosen.get('abs_tau')):.5f}  "
            f"fit_split={fit.get('fit_split')!r}",
            "  VAL (gate):",
            _fmt_rel_row("H∩E stack", va),
            f"  VAL E sleeve up {_as_float(va_e.get('up_pct')):.2f}%  "
            f"xs {_as_float(va_e.get('excess_pp')):+.2f}pp  "
            f"cover {100.0 * _as_float(va_e.get('coverage')):.1f}%",
            f"  VAL H sleeve rel {_as_float(va_h.get('rel_hit_pct')):.2f}%  "
            f"z={_as_float(va_h.get('rel_z')):.2f}",
            f"  VAL rel vs 50% {_as_float(promo.get('val_rel_hit')):.2f}%  "
            f"z={_as_float(promo.get('val_rel_z')):.2f}  "
            f"(need ≥{50.0 + REL_DIR_LIFT_PP:.1f}% and z≥{REL_DIR_Z:.1f})  "
            f"abs vs E {_as_float(promo.get('val_vs_e_pp')):+.2f}pp  "
            f"(need ≥+{BOOK_UP_BASE_PP:.1f}pp)",
            _fmt_stack_live("q20", va_live.get("q20")),
            _fmt_stack_live("H∩E live", va_live.get("stack")),
            _fmt_stack_live("E live", va_live.get("e")),
            "  TEST (report-only):",
            _fmt_rel_row("H∩E stack", te),
            f"  TEST E sleeve up {_as_float(te_e.get('up_pct')):.2f}%  "
            f"xs {_as_float(te_e.get('excess_pp')):+.2f}pp",
            _fmt_stack_live("q20", te_live.get("q20")),
            _fmt_stack_live("H∩E live", te_live.get("stack")),
            f"  {promo.get('rel_reason') or 'no stack relative decision'}",
            f"  {promo.get('abs_reason') or 'no stack absolute decision'}",
            f"  {promo.get('live_reason') or 'no stack live decision'}",
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
    book_aligned_fit = fit_book_aligned_on_train(tr, min_names=min_names)
    short_aligned_fit = fit_short_aligned_on_train(tr, min_names=min_names)
    relative_dir_fit = fit_relative_dir_on_train(tr, min_names=min_names)
    rel_e_stack_fit = fit_rel_e_stack_on_train(
        tr, min_names=min_names, e_chosen=book_aligned_fit.get("chosen") or {}
    )
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
    by_ablate = {str(r.get("name")): r for r in rows}
    sector_mae_fit = fit_residual_mae_maps(train_pred_r, train_r)
    sector_mae_val = {
        n: dict((by_ablate.get(n) or {}).get("val") or {}) for n in SECTOR_MAE_MAPS
    }
    sector_mae_test = {
        n: dict((by_ablate.get(n) or {}).get("test") or {}) for n in SECTOR_MAE_MAPS
    }
    sector_mae_promotion = decide_sector_mae_promote(
        val_maps=sector_mae_val,
        val_residual=dict((by_ablate.get("residual_sigma") or {}).get("val") or {}),
        val_zero=dict((by_ablate.get("zero_move") or {}).get("val") or {}),
        val_median=dict((by_ablate.get("train_median_gap") or {}).get("val") or {}),
        val_current=dict((by_ablate.get(default_name) or {}).get("val") or {}),
        current_name=default_name,
        maps=sector_mae_fit.get("maps") or {},
    )
    if sector_mae_promotion.get("promote_sector_mae"):
        winner = str(sector_mae_promotion.get("best_name") or default_name)
        promotion["price"] = winner
        promotion["accuracy_default"] = winner
        default_name = winner
        default_pr_test = pred_r_by_name.get(winner, {}).get("test")
        if default_pr_test is None:
            default_pr_test = te["pred_r"].to_numpy(dtype=np.float64)

    two_stage_mae_fit = fit_two_stage_mae_maps(tr)
    two_stage_maps = two_stage_mae_fit.get("maps") or {}
    two_stage_mae_val = {
        n: score_two_stage_mae_map(va, spec, min_names=min_names)
        for n, spec in two_stage_maps.items()
    }
    two_stage_mae_test = {
        n: score_two_stage_mae_map(te, spec, min_names=min_names)
        for n, spec in two_stage_maps.items()
    }
    two_stage_mae_promotion = decide_two_stage_mae_promote(
        val_maps=two_stage_mae_val,
        val_residual=dict((by_ablate.get("residual_sigma") or {}).get("val") or {}),
        val_zero=dict((by_ablate.get("zero_move") or {}).get("val") or {}),
        val_median=dict((by_ablate.get("train_median_gap") or {}).get("val") or {}),
        val_current=dict((by_ablate.get(default_name) or {}).get("val") or {}),
        current_name=default_name,
        maps=two_stage_maps,
    )
    if two_stage_mae_promotion.get("promote_two_stage_mae"):
        winner = str(two_stage_mae_promotion.get("best_name") or default_name)
        promotion["price"] = winner
        promotion["accuracy_default"] = winner
        default_name = winner
        default_pr_test = apply_two_stage_map(
            te["pred_r"].to_numpy(dtype=np.float64),
            two_stage_maps.get(winner) or {},
            close=te["close"].to_numpy(dtype=np.float64),
            dates=te["date"].to_numpy(dtype=np.int64),
            vol_level=(
                te["vol_level"].to_numpy(dtype=np.float64)
                if "vol_level" in te.columns
                else None
            ),
        )
        pred_r_by_name[winner] = {
            "train": apply_two_stage_map(
                train_pred_r,
                two_stage_maps.get(winner) or {},
                close=tr["close"].to_numpy(dtype=np.float64),
                dates=tr["date"].to_numpy(dtype=np.int64),
                vol_level=(
                    tr["vol_level"].to_numpy(dtype=np.float64)
                    if "vol_level" in tr.columns
                    else None
                ),
            ),
            "val": apply_two_stage_map(
                va["pred_r"].to_numpy(dtype=np.float64),
                two_stage_maps.get(winner) or {},
                close=va["close"].to_numpy(dtype=np.float64),
                dates=va["date"].to_numpy(dtype=np.int64),
                vol_level=(
                    va["vol_level"].to_numpy(dtype=np.float64)
                    if "vol_level" in va.columns
                    else None
                ),
            ),
            "test": default_pr_test,
        }

    sparse_mae_fit = fit_sparse_mae_maps(tr, min_names=min_names)
    sparse_maps = sparse_mae_fit.get("maps") or {}
    sparse_mae_val = {
        n: score_sparse_mae_map(va, spec, min_names=min_names)
        for n, spec in sparse_maps.items()
    }
    sparse_mae_test = {
        n: score_sparse_mae_map(te, spec, min_names=min_names)
        for n, spec in sparse_maps.items()
    }
    sparse_mae_promotion = decide_sparse_mae_promote(
        val_maps=sparse_mae_val,
        val_residual=dict((by_ablate.get("residual_sigma") or {}).get("val") or {}),
        val_zero=dict((by_ablate.get("zero_move") or {}).get("val") or {}),
        val_median=dict((by_ablate.get("train_median_gap") or {}).get("val") or {}),
        val_current=dict((by_ablate.get(default_name) or {}).get("val") or {}),
        current_name=default_name,
        maps=sparse_maps,
    )
    if sparse_mae_promotion.get("promote_sparse_mae"):
        winner = str(sparse_mae_promotion.get("best_name") or default_name)
        promotion["price"] = winner
        promotion["accuracy_default"] = winner
        default_name = winner
        default_pr_test = apply_sparse_mae(
            te["pred_r"].to_numpy(dtype=np.float64),
            float((sparse_maps.get(winner) or {}).get("a") or 0.0),
            float((sparse_maps.get(winner) or {}).get("b") or 0.0),
            float((sparse_maps.get(winner) or {}).get("tau") or 0.0),
        )
        pred_r_by_name[winner] = {
            "train": apply_sparse_mae(
                train_pred_r,
                float((sparse_maps.get(winner) or {}).get("a") or 0.0),
                float((sparse_maps.get(winner) or {}).get("b") or 0.0),
                float((sparse_maps.get(winner) or {}).get("tau") or 0.0),
            ),
            "val": apply_sparse_mae(
                va["pred_r"].to_numpy(dtype=np.float64),
                float((sparse_maps.get(winner) or {}).get("a") or 0.0),
                float((sparse_maps.get(winner) or {}).get("b") or 0.0),
                float((sparse_maps.get(winner) or {}).get("tau") or 0.0),
            ),
            "test": default_pr_test,
        }

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
            "sector_mae": {
                "hedge": "sector_overnight",
                "fit_split": "train",
                "maps": list(SECTOR_MAE_MAPS),
            },
            "two_stage_mae": {
                "hedge": "sector_overnight",
                "fit_split": "train",
                "maps": list(TWO_STAGE_MAE_MAPS),
            },
            "sparse_mae": {
                "hedge": "sector_overnight",
                "fit_split": "train",
                "maps": list(SPARSE_MAE_MAPS),
            },
            "book_aligned": {
                "q": float((book_aligned_fit.get("chosen") or {}).get("q") or 0.80),
                "abs_tau": float(
                    (book_aligned_fit.get("chosen") or {}).get("abs_tau") or 0.0
                ),
                "abs_q": float(
                    (book_aligned_fit.get("chosen") or {}).get("abs_q") or 0.0
                ),
                "fit_split": "train",
            },
            "relative_dir": {
                "abs_tau": float(
                    (relative_dir_fit.get("chosen") or {}).get("abs_tau") or 0.0
                ),
                "abs_q": float(
                    (relative_dir_fit.get("chosen") or {}).get("abs_q") or 0.0
                ),
                "score_col": "pred",
                "fit_split": "train",
            },
            "short_aligned": {
                "q": float((short_aligned_fit.get("chosen") or {}).get("q") or 0.20),
                "abs_tau": float(
                    (short_aligned_fit.get("chosen") or {}).get("abs_tau") or 0.0
                ),
                "abs_q": float(
                    (short_aligned_fit.get("chosen") or {}).get("abs_q") or 0.0
                ),
                "fit_split": "train",
            },
            "rel_e_stack": {
                "q": float((rel_e_stack_fit.get("chosen") or {}).get("q") or 0.80),
                "abs_tau": float(
                    (rel_e_stack_fit.get("chosen") or {}).get("abs_tau") or 0.0
                ),
                "abs_q": float(
                    (rel_e_stack_fit.get("chosen") or {}).get("abs_q") or 0.0
                ),
                "score_col": "pred",
                "fit_split": "train",
            },
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
    if default_name in two_stage_maps:
        default_params = dict(two_stage_maps[default_name])
    if default_name in sparse_maps:
        default_params = dict(sparse_maps[default_name])
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
        **{
            n: apply_two_stage_map(
                train_pred_r,
                spec,
                close=tr["close"].to_numpy(dtype=np.float64),
                dates=tr["date"].to_numpy(dtype=np.int64),
                vol_level=(
                    tr["vol_level"].to_numpy(dtype=np.float64)
                    if "vol_level" in tr.columns
                    else None
                ),
            )
            for n, spec in two_stage_maps.items()
        },
        **{
            n: apply_sparse_mae(
                train_pred_r,
                float(spec.get("a") or 0.0),
                float(spec.get("b") or 0.0),
                float(spec.get("tau") or 0.0),
            )
            for n, spec in sparse_maps.items()
        },
    }.get(default_name, train_pred_r)
    payload["confidence"] = _confidence_block(
        te, default_pr_test, train_abs=train_default, min_names=min_names
    )
    payload["year_slices"] = _year_direction(te)
    payload["year_slices_readout"] = _year_direction(apply_readout(te, default_pr_test))
    chosen_ba = book_aligned_fit.get("chosen") or {}
    q_hat = float(chosen_ba.get("q") or 0.80)
    abs_tau_hat = float(chosen_ba.get("abs_tau") or 0.0)
    train_pred_cs = tr["pred"].to_numpy(dtype=np.float64)
    val_chosen_ba = score_book_aligned_sleeve(
        va, q=q_hat, abs_tau=abs_tau_hat, min_names=min_names
    )
    test_chosen_ba = score_book_aligned_sleeve(
        te, q=q_hat, abs_tau=abs_tau_hat, min_names=min_names
    )
    val_top20_ba = score_book_aligned_sleeve(
        va, q=0.80, abs_tau=0.0, min_names=min_names
    )
    test_top20_ba = score_book_aligned_sleeve(
        te, q=0.80, abs_tau=0.0, min_names=min_names
    )
    book_aligned_promotion = decide_book_aligned_promote(
        val_chosen=val_chosen_ba,
        val_top20=val_top20_ba,
        chosen=chosen_ba,
    )
    payload["book_aligned_fit"] = book_aligned_fit
    payload["book_aligned_compare"] = {
        "train": {
            "chosen": book_aligned_fit.get("chosen"),
            "top20": book_aligned_fit.get("baseline"),
            "grid": book_aligned_fit.get("rows"),
        },
        "val": {
            "chosen": val_chosen_ba,
            "top20": val_top20_ba,
            "grid": book_aligned_grid(
                va, train_pred=train_pred_cs, min_names=min_names
            ),
        },
        "test": {
            "chosen": test_chosen_ba,
            "top20": test_top20_ba,
            "grid": book_aligned_grid(
                te, train_pred=train_pred_cs, min_names=min_names
            ),
        },
        "note": (
            "Primary object = overnight up-rate of within-date top-q residual "
            "skip names vs unconditional overnight-up. Pooled TS dir is "
            "report-only. TEST is report-only. Sleeve MAE vs full tape is secondary."
        ),
    }
    payload["book_aligned_promotion"] = book_aligned_promotion
    chosen_rel = relative_dir_fit.get("chosen") or {}
    rtau = float(chosen_rel.get("abs_tau") or 0.0)
    val_chosen_rel = score_relative_direction(
        va, abs_tau=rtau, score_col="pred", min_names=min_names
    )
    test_chosen_rel = score_relative_direction(
        te, abs_tau=rtau, score_col="pred", min_names=min_names
    )
    val_full_rel = score_relative_direction(
        va, abs_tau=0.0, score_col="pred", min_names=min_names
    )
    test_full_rel = score_relative_direction(
        te, abs_tau=0.0, score_col="pred", min_names=min_names
    )
    relative_dir_promotion = decide_relative_dir_promote(
        val_chosen=val_chosen_rel,
        val_full=val_full_rel,
        val_top20=val_top20_ba,
        chosen=chosen_rel,
    )
    payload["relative_dir_fit"] = relative_dir_fit
    payload["relative_dir_compare"] = {
        "train": {
            "chosen": chosen_rel,
            "full": relative_dir_fit.get("full"),
            "top20": book_aligned_fit.get("baseline"),
        },
        "val": {
            "chosen": val_chosen_rel,
            "full": val_full_rel,
            "top20": val_top20_ba,
        },
        "test": {
            "chosen": test_chosen_rel,
            "full": test_full_rel,
            "top20": test_top20_ba,
        },
        "note": (
            "Within-date sign(pred − CS median) vs sign(r_on − CS median) "
            "on sector-overnight skip pred. Hit vs 50%. Long-half = pred > "
            "CS median, absolute overnight-up vs uncond floor and vs top-20%. "
            "Optional TRAIN |pred−median| floor. TEST is report-only. "
            "Live q20 book unchanged. Pooled TS dir stays report-only."
        ),
    }
    payload["relative_dir_promotion"] = relative_dir_promotion
    payload["sector_mae_fit"] = sector_mae_fit
    payload["sector_mae_compare"] = {
        "hedge": "sector_overnight",
        "val": sector_mae_val,
        "test": sector_mae_test,
        "val_residual": dict((by_ablate.get("residual_sigma") or {}).get("val") or {}),
        "val_zero": dict((by_ablate.get("zero_move") or {}).get("val") or {}),
        "val_median": dict((by_ablate.get("train_median_gap") or {}).get("val") or {}),
        "val_current": dict((by_ablate.get(str(promotion.get("accuracy_default") or default_name)) or {}).get("val") or {}),
        "test_residual": dict((by_ablate.get("residual_sigma") or {}).get("test") or {}),
        "test_zero": dict((by_ablate.get("zero_move") or {}).get("test") or {}),
        "test_median": dict((by_ablate.get("train_median_gap") or {}).get("test") or {}),
        "current_name": str(promotion.get("accuracy_default") or default_name),
        "note": (
            "VAL % MAE vs residual×σ / zero-move / train-median. "
            "Promote new MAE default only with a clear margin. "
            "Dir excess is report-only unless it also clears dir gates. "
            "Live q20 book unchanged."
        ),
    }
    payload["sector_mae_promotion"] = sector_mae_promotion
    payload["two_stage_mae_fit"] = two_stage_mae_fit
    payload["two_stage_mae_compare"] = {
        "hedge": "sector_overnight",
        "val": two_stage_mae_val,
        "test": two_stage_mae_test,
        "val_residual": dict((by_ablate.get("residual_sigma") or {}).get("val") or {}),
        "val_zero": dict((by_ablate.get("zero_move") or {}).get("val") or {}),
        "val_median": dict((by_ablate.get("train_median_gap") or {}).get("val") or {}),
        "val_current": dict((by_ablate.get(str(two_stage_mae_promotion.get("current_name") or default_name)) or {}).get("val") or {}),
        "test_residual": dict((by_ablate.get("residual_sigma") or {}).get("test") or {}),
        "test_zero": dict((by_ablate.get("zero_move") or {}).get("test") or {}),
        "test_median": dict((by_ablate.get("train_median_gap") or {}).get("test") or {}),
        "current_name": str(two_stage_mae_promotion.get("current_name") or default_name),
        "note": (
            "Two-stage next-open MAE (residual→gap→price) and residual+DOW+vol "
            "ridge. VAL % MAE vs residual×σ / zero-move / train-median / current "
            "default. Dir report-only unless it clears dir gates. "
            "Live q20 book unchanged. Next open is never a feature."
        ),
    }
    payload["two_stage_mae_promotion"] = two_stage_mae_promotion
    payload["sparse_mae_fit"] = sparse_mae_fit
    payload["sparse_mae_compare"] = {
        "hedge": "sector_overnight",
        "val": sparse_mae_val,
        "test": sparse_mae_test,
        "val_residual": dict((by_ablate.get("residual_sigma") or {}).get("val") or {}),
        "val_zero": dict((by_ablate.get("zero_move") or {}).get("val") or {}),
        "val_median": dict((by_ablate.get("train_median_gap") or {}).get("val") or {}),
        "val_current": dict((by_ablate.get(str(sparse_mae_promotion.get("current_name") or default_name)) or {}).get("val") or {}),
        "test_residual": dict((by_ablate.get("residual_sigma") or {}).get("test") or {}),
        "test_zero": dict((by_ablate.get("zero_move") or {}).get("test") or {}),
        "test_median": dict((by_ablate.get("train_median_gap") or {}).get("test") or {}),
        "current_name": str(sparse_mae_promotion.get("current_name") or default_name),
        "note": (
            "Sparse MAE: TRAIN affine_l1/huber, TRAIN |pred| τ, else zero-move. "
            "VAL % MAE vs residual×σ / zero-move / train-median / current default. "
            "Dir report-only unless it clears dir gates. Live q20 unchanged."
        ),
    }
    payload["sparse_mae_promotion"] = sparse_mae_promotion
    from forecast.shorting import (
        compare_conviction_live,
        decide_conviction_live_promote,
        score_conviction_live_book,
        score_short_aligned_live_book,
        score_ej_ls_book,
        PAPER_ZERO_BUNDLE,
        LIVE_LOCATE_BUNDLE,
    )

    conviction_live = compare_conviction_live(
        {"train": tr, "val": va, "test": te},
        chosen=chosen_ba,
        min_names=min_names,
        vol_target=0.15,
    )
    conviction_live_promotion = decide_conviction_live_promote(
        val_q20=(conviction_live.get("val") or {}).get("q20") or {},
        val_chosen=(conviction_live.get("val") or {}).get("chosen") or {},
        chosen=chosen_ba,
    )
    payload["conviction_live"] = conviction_live
    payload["conviction_live_promotion"] = conviction_live_promotion
    chosen_stack = rel_e_stack_fit.get("chosen") or {}
    sq = float(chosen_stack.get("q") or 0.80)
    stau = float(chosen_stack.get("abs_tau") or 0.0)
    e_q = float((book_aligned_fit.get("chosen") or {}).get("q") or 0.80)
    e_tau = float((book_aligned_fit.get("chosen") or {}).get("abs_tau") or 0.0)
    val_stack = score_rel_e_stack(va, q=sq, abs_tau=stau, min_names=min_names)
    test_stack = score_rel_e_stack(te, q=sq, abs_tau=stau, min_names=min_names)
    val_e_on_stack = score_rel_e_stack(va, q=e_q, abs_tau=e_tau, min_names=min_names)
    test_e_on_stack = score_rel_e_stack(te, q=e_q, abs_tau=e_tau, min_names=min_names)

    rel_e_stack_live: dict[str, Any] = {}
    for split_name, split_df in (("train", tr), ("val", va), ("test", te)):
        q20_row = (conviction_live.get(split_name) or {}).get("q20") or {}
        e_live = (conviction_live.get(split_name) or {}).get("chosen") or {}
        if not q20_row:
            q20_row = score_conviction_live_book(
                split_df,
                min_names=min_names,
                vol_target=0.15,
                name="q20_equal",
            )
        rel_e_stack_live[split_name] = {
            "q20": q20_row,
            "stack": score_conviction_live_book(
                split_df,
                min_names=min_names,
                vol_target=0.15,
                conviction_q=sq,
                conf_abs=stau,
                stack_long_half=True,
                name=f"rel_e_q{sq:.2f}_abs{float(chosen_stack.get('abs_q') or 0.0):.2f}",
            ),
            "e": e_live,
        }
    rel_e_stack_promotion = decide_rel_e_stack_promote(
        val_chosen=val_stack,
        val_e=val_chosen_ba,
        val_live_stack=(rel_e_stack_live.get("val") or {}).get("stack") or {},
        val_live_q20=(rel_e_stack_live.get("val") or {}).get("q20") or {},
        chosen=chosen_stack,
    )
    payload["rel_e_stack_fit"] = rel_e_stack_fit
    payload["rel_e_stack_compare"] = {
        "train": {
            "chosen": chosen_stack,
            "e_ref": rel_e_stack_fit.get("e_ref"),
            "e_sleeve": book_aligned_fit.get("chosen"),
        },
        "val": {
            "chosen": val_stack,
            "e_ref": val_e_on_stack,
            "e_sleeve": val_chosen_ba,
            "h_chosen": val_chosen_rel,
        },
        "test": {
            "chosen": test_stack,
            "e_ref": test_e_on_stack,
            "e_sleeve": test_chosen_ba,
            "h_chosen": test_chosen_rel,
        },
        "note": (
            "H∩E stack: pred > CS median and TRAIN top-q / |pred| floor. "
            "Relative hit vs 50%. Absolute overnight-up vs E's sleeve. "
            "Live IR vs q20 (F gate). TEST report-only. "
            "Live q20 unchanged unless the IR gate clears."
        ),
    }
    payload["rel_e_stack_live"] = rel_e_stack_live
    payload["rel_e_stack_promotion"] = rel_e_stack_promotion
    chosen_sh = short_aligned_fit.get("chosen") or {}
    sh_q = float(chosen_sh.get("q") or 0.20)
    sh_tau = float(chosen_sh.get("abs_tau") or 0.0)
    val_chosen_sh = score_short_aligned_sleeve(
        va, q=sh_q, abs_tau=sh_tau, min_names=min_names
    )
    test_chosen_sh = score_short_aligned_sleeve(
        te, q=sh_q, abs_tau=sh_tau, min_names=min_names
    )
    val_bot20_sh = score_short_aligned_sleeve(
        va, q=0.20, abs_tau=0.0, min_names=min_names
    )
    test_bot20_sh = score_short_aligned_sleeve(
        te, q=0.20, abs_tau=0.0, min_names=min_names
    )
    short_aligned_live: dict[str, Any] = {}
    for split_name, split_df in (("train", tr), ("val", va), ("test", te)):
        q20_row = (conviction_live.get(split_name) or {}).get("q20") or {}
        if not q20_row:
            q20_row = score_conviction_live_book(
                split_df,
                min_names=min_names,
                vol_target=0.15,
                name="q20_equal",
            )
        short_aligned_live[split_name] = {
            "q20": q20_row,
            "ls": score_short_aligned_live_book(
                split_df,
                min_names=min_names,
                vol_target=0.15,
                short_q=sh_q,
                conf_abs=sh_tau,
                name=f"short_ls_q{sh_q:.2f}_abs{float(chosen_sh.get('abs_q') or 0.0):.2f}",
            ),
        }
    short_aligned_promotion = decide_short_aligned_promote(
        val_chosen=val_chosen_sh,
        val_bot20=val_bot20_sh,
        chosen=chosen_sh,
        val_live_ls=(short_aligned_live.get("val") or {}).get("ls") or {},
        val_live_q20=(short_aligned_live.get("val") or {}).get("q20") or {},
    )
    payload["short_aligned_fit"] = short_aligned_fit
    payload["short_aligned_compare"] = {
        "train": {
            "chosen": chosen_sh,
            "bot20": short_aligned_fit.get("baseline"),
            "grid": short_aligned_fit.get("rows"),
        },
        "val": {
            "chosen": val_chosen_sh,
            "bot20": val_bot20_sh,
            "grid": short_aligned_grid(
                va, train_pred=train_pred_cs, min_names=min_names
            ),
        },
        "test": {
            "chosen": test_chosen_sh,
            "bot20": test_bot20_sh,
            "grid": short_aligned_grid(
                te, train_pred=train_pred_cs, min_names=min_names
            ),
        },
        "note": (
            "Primary object = overnight down-rate of within-date bottom-q "
            "residual skip names vs unconditional overnight-down. "
            "Pooled TS dir is report-only. TEST is report-only. "
            "Live q20 unchanged on hit-rate-only."
        ),
    }
    payload["short_aligned_live"] = short_aligned_live
    payload["short_aligned_promotion"] = short_aligned_promotion
    chosen_e = book_aligned_fit.get("chosen") or chosen_ba
    chosen_j = short_aligned_fit.get("chosen") or chosen_sh
    ej_ls_compare: dict[str, Any] = {}
    for split_name, split_df in (("train", tr), ("val", va), ("test", te)):
        q20_row = (conviction_live.get(split_name) or {}).get("q20") or {}
        if not q20_row:
            q20_row = score_conviction_live_book(
                split_df,
                min_names=min_names,
                vol_target=0.15,
                name="q20_equal",
            )
        ej_ls_compare[split_name] = {
            "q20": q20_row,
            "paper": score_ej_ls_book(
                split_df,
                min_names=min_names,
                vol_target=0.15,
                e_chosen=chosen_e,
                j_chosen=chosen_j,
                bundle=PAPER_ZERO_BUNDLE,
                name="ej_ls_paper",
            ),
            "live": score_ej_ls_book(
                split_df,
                min_names=min_names,
                vol_target=0.15,
                e_chosen=chosen_e,
                j_chosen=chosen_j,
                bundle=LIVE_LOCATE_BUNDLE,
                name="ej_ls_live_locate",
            ),
        }
    ej_ls_promotion = decide_ej_ls_promote(
        val_live=(ej_ls_compare.get("val") or {}).get("live") or {},
        val_q20=(ej_ls_compare.get("val") or {}).get("q20") or {},
        val_paper=(ej_ls_compare.get("val") or {}).get("paper") or {},
        e_chosen=chosen_e,
        j_chosen=chosen_j,
    )
    payload["ej_ls_compare"] = {
        **ej_ls_compare,
        "e_chosen": {
            "q": float(chosen_e.get("q") or 0.80),
            "abs_q": float(chosen_e.get("abs_q") or 0.0),
            "abs_tau": float(chosen_e.get("abs_tau") or 0.0),
        },
        "j_chosen": {
            "q": float(chosen_j.get("q") or 0.20),
            "abs_q": float(chosen_j.get("abs_q") or 0.0),
            "abs_tau": float(chosen_j.get("abs_tau") or 0.0),
        },
        "note": (
            "Symmetric long-E / short-J LS. Paper is zero-cost. "
            "live_locate vs long-only q20 is the VAL gate. TEST report-only. "
            "Live q20 unchanged unless the IR/DD gate clears."
        ),
    }
    payload["ej_ls_promotion"] = ej_ls_promotion
    if log_fn:
        log_fn(
            f"accuracy default={default_name!r}  "
            f"promote_dir={promotion.get('direction')!r}  "
            f"promote_mae={promotion.get('price')!r}  "
            f"promote_book_aligned="
            f"{bool(book_aligned_promotion.get('promote_book_aligned'))}  "
            f"promote_conviction_live="
            f"{bool(conviction_live_promotion.get('promote_conviction_live'))}  "
            f"promote_sector_mae="
            f"{bool(sector_mae_promotion.get('promote_sector_mae'))}  "
            f"promote_relative_dir="
            f"{bool(relative_dir_promotion.get('promote_relative_dir'))}  "
            f"promote_long_half="
            f"{bool(relative_dir_promotion.get('promote_long_half'))}  "
            f"promote_stack_rel="
            f"{bool(rel_e_stack_promotion.get('promote_stack_rel'))}  "
            f"promote_stack_abs="
            f"{bool(rel_e_stack_promotion.get('promote_stack_abs'))}  "
            f"promote_stack_live="
            f"{bool(rel_e_stack_promotion.get('promote_stack_live'))}  "
            f"promote_short_aligned="
            f"{bool(short_aligned_promotion.get('promote_short_aligned'))}  "
            f"promote_short_live="
            f"{bool(short_aligned_promotion.get('promote_short_live'))}  "
            f"promote_ej_ls="
            f"{bool(ej_ls_promotion.get('promote_ej_ls'))}  "
            f"promote_two_stage_mae="
            f"{bool(two_stage_mae_promotion.get('promote_two_stage_mae'))}  "
            f"promote_sparse_mae="
            f"{bool(sparse_mae_promotion.get('promote_sparse_mae'))}"
        )
    return payload
