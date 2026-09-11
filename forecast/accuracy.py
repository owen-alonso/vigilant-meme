"""Locked-window TRUE/FALSE overnight accuracy (not live P&L).

The promoted overnight skip predicts a *residual* in vol units. This module
converts that to an implied overnight log-return ``pred * sigma`` (same as
``generate.py``) and scores:

- direction vs realized ``r_on = log(open_{t+1}) - log(close_t)``
- implied next-open vs actual next open

Default recipe is the PR #5 overnight skip (rank-target ridge, ``no_long_ts``).
PR #7 levers stay off unless a checkpoint documents them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from forecast.alphavantage import symbol_parquet_path
from forecast.config import DataConfig, interval_data_kwargs
from forecast.data import (
    SymbolArrays,
    _date_keys,
    assemble_panel,
    build_datasets,
    load_bars,
    symbol_from_path,
)
from forecast.overnight import OVERNIGHT_FORMULA, formula_log_line
from forecast.ridge import cs_stats, feature_mask, fit_ridge_xy, labelled_rows as ridge_labelled_rows

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
) -> DataConfig:
    """Locked overnight skip DataConfig (same cuts/features as ``cs_overnight``)."""
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
        sector_residual=True,
        equities_only=not synthetic,
        train_from="" if synthetic else "1999-01-01",
        label_return=label_return,
        fill_minutes=0,
    )


def two_sided_normal_p(z: float) -> float:
    """Two-sided p-value from a standard-normal z (erfc)."""
    if not np.isfinite(z):
        return float("nan")
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


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
) -> pd.DataFrame:
    """One labelled last-bar row per name/date with pred, y, r_on, prices."""
    mean = np.asarray(feature_mean, dtype=np.float64)
    std = np.asarray(feature_std, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    rows: list[dict[str, Any]] = []
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
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "pred",
                "y",
                "scale",
                "close",
                "next_open",
                "r_on",
                "pred_r",
                "implied_open",
                "implied_open_given_hedge",
            ]
        )
    df = pd.DataFrame(rows)
    df["pred_r"] = df["pred"] * df["scale"]
    df["implied_open"] = df["close"] * np.exp(df["pred_r"])
    # resid = y * scale = r_on - beta * r_hedge. Adding the realized hedge
    # term is *not* a live next-open forecast (uses next hedge open).
    hedge = df["r_on"] - df["y"] * df["scale"]
    df["implied_open_given_hedge"] = df["close"] * np.exp(df["pred_r"] + hedge)
    return df


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
        f"HEADLINE direction accuracy = {hit:.1f}%  (vs 50% chance)",
        f"  {better}  z={z:.2f}  p={p:.3g}  t={overall.get('t_vs_half')}",
        f"  this is pooled name-date TIME-SERIES direction of implied overnight "
        f"move (pred*sigma) vs realized r_on = log(open_{{t+1}})-log(close_t).",
        f"  residual-vs-residual pooled dir={resid.get('hit_rate_pct'):.1f}%  "
        f"CS demeaned residual sign={float(cs_sign.get('cs_sign_hit_pct') or float('nan')):.1f}% "
        f"(that last one is the cross-sectional object).",
        f"  long-only (pred>0): {long_only.get('hit_rate_pct'):.1f}%  "
        f"(unconditional overnight up-rate {direction.get('realized_overnight_up_pct'):.1f}%; "
        f"model predicts up {direction.get('pred_positive_pct'):.1f}% of the time)  "
        f"|pred|>=median: {conv.get('hit_rate_pct'):.1f}%",
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
    return "\n".join(lines)
