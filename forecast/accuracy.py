"""Locked-window TRUE/FALSE overnight accuracy (not live P&L).

The promoted overnight skip predicts a *residual* in vol units. This module
converts that to an implied overnight log-return ``pred * sigma`` (same as
``generate.py``) and scores:

- direction vs realized ``r_on = log(open_{t+1}) - log(close_t)``
- implied next-open vs actual next open

Default recipe is the PR #5 overnight skip (rank-target ridge, ``no_long_ts``).
PR #7 levers stay off unless a checkpoint documents them.

Train-only accuracy readouts (affine residual→raw overnight, TS overnight
ridge, sign ridge, ADV sleeve, confidence slices) are fit on TRAIN and gated
on locked VAL. They do not replace the residual CS skip. Next open is never a
feature.
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
            if TURNOVER_COL is not None and TURNOVER_COL < raw.shape[1]:
                turn = float(raw[i, TURNOVER_COL])
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
    df["pred_r"] = df["pred"] * df["scale"]
    df["implied_open"] = df["close"] * np.exp(df["pred_r"])
    # resid = y * scale = r_on - beta * r_hedge. Adding the realized hedge
    # term is *not* a live next-open forecast (uses next hedge open).
    hedge = df["r_on"] - df["y"] * df["scale"]
    df["implied_open_given_hedge"] = df["close"] * np.exp(df["pred_r"] + hedge)
    if return_features:
        return df, np.stack(feat_rows, axis=0).astype(np.float64)
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
                    f"MAE ${_as_float(t.get('mae_usd')):.4f} / "
                    f"{100.0 * _as_float(t.get('mae_pct')):.4f}%  "
                    f"vs residual*sigma {hit:.1f}% / "
                    f"${_as_float(price.get('mae')):.4f} / "
                    f"{100.0 * _as_float(pct.get('mae')):.4f}%  "
                    f"vs zero-move ${_as_float(zero_d.get('mae')):.4f} / "
                    f"{100.0 * _as_float(zero_p.get('mae')):.4f}%",
                    f"  VAL dir={_as_float(v.get('dir_pct')):.1f}%  "
                    f"MAE%={100.0 * _as_float(v.get('mae_pct')):.4f}  "
                    f"a={cal.get('a')} b={cal.get('b')}  "
                    f"promote_dir={promo.get('direction')!r} promote_mae={promo.get('price')!r}",
                    "  Direction above 51% here can mix residual signal with overnight drift "
                    "(train intercept). Unconditional up-rate / train-median gap is the "
                    "drift baseline; this is not a live P&L claim.",
                ]
            )
    ablate = payload.get("ablation")
    if ablate:
        lines.extend(["", format_ablation_table(ablate, payload.get("promotion") or {})])
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
) -> tuple[float, float]:
    """Train-only L1 affine: grid ``a``, ``b = median(r_on - a*pred_r)``.

    ``a=0`` recovers the train-median overnight gap (MAE-optimal constant).
    """
    p = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if y.size < 8:
        return 0.0, float(np.median(y) if y.size else 0.0)
    p_std = float(np.std(p))
    span = 3.0 if p_std < 1e-12 else max(2.0, 4.0 * float(np.std(y)) / max(p_std, 1e-12))
    grid = np.linspace(-span, span, max(9, int(n_grid)))
    best_a = 0.0
    best_b = float(np.median(y))
    best = float(np.mean(np.abs(y - best_b)))
    for a in grid:
        resid = y - float(a) * p
        b = float(np.median(resid))
        mae = float(np.mean(np.abs(resid - b)))
        if mae < best:
            best, best_a, best_b = mae, float(a), b
    # Fine grid around the winner.
    half = (grid[1] - grid[0]) if grid.size > 1 else 0.05
    fine = np.linspace(best_a - half, best_a + half, 21)
    for a in fine:
        resid = y - float(a) * p
        b = float(np.median(resid))
        mae = float(np.mean(np.abs(resid - b)))
        if mae < best:
            best, best_a, best_b = mae, float(a), b
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
        }
    direction = stats.get("direction") or {}
    overall = direction.get("overall") or {}
    pe = stats.get("price_error") or {}
    cs = stats.get("cs_ic") or {}
    long_only = direction.get("long_only_pred_positive") or {}
    return {
        "n": _as_float(stats.get("n_samples"), 0.0),
        "dir_pct": _as_float(overall.get("hit_rate_pct")),
        "dir_z": _as_float(overall.get("z_vs_half")),
        "mae_usd": _as_float((pe.get("dollars") or {}).get("mae")),
        "mae_pct": _as_float((pe.get("pct_of_prior_close") or {}).get("mae")),
        "zero_mae_pct": _as_float((pe.get("zero_pred_baseline_pct") or {}).get("mae")),
        "cs_ic": _as_float(cs.get("cs_ic")),
        "long_only_pct": _as_float(long_only.get("hit_rate_pct")),
        "up_pct": _as_float(direction.get("realized_overnight_up_pct")),
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
            "train-median drift, or % MAE below residual/zero/median). "
            "TEST is report-only."
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
        inf["year"] = float(year)
        inf["n_dates"] = float(sub["date"].nunique())
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
        "  variant                         val dir%   val MAE%   test dir%  "
        "test MAE%  test MAE$  vs PR8  gate",
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
        vs = ""
        lines.append(
            f"  {name:<30} {_as_float(v.get('dir_pct')):8.2f} "
            f"{100.0 * _as_float(v.get('mae_pct')):9.4f} "
            f"{_as_float(t.get('dir_pct')):9.2f} "
            f"{100.0 * _as_float(t.get('mae_pct')):9.4f} "
            f"{_as_float(t.get('mae_usd')):9.4f}  "
            f"{mark}"
        )
    lines.append(
        "  PR#8 locked TEST reference: 51.1% dir / $1.17 / 0.686% MAE "
        "(same window; residual*sigma)."
    )
    return "\n".join(lines)


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
            f"n={int(row.get('n') or 0)} z={float(row.get('z_vs_half') or float('nan')):.2f}"
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
            "train_median_gap": mu_med,
            "train_mean_gap": mu_mean,
            "hedge_mean": hedge_mu,
            "sign_affine": {"a": a_sgn, "b": b_sgn_aff},
            "hybrid_affine": {"a": a_hy, "b": b_hy},
            "ts_ridge_train_ic": float(ic_ts),
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
        "residual_plus_hedge_mean": (1.0, hedge_mu),
    }.get(default_name)
    payload["calibrate"] = {
        "name": default_name,
        "a": None if residual_affine is None else residual_affine[0],
        "b": None if residual_affine is None else residual_affine[1],
        "applies_to": "pred*sigma residual overnight log-return (generate.py --calibrate-json)",
        "params": next((r["params"] for r in rows if r["name"] == default_name), {}),
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
