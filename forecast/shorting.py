"""Overnight long-top / short-bottom book: VAL gate, TEST report-only.

The residual skip ranks names. This module scores:

- long sleeve overnight *up*-rate vs the unconditional overnight-up floor
- short sleeve overnight *down*-rate vs the unconditional overnight-down floor
- combined LS live_locate (honest locate/borrow) vs live_long_only on the
  same window: unlevered net IR, max DD, mean cost bp

Promote LS over long-only only on locked VAL. Locked TEST is report-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecast.accuracy import (
    DIR_LIFT,
    PROMOTED_SKIP,
    _frame_for_split,
    fit_promoted_overnight_skip,
    load_split_px,
    overnight_skip_data_config,
    score_eval_frame,
)
from forecast.backtest import book_pnl
from forecast.data import build_datasets
from forecast.overnight import (
    LIVE_BUNDLE,
    LIVE_LOCATE_BUNDLE,
    LIVE_LONG_ONLY_BUNDLE,
    LS_HAIRCUT_EXPERIMENT,
    OVERNIGHT_FORMULA,
    PAPER_BUNDLE,
    formula_log_line,
)

# Sleeve excess is printed in percentage points. DIR_LIFT is a fraction.
SHORT_EXCESS_LIFT_PP = 100.0 * float(DIR_LIFT)  # 0.2 pp
# Unlevered net IR: LS must beat long-only by this much on VAL.
IR_LIFT = 0.05
# Long-only book: VAL IR lift vs live_long_only q20, and max-DD not much worse.
LO_IR_LIFT = 0.05
LO_DD_TOL = 0.05

_BOOK_COST_KEYS = (
    "round_trip_bps",
    "open_auction_bps",
    "moc_bps",
    "moo_bps",
    "session_exit_bps",
    "borrow_bps",
    "hedge_cost_bps",
    "impact_vol_k",
    "thin_mult",
    "thin_pctile",
    "locate_pctile",
    "ex_post_gap_k",
)

_SERIES_KEYS = {
    "net",
    "gross",
    "cs_ic",
    "weights",
    "leverage",
    "unlevered_net",
}

BOOK_VARIANTS: tuple[tuple[str, bool, dict[str, Any]], ...] = (
    ("live_locate", False, LIVE_LOCATE_BUNDLE),
    ("live", False, LIVE_BUNDLE),
    ("live_long_only", True, LIVE_LONG_ONLY_BUNDLE),
    ("paper_ls", False, PAPER_BUNDLE),
)

DESKTOP_COMMANDS = """\
# Yahoo liquid tape is local (cloud VM has no tape). After the overnight skip:
python -m forecast.download --universe liquid --source yahoo --replace --interval daily
python -m forecast.training --universe liquid --interval daily --skip-only \\
  --label-return overnight --checkpoint-dir checkpoints/forecast_ridge_overnight

# VAL-gated LS vs long-only (prints TEST report-only; promote on VAL only)
python scripts/overnight_shorting.py --data-dir data --universe liquid \\
    --json checkpoints/forecast_ridge_overnight/shorting.json

# Honest LS (live_locate locate/borrow) vs long-only on the same scores
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs
# unconstrained shorts (old live; not the honest default)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --cost-bundle live --compare-long-only
# long-only live (no locate, borrow=0) — this is the default live book
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only
# VAL-gated LS haircut experiment (NOT default): HTB shorts at half size, short NAV 0.30
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --ls-haircut-experiment
"""


def frame_to_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot a name-date eval frame onto a datetime x symbol panel."""
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(
        np.datetime64("1970-01-01")
        + df["date"].to_numpy(dtype=np.int64).astype("timedelta64[D]")
    )
    tmp = pd.DataFrame(
        {
            "dt": dates,
            "symbol": df["symbol"].astype(str),
            "v": df[value_col].to_numpy(dtype=np.float64),
        }
    )
    wide = tmp.pivot_table(index="dt", columns="symbol", values="v", aggfunc="last")
    return wide.sort_index()


def scalar_book(stats: dict[str, Any]) -> dict[str, Any]:
    """Drop path objects so the payload is JSON-safe."""
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if key in _SERIES_KEYS:
            continue
        if hasattr(value, "iloc"):
            continue
        out[key] = value
    sleeve = stats.get("sleeve") or {}
    if isinstance(sleeve, dict):
        out["sleeve"] = dict(sleeve)
    parts = stats.get("cost_parts") or {}
    if isinstance(parts, dict):
        out["cost_parts"] = {str(k): float(v) if _is_num(v) else v for k, v in parts.items()}
    return out


def _is_num(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value))) or np.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _bundle_costs(bundle: dict[str, Any]) -> dict[str, float]:
    return {k: float(bundle[k]) for k in _BOOK_COST_KEYS if k in bundle}


def _run_overnight_book(
    pred: pd.DataFrame,
    y: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    long_only: bool,
    min_names: int,
    vol_target: float,
    overnight_r: pd.DataFrame | None = None,
    turnover_z: pd.DataFrame | None = None,
    vol_level: pd.DataFrame | None = None,
    quantile: float = 0.2,
    locate_haircut: float = 1.0,
    locate_frac: float = 1.0,
    max_short_gross: float = 0.5,
    weighting: str = "quantile",
    adv_floor_pctile: float = 0.0,
) -> dict[str, Any]:
    stats = book_pnl(
        pred,
        y,
        quantile=float(quantile),
        weighting=str(weighting),
        holding="overnight",
        hold_halflife=0.0,
        vol_target=float(vol_target),
        causal_vol=True,
        lever_cap=3.0,
        min_names=int(min_names),
        long_only=bool(long_only),
        locate_haircut=float(locate_haircut),
        locate_frac=float(locate_frac),
        max_short_gross=float(max_short_gross),
        adv_floor_pctile=float(adv_floor_pctile),
        overnight_r=overnight_r,
        turnover_z=turnover_z,
        vol_level=vol_level,
        **_bundle_costs(bundle),
    )
    return scalar_book(stats)


def val_knob_grid(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """VAL-only quantile / locate / short-gross grid. TEST never enters."""
    empty = {"rows": [], "best": {}, "best_ls": {}, "lo_best": {}}
    if df.empty:
        return empty
    pred = frame_to_wide(df, "pred")
    y = frame_to_wide(df, "y")
    r_on = frame_to_wide(df, "r_on")
    tz = frame_to_wide(df, "turnover_z") if "turnover_z" in df.columns else None
    vol = frame_to_wide(df, "vol_level") if "vol_level" in df.columns else None
    if pred.empty or pred.shape[1] < 2:
        return empty
    rows: list[dict[str, Any]] = []
    for q in (0.15, 0.20, 0.30):
        lo = _run_overnight_book(
            pred,
            y,
            bundle=LIVE_LONG_ONLY_BUNDLE,
            long_only=True,
            min_names=min_names,
            vol_target=vol_target,
            overnight_r=r_on,
            turnover_z=tz,
            vol_level=vol,
            quantile=q,
        )
        rows.append(
            {
                "name": f"live_long_only_q{int(100 * q)}",
                "kind": "long_only",
                "quantile": q,
                "locate_haircut": 0.0,
                "max_short_gross": 0.0,
                "unlevered_net_ir": lo.get("unlevered_net_ir"),
                "unlevered_max_dd": lo.get("unlevered_max_dd"),
                "mean_cost_unlev_bp": lo.get("mean_cost_unlev_bp"),
                "mean_short_nav": lo.get("mean_short_nav"),
            }
        )
        for haircut in (0.5, 1.0):
            for cap in (0.30, 0.50):
                ls = _run_overnight_book(
                    pred,
                    y,
                    bundle=LIVE_LOCATE_BUNDLE,
                    long_only=False,
                    min_names=min_names,
                    vol_target=vol_target,
                    overnight_r=r_on,
                    turnover_z=tz,
                    vol_level=vol,
                    quantile=q,
                    locate_haircut=haircut,
                    max_short_gross=cap,
                )
                rows.append(
                    {
                        "name": (
                            f"live_locate_q{int(100 * q)}"
                            f"_h{haircut:.1f}_s{cap:.2f}"
                        ),
                        "kind": "live_locate",
                        "quantile": q,
                        "locate_haircut": haircut,
                        "max_short_gross": cap,
                        "unlevered_net_ir": ls.get("unlevered_net_ir"),
                        "unlevered_max_dd": ls.get("unlevered_max_dd"),
                        "mean_cost_unlev_bp": ls.get("mean_cost_unlev_bp"),
                        "mean_short_nav": ls.get("mean_short_nav"),
                    }
                )
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            str(r.get("name")),
        )
    )
    lo_rows = [r for r in rows if r.get("kind") == "long_only"]
    ls_rows = [r for r in rows if r.get("kind") == "live_locate"]
    return {
        "rows": rows,
        "best": dict(rows[0]) if rows else {},
        "best_ls": dict(ls_rows[0]) if ls_rows else {},
        "lo_best": dict(lo_rows[0]) if lo_rows else {},
        "experiment": dict(LS_HAIRCUT_EXPERIMENT),
    }


def _lo_row_name(weighting: str, quantile: float, adv_floor: float) -> str:
    if str(weighting) == "rank":
        return f"lo_rank_adv{int(100 * adv_floor)}"
    return f"lo_q{int(100 * quantile)}_adv{int(100 * adv_floor)}"


def val_long_only_grid(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """VAL-only live_long_only grid: quantile, rank, ADV floor. TEST never enters."""
    empty = {"rows": [], "best": {}, "baseline": {}, "spec": {}}
    if df.empty:
        return empty
    pred = frame_to_wide(df, "pred")
    y = frame_to_wide(df, "y")
    r_on = frame_to_wide(df, "r_on")
    tz = frame_to_wide(df, "turnover_z") if "turnover_z" in df.columns else None
    vol = frame_to_wide(df, "vol_level") if "vol_level" in df.columns else None
    if pred.empty or pred.shape[1] < 2:
        return empty
    specs: list[tuple[str, float, float]] = []
    for q in (0.10, 0.15, 0.20, 0.30):
        for adv in (0.0, 0.33, 0.67):
            specs.append(("quantile", q, adv))
    for adv in (0.0, 0.33, 0.67):
        specs.append(("rank", 0.2, adv))
    rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] = {}
    for weighting, q, adv in specs:
        floor_min = int(min_names)
        if adv > 0:
            floor_min = max(5, min(int(min_names), 8))
        stats = _run_overnight_book(
            pred,
            y,
            bundle=LIVE_LONG_ONLY_BUNDLE,
            long_only=True,
            min_names=floor_min,
            vol_target=vol_target,
            overnight_r=r_on,
            turnover_z=tz,
            vol_level=vol,
            quantile=q,
            weighting=weighting,
            adv_floor_pctile=adv,
        )
        row = {
            "name": _lo_row_name(weighting, q, adv),
            "kind": "long_only",
            "weighting": weighting,
            "quantile": q,
            "adv_floor_pctile": adv,
            "min_names": floor_min,
            "unlevered_net_ir": stats.get("unlevered_net_ir"),
            "unlevered_max_dd": stats.get("unlevered_max_dd"),
            "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
            "mean_long_nav": stats.get("mean_long_nav"),
            "n_dates": stats.get("n_dates"),
        }
        rows.append(row)
        if weighting == "quantile" and abs(q - 0.2) < 1e-12 and adv == 0.0:
            baseline = dict(row)
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            _as_float(r.get("unlevered_max_dd"), default=-1e9),
            str(r.get("name")),
        )
    )
    return {
        "rows": rows,
        "best": dict(rows[0]) if rows else {},
        "baseline": baseline,
    }


def decide_lo_promote(grid: dict[str, Any]) -> dict[str, Any]:
    """VAL-only. Promote a long-only spec vs live_long_only q20 / adv0."""
    baseline = dict(grid.get("baseline") or {})
    best = dict(grid.get("best") or {})
    ir_base = _as_float(baseline.get("unlevered_net_ir"))
    ir_best = _as_float(best.get("unlevered_net_ir"))
    dd_base = _as_float(baseline.get("unlevered_max_dd"))
    dd_best = _as_float(best.get("unlevered_max_dd"))
    ir_delta = (
        float(ir_best - ir_base)
        if np.isfinite(ir_best) and np.isfinite(ir_base)
        else float("nan")
    )
    dd_delta = (
        float(dd_best - dd_base)
        if np.isfinite(dd_best) and np.isfinite(dd_base)
        else float("nan")
    )
    same = str(best.get("name") or "") == str(baseline.get("name") or "")
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    promote = bool(ir_ok and dd_ok and not same)
    spec = {
        "weighting": baseline.get("weighting", "quantile"),
        "quantile": baseline.get("quantile", 0.2),
        "adv_floor_pctile": baseline.get("adv_floor_pctile", 0.0),
    }
    if promote:
        spec = {
            "weighting": best.get("weighting", "quantile"),
            "quantile": best.get("quantile", 0.2),
            "adv_floor_pctile": best.get("adv_floor_pctile", 0.0),
        }
        reason = (
            f"PROMOTE long-only {best.get('name')}: VAL unlev net IR "
            f"{ir_best:+.3f} vs q20 {ir_base:+.3f} (delta {ir_delta:+.3f} >= "
            f"{LO_IR_LIFT:.2f}) and max DD {dd_best:+.3f} vs {dd_base:+.3f}."
        )
    elif same or not ir_ok:
        reason = (
            "NO LIFT: no long-only VAL spec beats live_long_only q20 by "
            f"{LO_IR_LIFT:.2f} unlev net IR "
            f"(best {best.get('name')} {ir_best:+.3f} vs q20 {ir_base:+.3f}, "
            f"delta {ir_delta:+.3f}). Keep q20 / adv0."
        )
    else:
        reason = (
            f"NO PROMOTE: {best.get('name')} IR lift {ir_delta:+.3f} but max DD "
            f"{dd_best:+.3f} is worse than q20 {dd_base:+.3f} by more than "
            f"{LO_DD_TOL:.2f}. Keep q20 / adv0."
        )
    return {
        "promote_lo": promote,
        "gated_on": "val",
        "reason": reason,
        "baseline": baseline,
        "best": best,
        "spec": spec,
        "ir_delta": ir_delta,
        "dd_delta": dd_delta,
        "ir_lift": LO_IR_LIFT,
        "dd_tol": LO_DD_TOL,
    }


def decide_ls_experiment(val: dict[str, Any], experiment_book: dict[str, Any]) -> dict[str, Any]:
    """VAL-only haircut experiment. Never changes the default live book."""
    lo = (val.get("books") or {}).get("live_long_only") or {}
    ir_ls = _as_float(experiment_book.get("unlevered_net_ir"))
    ir_lo = _as_float(lo.get("unlevered_net_ir"))
    ir_delta = (
        float(ir_ls - ir_lo)
        if np.isfinite(ir_ls) and np.isfinite(ir_lo)
        else float("nan")
    )
    short = (val.get("sleeves") or {}).get("short_bottom20") or {}
    short_excess = _as_float(short.get("excess_pp"))
    short_skill = bool(np.isfinite(short_excess) and short_excess >= SHORT_EXCESS_LIFT_PP)
    beats = bool(short_skill and np.isfinite(ir_delta) and ir_delta >= IR_LIFT)
    return {
        "name": "ls_haircut_experiment",
        "gated_on": "val",
        "promote_as_default": False,
        "would_beat_long_only_on_val": beats,
        "reason": (
            "EXPERIMENT ONLY (not default). "
            + (
                f"Haircut 0.5 / short NAV 0.30 beats long-only on VAL "
                f"(unlev net IR {ir_ls:+.3f} vs {ir_lo:+.3f}, delta {ir_delta:+.3f}). "
                "Keep live_locate skip as the honest default until liquid VAL agrees."
                if beats
                else (
                    f"Haircut 0.5 / short NAV 0.30 does not beat long-only on VAL "
                    f"(unlev net IR {ir_ls:+.3f} vs {ir_lo:+.3f}, delta {ir_delta:+.3f})."
                )
            )
        ),
        "spec": dict(LS_HAIRCUT_EXPERIMENT),
        "unlevered_net_ir": ir_ls,
        "ir_lo_unlev_net": ir_lo,
        "ir_delta": ir_delta,
        "unlevered_max_dd": experiment_book.get("unlevered_max_dd"),
        "mean_cost_unlev_bp": experiment_book.get("mean_cost_unlev_bp"),
        "mean_short_nav": experiment_book.get("mean_short_nav"),
        "short_skill": short_skill,
    }


def split_shorting_metrics(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Sleeves vs raw r_on plus residual-PnL live books on one locked split."""
    if df.empty:
        return {
            "n_samples": 0,
            "n_dates": 0,
            "n_names": 0,
            "date_start": "",
            "date_end": "",
            "sleeves": {},
            "cs_ic": {},
            "books": {},
            "uncond_up_pct": float("nan"),
            "uncond_down_pct": float("nan"),
        }
    scored = score_eval_frame(df, min_names=min_names)
    book = scored.get("book") or {}
    pred = frame_to_wide(df, "pred")
    y = frame_to_wide(df, "y")
    r_on = frame_to_wide(df, "r_on")
    tz = frame_to_wide(df, "turnover_z") if "turnover_z" in df.columns else None
    vol = frame_to_wide(df, "vol_level") if "vol_level" in df.columns else None
    books: dict[str, Any] = {}
    if len(pred) and pred.shape[1] >= 2:
        for name, long_only, bundle in BOOK_VARIANTS:
            books[name] = _run_overnight_book(
                pred,
                y,
                bundle=bundle,
                long_only=bool(long_only),
                min_names=int(min_names),
                vol_target=float(vol_target),
                overnight_r=r_on,
                turnover_z=tz,
                vol_level=vol,
            )
    short20 = book.get("short_bottom20") or {}
    direction = scored.get("direction") or {}
    return {
        "n_samples": scored.get("n_samples"),
        "n_dates": scored.get("n_dates"),
        "n_names": scored.get("n_names"),
        "date_start": scored.get("date_start"),
        "date_end": scored.get("date_end"),
        "sleeves": {
            "long_only_top20": book.get("long_only_top20") or {},
            "short_bottom20": short20,
            "long_only_top30": book.get("long_only_top30") or {},
            "short_bottom30": book.get("short_bottom30") or {},
        },
        "cs_ic": scored.get("cs_ic") or {},
        "books": books,
        "uncond_up_pct": direction.get("realized_overnight_up_pct", float("nan")),
        "uncond_down_pct": short20.get("uncond_down_pct", float("nan")),
    }


def decide_ls_promote(val: dict[str, Any]) -> dict[str, Any]:
    """VAL-only. TEST must never enter this function."""
    sleeves = val.get("sleeves") or {}
    books = val.get("books") or {}
    short = sleeves.get("short_bottom20") or {}
    ls = books.get("live_locate") or {}
    lo = books.get("live_long_only") or {}
    short_excess = _as_float(short.get("excess_pp"))
    ir_ls = _as_float(ls.get("unlevered_net_ir"))
    ir_lo = _as_float(lo.get("unlevered_net_ir"))
    ir_delta = (
        float(ir_ls - ir_lo)
        if np.isfinite(ir_ls) and np.isfinite(ir_lo)
        else float("nan")
    )
    short_skill = bool(np.isfinite(short_excess) and short_excess >= SHORT_EXCESS_LIFT_PP)
    ir_beats = bool(np.isfinite(ir_delta) and ir_delta >= IR_LIFT)
    promote = bool(short_skill and ir_beats)
    if not short_skill:
        reason = (
            "NO PROMOTE: short sleeve has no skill vs overnight-down floor "
            f"(VAL excess {short_excess:+.2f} pp < {SHORT_EXCESS_LIFT_PP:.1f} pp). "
            "Default live book stays long-only."
        )
        default = "live_long_only"
    elif not ir_beats:
        reason = (
            "NO PROMOTE LS: short sleeve has skill but live locate/borrow wipe "
            f"the IR edge vs long-only (VAL unlev net IR {ir_ls:+.3f} vs "
            f"{ir_lo:+.3f}, delta {ir_delta:+.3f} < {IR_LIFT:.2f}). "
            "Default live book stays long-only."
        )
        default = "live_long_only"
    else:
        reason = (
            "PROMOTE live_locate LS: VAL short-sleeve down excess and "
            f"live_locate unlev net IR beat long-only "
            f"(excess {short_excess:+.2f} pp, IR {ir_ls:+.3f} vs {ir_lo:+.3f})."
        )
        default = "live_locate"
    return {
        "promote_ls": promote,
        "default_book": default,
        "reason": reason,
        "gated_on": "val",
        "short_excess_pp": short_excess,
        "short_excess_lift_pp": SHORT_EXCESS_LIFT_PP,
        "ir_ls_unlev_net": ir_ls,
        "ir_lo_unlev_net": ir_lo,
        "ir_delta": ir_delta,
        "ir_lift": IR_LIFT,
        "short_skill": short_skill,
        "ir_beats": ir_beats,
    }


def evaluate_overnight_shorting(
    data_dir: str,
    universe: str = "liquid",
    *,
    log_fn: Any | None = print,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Fit the promoted overnight skip on TRAIN; gate LS on VAL; report TEST."""
    cfg = overnight_skip_data_config(data_dir, universe)
    if log_fn:
        log_fn(formula_log_line("overnight"))
        log_fn(
            f"recipe=promoted overnight skip {PROMOTED_SKIP}  "
            "LS vs long-only: sleeves + live_locate costs; promote on locked VAL only"
        )
    bundle = build_datasets(cfg, log_fn=log_fn)
    weights, bias, train_ic = fit_promoted_overnight_skip(bundle)
    min_names = int(bundle.get("cs_min_names", 30))
    px, missing_px = load_split_px(bundle, cfg, log_fn=log_fn)
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        df, _x = _frame_for_split(bundle, split, px, weights, bias, min_names)
        frames[split] = df
        if log_fn:
            log_fn(
                f"{split}: {len(df)} name-dates, "
                f"{df['date'].nunique() if not df.empty else 0} dates"
            )
    val = split_shorting_metrics(frames["val"], min_names=min_names, vol_target=vol_target)
    test = split_shorting_metrics(frames["test"], min_names=min_names, vol_target=vol_target)
    promo = decide_ls_promote(val)
    grid = val_knob_grid(frames["val"], min_names=min_names, vol_target=vol_target)
    lo_grid = val_long_only_grid(frames["val"], min_names=min_names, vol_target=vol_target)
    lo_promo = decide_lo_promote(lo_grid)
    exp_book = {}
    if not frames["val"].empty:
        pred = frame_to_wide(frames["val"], "pred")
        y = frame_to_wide(frames["val"], "y")
        r_on = frame_to_wide(frames["val"], "r_on")
        tz = frame_to_wide(frames["val"], "turnover_z") if "turnover_z" in frames["val"].columns else None
        vol = frame_to_wide(frames["val"], "vol_level") if "vol_level" in frames["val"].columns else None
        if len(pred) and pred.shape[1] >= 2:
            exp_book = _run_overnight_book(
                pred,
                y,
                bundle=LIVE_LOCATE_BUNDLE,
                long_only=False,
                min_names=min_names,
                vol_target=vol_target,
                overnight_r=r_on,
                turnover_z=tz,
                vol_level=vol,
                quantile=float(LS_HAIRCUT_EXPERIMENT["quantile"]),
                locate_haircut=float(LS_HAIRCUT_EXPERIMENT["locate_haircut"]),
                max_short_gross=float(LS_HAIRCUT_EXPERIMENT["max_short_gross"]),
            )
    ls_exp = decide_ls_experiment(val, exp_book)
    test_lo_promoted = {}
    spec = lo_promo.get("spec") or {}
    if frames["test"].empty is False and spec:
        pred_t = frame_to_wide(frames["test"], "pred")
        y_t = frame_to_wide(frames["test"], "y")
        r_t = frame_to_wide(frames["test"], "r_on")
        tz_t = (
            frame_to_wide(frames["test"], "turnover_z")
            if "turnover_z" in frames["test"].columns
            else None
        )
        vol_t = (
            frame_to_wide(frames["test"], "vol_level")
            if "vol_level" in frames["test"].columns
            else None
        )
        if len(pred_t) and pred_t.shape[1] >= 2:
            test_lo_promoted = _run_overnight_book(
                pred_t,
                y_t,
                bundle=LIVE_LONG_ONLY_BUNDLE,
                long_only=True,
                min_names=min_names,
                vol_target=vol_target,
                overnight_r=r_t,
                turnover_z=tz_t,
                vol_level=vol_t,
                quantile=float(spec.get("quantile") or 0.2),
                weighting=str(spec.get("weighting") or "quantile"),
                adv_floor_pctile=float(spec.get("adv_floor_pctile") or 0.0),
            )
    cuts: dict[str, Any] = {}
    if bundle.get("meta"):
        cuts = {
            "train_end": bundle["meta"][0].get("train_end"),
            "val_end": bundle["meta"][0].get("val_end"),
        }
    return {
        "estimand": (
            "overnight residual skip; long top / short bottom vs live_long_only"
        ),
        "formula": OVERNIGHT_FORMULA,
        "recipe": dict(PROMOTED_SKIP),
        "levers": "default overnight skip; locate/borrow honest via live_locate",
        "split": "locked VAL gate; locked TEST report-only",
        "cs_min_names": min_names,
        "n_trading_names": int(bundle.get("n_trading_names") or 0),
        "missing_price_files": missing_px,
        "calendar_cuts": cuts,
        "train_cs_ic": float(train_ic),
        "val": val,
        "test": test,
        "val_grid": grid,
        "val_long_only_grid": lo_grid,
        "promotion": promo,
        "lo_promotion": lo_promo,
        "ls_experiment": ls_exp,
        "test_long_only_promoted": test_lo_promoted,
        "desktop_commands": DESKTOP_COMMANDS,
    }


def format_shorting_report(payload: dict[str, Any]) -> str:
    promo = payload.get("promotion") or {}
    lo_promo = payload.get("lo_promotion") or {}
    ls_exp = payload.get("ls_experiment") or {}
    lines = [
        "OVERNIGHT LONG-SHORT vs LONG-ONLY (VAL gate, TEST report-only)",
        f"  recipe: {payload.get('recipe')}  {payload.get('levers')}",
        f"  calendar: {payload.get('calendar_cuts')}",
        f"  train CS IC={_fmt(payload.get('train_cs_ic'), '+.4f')}",
        "",
        _split_block("LOCKED VAL (gate)", payload.get("val") or {}),
        "",
        f"PROMOTE LS? {'YES' if promo.get('promote_ls') else 'NO'}",
        f"  default live book = {promo.get('default_book')}",
        f"  {promo.get('reason')}",
        "",
        f"PROMOTE LONG-ONLY KNOBS? {'YES' if lo_promo.get('promote_lo') else 'NO'}",
        f"  spec = {lo_promo.get('spec')}",
        f"  {lo_promo.get('reason')}",
        "",
        _lo_grid_block(payload.get("val_long_only_grid") or {}),
        "",
        f"LS HAIRCUT EXPERIMENT (not default)  would_beat_LO_on_VAL="
        f"{ls_exp.get('would_beat_long_only_on_val')}",
        f"  spec = {ls_exp.get('spec')}  "
        f"IR {_fmt(ls_exp.get('unlevered_net_ir'), '+.3f')}  "
        f"vs LO {_fmt(ls_exp.get('ir_lo_unlev_net'), '+.3f')}  "
        f"delta {_fmt(ls_exp.get('ir_delta'), '+.3f')}  "
        f"maxDD {_fmt(ls_exp.get('unlevered_max_dd'), '+.3f')}  "
        f"cost {_fmt(ls_exp.get('mean_cost_unlev_bp'), '.1f')} bp  "
        f"short NAV {_fmt(ls_exp.get('mean_short_nav'), '.3f')}",
        f"  {ls_exp.get('reason')}",
        "",
        _grid_block(payload.get("val_grid") or {}),
        "",
        _split_block("LOCKED TEST (report-only; not a gate)", payload.get("test") or {}),
    ]
    test_lo = payload.get("test_long_only_promoted") or {}
    if test_lo:
        lines.extend(
            [
                "",
                "TEST long-only promoted spec (report-only)",
                f"  unlev net IR {_fmt(test_lo.get('unlevered_net_ir'), '+.3f')}  "
                f"maxDD {_fmt(test_lo.get('unlevered_max_dd'), '+.3f')}  "
                f"cost {_fmt(test_lo.get('mean_cost_unlev_bp'), '.1f')} bp  "
                f"q={test_lo.get('quantile')}  weighting={test_lo.get('weighting')}  "
                f"adv_floor={test_lo.get('adv_floor_pctile')}",
            ]
        )
    lines.extend(
        [
            "",
            "DESKTOP (Yahoo liquid tape — cloud VM has none):",
            DESKTOP_COMMANDS.rstrip(),
        ]
    )
    return "\n".join(lines)


def _lo_grid_block(grid: dict[str, Any]) -> str:
    rows = list(grid.get("rows") or [])
    best = grid.get("best") or {}
    base = grid.get("baseline") or {}
    lines = [
        "VAL LONG-ONLY GRID (live_long_only; quantile/rank/ADV floor; not a TEST look)",
        f"  baseline q20 adv0 = {base.get('name')}  "
        f"IR {_fmt(base.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(base.get('unlevered_max_dd'), '+.3f')}  "
        f"cost {_fmt(base.get('mean_cost_unlev_bp'), '.1f')} bp",
        f"  best = {best.get('name')}  "
        f"IR {_fmt(best.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(best.get('unlevered_max_dd'), '+.3f')}",
        "  name                    IR      maxDD   cost bp",
    ]
    for row in rows[:8]:
        lines.append(
            f"  {str(row.get('name')):<22}  "
            f"{_fmt(row.get('unlevered_net_ir'), '+.3f'):>7}  "
            f"{_fmt(row.get('unlevered_max_dd'), '+.3f'):>7}  "
            f"{_fmt(row.get('mean_cost_unlev_bp'), '.1f'):>7}"
        )
    if len(rows) > 8:
        lines.append(f"  ... {len(rows) - 8} more rows in JSON")
    return "\n".join(lines)


def _split_block(title: str, split: dict[str, Any]) -> str:
    sleeves = split.get("sleeves") or {}
    long20 = sleeves.get("long_only_top20") or {}
    short20 = sleeves.get("short_bottom20") or {}
    books = split.get("books") or {}
    up = _as_float(split.get("uncond_up_pct", long20.get("uncond_up_pct")))
    down = _as_float(split.get("uncond_down_pct", short20.get("uncond_down_pct")))
    lines = [
        title,
        f"  coverage: n={split.get('n_samples')} name-dates, "
        f"{split.get('n_dates')} dates, {split.get('n_names')} names  "
        f"{split.get('date_start')} -> {split.get('date_end')}",
        f"  CS IC={_fmt((split.get('cs_ic') or {}).get('cs_ic'), '+.4f')}  "
        f"t={_fmt((split.get('cs_ic') or {}).get('cs_ic_tstat'), '.2f')}",
        "",
        "  sleeve                 rate     floor    excess     n",
        f"  long top20 overnight up  "
        f"{_pct(long20.get('up_pct', long20.get('hit_pct')))}  "
        f"{_pct(up)}  "
        f"{_pp(long20.get('excess_pp'))}  "
        f"{_n(long20.get('n'))}",
        f"  short bot20 overnight dn "
        f"{_pct(short20.get('down_pct', short20.get('hit_pct')))}  "
        f"{_pct(down)}  "
        f"{_pp(short20.get('excess_pp'))}  "
        f"{_n(short20.get('n'))}",
        "",
        "  book            unlev net IR  lev net IR  unlev maxDD  lev maxDD  "
        "cost bp  long NAV  short NAV",
    ]
    for name in ("live_locate", "live", "live_long_only", "paper_ls"):
        row = books.get(name) or {}
        if not row:
            continue
        lines.append(
            f"  {name:<14}  "
            f"{_fmt(row.get('unlevered_net_ir'), '+.3f'):>11}  "
            f"{_fmt(row.get('net_ir'), '+.3f'):>9}  "
            f"{_fmt(row.get('unlevered_max_dd'), '+.3f'):>11}  "
            f"{_fmt(row.get('max_dd'), '+.3f'):>8}  "
            f"{_fmt(row.get('mean_cost_unlev_bp'), '.1f'):>7}  "
            f"{_fmt(row.get('mean_long_nav'), '.3f'):>8}  "
            f"{_fmt(row.get('mean_short_nav'), '.3f'):>9}"
        )
    ls = books.get("live_locate") or {}
    lo = books.get("live_long_only") or {}
    if ls and lo:
        d_ir = _as_float(ls.get("unlevered_net_ir")) - _as_float(lo.get("unlevered_net_ir"))
        d_dd = _as_float(ls.get("unlevered_max_dd")) - _as_float(lo.get("unlevered_max_dd"))
        d_bp = _as_float(ls.get("mean_cost_unlev_bp")) - _as_float(
            lo.get("mean_cost_unlev_bp")
        )
        lines.append(
            f"  LS minus LO     "
            f"{_fmt(d_ir, '+.3f'):>11}  "
            f"{'':>9}  "
            f"{_fmt(d_dd, '+.3f'):>11}  "
            f"{'':>8}  "
            f"{_fmt(d_bp, '+.1f'):>7}"
        )
    return "\n".join(lines)


def _grid_block(grid: dict[str, Any]) -> str:
    rows = list(grid.get("rows") or [])
    best = grid.get("best") or {}
    best_ls = grid.get("best_ls") or {}
    lo_best = grid.get("lo_best") or {}
    lines = [
        "VAL KNOB GRID (quantile / locate haircut / max short NAV; not a TEST look)",
        f"  best overall = {best.get('name')}  "
        f"unlev net IR {_fmt(best.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(best.get('unlevered_max_dd'), '+.3f')}  "
        f"cost {_fmt(best.get('mean_cost_unlev_bp'), '.1f')} bp",
        f"  best long-only = {lo_best.get('name')}  "
        f"IR {_fmt(lo_best.get('unlevered_net_ir'), '+.3f')}",
        f"  best live_locate = {best_ls.get('name')}  "
        f"IR {_fmt(best_ls.get('unlevered_net_ir'), '+.3f')}",
        "  name                          IR      maxDD   cost bp  short NAV",
    ]
    for row in rows[:8]:
        lines.append(
            f"  {str(row.get('name')):<28}  "
            f"{_fmt(row.get('unlevered_net_ir'), '+.3f'):>7}  "
            f"{_fmt(row.get('unlevered_max_dd'), '+.3f'):>7}  "
            f"{_fmt(row.get('mean_cost_unlev_bp'), '.1f'):>7}  "
            f"{_fmt(row.get('mean_short_nav'), '.3f'):>9}"
        )
    if len(rows) > 8:
        lines.append(f"  ... {len(rows) - 8} more rows in JSON")
    return "\n".join(lines)


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out


def _fmt(value: Any, spec: str) -> str:
    x = _as_float(value)
    if not np.isfinite(x):
        return "nan"
    return format(x, spec)


def _pct(value: Any) -> str:
    x = _as_float(value)
    return f"{x:6.1f}%" if np.isfinite(x) else "   nan%"


def _pp(value: Any) -> str:
    x = _as_float(value)
    return f"{x:+6.2f} pp" if np.isfinite(x) else "   nan pp"


def _n(value: Any) -> str:
    x = _as_float(value)
    return f"{int(x)}" if np.isfinite(x) else "0"
