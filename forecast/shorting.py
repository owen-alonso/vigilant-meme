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
from forecast.backtest import book_pnl, trailing_mean_cs_ic
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
# TEST veto only: promoted VAL spec may not fall this far below q20 TEST IR.
TEST_COLLAPSE = 0.05

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
# long-only live (no locate, borrow=0) — default live book after LS failed VAL
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only
# VAL-promoted long-only spec on synthetic (rank vs q20); TEST did not confirm
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --weighting rank
# long-only conviction / inv-vol (VAL refine: no lift vs q20 equal)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --long-size inv_vol --conf-pctile 0.5
# VAL-gated LS haircut experiment (NOT default): HTB shorts at half size, short NAV 0.30
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --ls-haircut-experiment
# causal trailing CS-IC trade gate (TRAIN-fit W,τ; default off until VAL promote)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --ic-gate-window 60 --ic-gate-tau 0.0
# causal Friday / weekend weekday mask (VAL-gated; default always-on)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --weekday-mask flat_friday
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --weekday-mask weekend_only
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --weekday-mask flat_monday
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
    long_size: str = "equal",
    conf_pctile: float = 0.0,
    ic_gate_window: int = 0,
    ic_gate_tau: float = 0.0,
    ic_gate_trail: pd.Series | None = None,
    weekday_mask: str = "always",
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
        long_size=str(long_size or "equal"),
        conf_pctile=float(conf_pctile),
        ic_gate_window=int(ic_gate_window or 0),
        ic_gate_tau=float(ic_gate_tau or 0.0),
        ic_gate_trail=ic_gate_trail,
        weekday_mask=str(weekday_mask or "always"),
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


def val_long_only_refine(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """VAL-only q-width × conviction × inv-vol. Overnight hold_hl stays 0."""
    empty = {"rows": [], "best": {}, "baseline": {}}
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
    baseline: dict[str, Any] = {}
    for q in (0.10, 0.15, 0.20, 0.30):
        for size in ("equal", "abs_pred", "inv_vol"):
            for conf in (0.0, 0.5):
                stats = _run_overnight_book(
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
                    weighting="quantile",
                    long_size=size,
                    conf_pctile=conf,
                )
                row = {
                    "name": f"lo_q{int(100 * q)}_{size}_c{int(100 * conf)}",
                    "kind": "long_only",
                    "weighting": "quantile",
                    "quantile": q,
                    "long_size": size,
                    "conf_pctile": conf,
                    "unlevered_net_ir": stats.get("unlevered_net_ir"),
                    "unlevered_max_dd": stats.get("unlevered_max_dd"),
                    "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
                    "n_dates": stats.get("n_dates"),
                }
                rows.append(row)
                if (
                    abs(q - 0.2) < 1e-12
                    and size == "equal"
                    and conf == 0.0
                ):
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
        "note": "overnight flatten forces hold_halflife=0; not a live knob",
    }


def decide_lo_refine(
    grid: dict[str, Any],
    *,
    test_baseline: dict[str, Any] | None = None,
    test_best: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """VAL selects. TEST vetoes collapse. TEST is never used to pick a winner."""
    val_promo = decide_lo_promote(grid)
    spec = dict(val_promo.get("spec") or {})
    spec.setdefault("long_size", (grid.get("baseline") or {}).get("long_size", "equal"))
    spec.setdefault("conf_pctile", (grid.get("baseline") or {}).get("conf_pctile", 0.0))
    best = dict(grid.get("best") or {})
    if val_promo.get("promote_lo"):
        spec["long_size"] = best.get("long_size", "equal")
        spec["conf_pctile"] = best.get("conf_pctile", 0.0)
    test_base_ir = _as_float((test_baseline or {}).get("unlevered_net_ir"))
    test_best_ir = _as_float((test_best or {}).get("unlevered_net_ir"))
    test_delta = (
        float(test_best_ir - test_base_ir)
        if np.isfinite(test_best_ir) and np.isfinite(test_base_ir)
        else float("nan")
    )
    collapse = bool(
        val_promo.get("promote_lo")
        and np.isfinite(test_delta)
        and test_delta < -TEST_COLLAPSE
    )
    promote = bool(val_promo.get("promote_lo") and not collapse)
    if not val_promo.get("promote_lo"):
        reason = val_promo.get("reason")
        spec = {
            "weighting": "quantile",
            "quantile": 0.2,
            "long_size": "equal",
            "conf_pctile": 0.0,
            "adv_floor_pctile": 0.0,
        }
    elif collapse:
        reason = (
            f"NO PROMOTE: VAL liked {best.get('name')} "
            f"(delta {val_promo.get('ir_delta'):+.3f}) but TEST collapsed "
            f"({test_best_ir:+.3f} vs q20 {test_base_ir:+.3f}, "
            f"delta {test_delta:+.3f} < -{TEST_COLLAPSE:.2f}). Keep q20 equal."
        )
        spec = {
            "weighting": "quantile",
            "quantile": 0.2,
            "long_size": "equal",
            "conf_pctile": 0.0,
            "adv_floor_pctile": 0.0,
        }
    else:
        reason = (
            f"{val_promo.get('reason')} TEST veto passed "
            f"({test_best_ir:+.3f} vs q20 {test_base_ir:+.3f}, "
            f"delta {test_delta:+.3f} >= -{TEST_COLLAPSE:.2f})."
        )
    return {
        **val_promo,
        "promote_lo": promote,
        "test_veto": collapse,
        "test_ir_delta": test_delta,
        "test_collapse": TEST_COLLAPSE,
        "reason": reason,
        "spec": spec,
        "test_baseline_ir": test_base_ir,
        "test_best_ir": test_best_ir,
    }


IC_GATE_WINDOWS = (20, 60, 120)
IC_GATE_TAUS = (-0.02, 0.0, 0.02, 0.04)
IC_GATE_COVER_TRAIN = 0.40
IC_GATE_COVER_VAL = 0.30


def _wide_from_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    pred = frame_to_wide(df, "pred")
    y = frame_to_wide(df, "y")
    r_on = frame_to_wide(df, "r_on")
    tz = frame_to_wide(df, "turnover_z") if "turnover_z" in df.columns else None
    vol = frame_to_wide(df, "vol_level") if "vol_level" in df.columns else None
    return pred, y, r_on, tz, vol


def _lo_q20_book(
    pred: pd.DataFrame,
    y: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float,
    overnight_r: pd.DataFrame | None,
    turnover_z: pd.DataFrame | None,
    vol_level: pd.DataFrame | None,
    ic_gate_window: int = 0,
    ic_gate_tau: float = 0.0,
    ic_gate_trail: pd.Series | None = None,
    weekday_mask: str = "always",
) -> dict[str, Any]:
    return _run_overnight_book(
        pred,
        y,
        bundle=LIVE_LONG_ONLY_BUNDLE,
        long_only=True,
        min_names=min_names,
        vol_target=vol_target,
        overnight_r=overnight_r,
        turnover_z=turnover_z,
        vol_level=vol_level,
        quantile=0.2,
        weighting="quantile",
        ic_gate_window=int(ic_gate_window),
        ic_gate_tau=float(ic_gate_tau),
        ic_gate_trail=ic_gate_trail,
        weekday_mask=str(weekday_mask or "always"),
    )


def fit_ic_gate_on_train(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Select (W, τ) on TRAIN only. VAL/TEST must never enter."""
    empty = {"rows": [], "chosen": {}, "baseline": {}, "fit_split": "train"}
    if df.empty:
        return empty
    pred, y, r_on, tz, vol = _wide_from_frame(df)
    if pred.empty or pred.shape[1] < 2:
        return empty
    base = _lo_q20_book(
        pred, y, min_names=min_names, vol_target=vol_target,
        overnight_r=r_on, turnover_z=tz, vol_level=vol,
    )
    rows: list[dict[str, Any]] = []
    chosen: dict[str, Any] = {}
    best_ir = -1e18
    for window in IC_GATE_WINDOWS:
        for tau in IC_GATE_TAUS:
            stats = _lo_q20_book(
                pred, y, min_names=min_names, vol_target=vol_target,
                overnight_r=r_on, turnover_z=tz, vol_level=vol,
                ic_gate_window=window, ic_gate_tau=tau,
            )
            cover = _as_float(stats.get("ic_gate_coverage"))
            ir = _as_float(stats.get("unlevered_net_ir"))
            row = {
                "name": f"ic_gate_W{window}_t{tau:+.2f}",
                "window": window,
                "tau": tau,
                "unlevered_net_ir": stats.get("unlevered_net_ir"),
                "unlevered_max_dd": stats.get("unlevered_max_dd"),
                "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
                "ic_gate_coverage": cover,
                "ic_gate_n_flat": stats.get("ic_gate_n_flat"),
            }
            rows.append(row)
            if (not np.isfinite(ir)) or (not np.isfinite(cover)):
                continue
            if cover < IC_GATE_COVER_TRAIN:
                continue
            if ir > best_ir + 1e-12:
                best_ir = ir
                chosen = dict(row)
    rows.sort(key=lambda r: (-_as_float(r.get("unlevered_net_ir"), default=-1e9), str(r.get("name"))))
    return {
        "rows": rows,
        "chosen": chosen,
        "baseline": {
            "name": "live_long_only_q20",
            "unlevered_net_ir": base.get("unlevered_net_ir"),
            "unlevered_max_dd": base.get("unlevered_max_dd"),
            "mean_cost_unlev_bp": base.get("mean_cost_unlev_bp"),
            "ic_gate_coverage": 1.0,
        },
        "fit_split": "train",
        "cover_min_train": IC_GATE_COVER_TRAIN,
    }


def decide_ic_gate_promote(
    *,
    val_always: dict[str, Any],
    val_gated: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only vs always-on q20. TEST never enters."""
    ir_on = _as_float(val_always.get("unlevered_net_ir"))
    ir_g = _as_float(val_gated.get("unlevered_net_ir"))
    dd_on = _as_float(val_always.get("unlevered_max_dd"))
    dd_g = _as_float(val_gated.get("unlevered_max_dd"))
    cover = _as_float(val_gated.get("ic_gate_coverage"))
    ir_delta = (
        float(ir_g - ir_on) if np.isfinite(ir_g) and np.isfinite(ir_on) else float("nan")
    )
    dd_delta = (
        float(dd_g - dd_on) if np.isfinite(dd_g) and np.isfinite(dd_on) else float("nan")
    )
    have_spec = bool(chosen.get("window"))
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    cover_ok = bool(np.isfinite(cover) and cover >= IC_GATE_COVER_VAL)
    promote = bool(have_spec and ir_ok and dd_ok and cover_ok)
    if not have_spec:
        reason = (
            "NO PROMOTE: TRAIN did not select a (W, τ) with coverage "
            f">= {IC_GATE_COVER_TRAIN:.0%}. Keep always-on q20."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE: TRAIN chose W={chosen.get('window')} τ={chosen.get('tau'):+.2f} "
            f"but VAL coverage {100 * cover:.0f}% < {100 * IC_GATE_COVER_VAL:.0f}% "
            "(catastrophic flatten). Keep always-on q20."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE: IC gate W={chosen.get('window')} τ={chosen.get('tau'):+.2f} "
            f"VAL unlev net IR {ir_g:+.3f} vs always-on {ir_on:+.3f} "
            f"(delta {ir_delta:+.3f} < {LO_IR_LIFT:.2f}). Keep always-on q20."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE: IC gate IR lift {ir_delta:+.3f} but VAL max DD "
            f"{dd_g:+.3f} vs always-on {dd_on:+.3f} exceeds {LO_DD_TOL:.2f}. "
            "Keep always-on q20."
        )
    else:
        reason = (
            f"PROMOTE IC gate W={chosen.get('window')} τ={chosen.get('tau'):+.2f}: "
            f"VAL IR {ir_g:+.3f} vs {ir_on:+.3f} (delta {ir_delta:+.3f}), "
            f"max DD {dd_g:+.3f} vs {dd_on:+.3f}, coverage {100 * cover:.0f}%."
        )
    return {
        "promote_ic_gate": promote,
        "gated_on": "val",
        "fit_split": "train",
        "reason": reason,
        "spec": {
            "window": chosen.get("window", 0),
            "tau": chosen.get("tau", 0.0),
        }
        if promote
        else {"window": 0, "tau": 0.0},
        "chosen": chosen,
        "ir_always": ir_on,
        "ir_gated": ir_g,
        "ir_delta": ir_delta,
        "dd_always": dd_on,
        "dd_gated": dd_g,
        "dd_delta": dd_delta,
        "coverage": cover,
        "cover_min_val": IC_GATE_COVER_VAL,
        "ir_lift": LO_IR_LIFT,
    }


WEEKDAY_MASKS = ("always", "flat_friday", "weekend_only", "flat_monday")
WEEKDAY_COVER_VAL = 0.30


def weekday_mask_grid(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Score causal weekday masks on one split. Caller decides VAL vs TEST."""
    empty = {"rows": [], "best": {}, "baseline": {}, "eligible": []}
    if df.empty:
        return empty
    pred, y, r_on, tz, vol = _wide_from_frame(df)
    if pred.empty or pred.shape[1] < 2:
        return empty
    rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] = {}
    for mask in WEEKDAY_MASKS:
        stats = _lo_q20_book(
            pred,
            y,
            min_names=min_names,
            vol_target=vol_target,
            overnight_r=r_on,
            turnover_z=tz,
            vol_level=vol,
            weekday_mask=mask,
        )
        cover = _as_float(stats.get("weekday_coverage"))
        if mask == "always" and not np.isfinite(cover):
            cover = 1.0
        row = {
            "name": f"wd_{mask}",
            "weekday_mask": mask,
            "unlevered_net_ir": stats.get("unlevered_net_ir"),
            "unlevered_max_dd": stats.get("unlevered_max_dd"),
            "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
            "weekday_coverage": cover,
            "weekday_n_flat": stats.get("weekday_n_flat"),
            "weekday_n_dates": stats.get("weekday_n_dates"),
            "n_dates": stats.get("n_dates"),
        }
        rows.append(row)
        if mask == "always":
            baseline = dict(row)
    eligible = [
        dict(r)
        for r in rows
        if np.isfinite(_as_float(r.get("unlevered_net_ir")))
        and np.isfinite(_as_float(r.get("weekday_coverage")))
        and _as_float(r.get("weekday_coverage")) >= WEEKDAY_COVER_VAL
    ]
    eligible.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            str(r.get("name")),
        )
    )
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            str(r.get("name")),
        )
    )
    return {
        "rows": rows,
        "best": dict(eligible[0]) if eligible else {},
        "baseline": baseline,
        "eligible": eligible,
        "cover_min": WEEKDAY_COVER_VAL,
        "note": (
            "weekday(t) at close t only; Friday=weekend gap; "
            "Monday=Mon close→Tue open; next open never a feature"
        ),
    }


def decide_weekday_promote(grid: dict[str, Any]) -> dict[str, Any]:
    """VAL-only vs always-on q20. TEST never enters."""
    baseline = dict(grid.get("baseline") or {})
    best = dict(grid.get("best") or {})
    ir_on = _as_float(baseline.get("unlevered_net_ir"))
    ir_g = _as_float(best.get("unlevered_net_ir"))
    dd_on = _as_float(baseline.get("unlevered_max_dd"))
    dd_g = _as_float(best.get("unlevered_max_dd"))
    cover = _as_float(best.get("weekday_coverage"))
    ir_delta = (
        float(ir_g - ir_on) if np.isfinite(ir_g) and np.isfinite(ir_on) else float("nan")
    )
    dd_delta = (
        float(dd_g - dd_on) if np.isfinite(dd_g) and np.isfinite(dd_on) else float("nan")
    )
    mask = str(best.get("weekday_mask") or "")
    same = mask in ("", "always") or str(best.get("name") or "") == str(
        baseline.get("name") or ""
    )
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    cover_ok = bool(np.isfinite(cover) and cover >= WEEKDAY_COVER_VAL)
    promote = bool((not same) and ir_ok and dd_ok and cover_ok and bool(mask))
    if not best:
        reason = (
            "NO PROMOTE: no weekday mask met VAL coverage "
            f">= {WEEKDAY_COVER_VAL:.0%}. Keep always-on q20."
        )
    elif same or not ir_ok:
        reason = (
            f"NO PROMOTE: no weekday mask beats always-on by {LO_IR_LIFT:.2f} "
            f"unlev net IR (best {best.get('name')} {ir_g:+.3f} vs always-on "
            f"{ir_on:+.3f}, delta {ir_delta:+.3f}). Keep always-on q20."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE: {best.get('name')} VAL coverage "
            f"{100 * cover:.0f}% < {100 * WEEKDAY_COVER_VAL:.0f}% "
            "(catastrophic flatten). Keep always-on q20."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE: weekday {mask} IR lift {ir_delta:+.3f} but VAL max DD "
            f"{dd_g:+.3f} vs always-on {dd_on:+.3f} exceeds {LO_DD_TOL:.2f}. "
            "Keep always-on q20."
        )
    else:
        reason = (
            f"PROMOTE weekday mask {mask}: VAL IR {ir_g:+.3f} vs {ir_on:+.3f} "
            f"(delta {ir_delta:+.3f}), max DD {dd_g:+.3f} vs {dd_on:+.3f}, "
            f"coverage {100 * cover:.0f}%."
        )
    return {
        "promote_weekday": promote,
        "gated_on": "val",
        "reason": reason,
        "spec": {"weekday_mask": mask} if promote else {"weekday_mask": "always"},
        "chosen": best,
        "baseline": baseline,
        "ir_always": ir_on,
        "ir_gated": ir_g,
        "ir_delta": ir_delta,
        "dd_always": dd_on,
        "dd_gated": dd_g,
        "dd_delta": dd_delta,
        "coverage": cover,
        "cover_min_val": WEEKDAY_COVER_VAL,
        "ir_lift": LO_IR_LIFT,
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
    lo_refine = val_long_only_refine(frames["val"], min_names=min_names, vol_target=vol_target)
    ic_fit = fit_ic_gate_on_train(
        frames["train"], min_names=min_names, vol_target=vol_target
    )
    ic_chosen = ic_fit.get("chosen") or {}
    ic_W = int(ic_chosen.get("window") or 0)
    ic_tau = float(ic_chosen.get("tau") or 0.0)

    def _gated_lo(split: str, hist: tuple[str, ...], window: int, tau: float) -> dict[str, Any]:
        if frames[split].empty:
            return {}
        pred, y, r_on, tz, vol = _wide_from_frame(frames[split])
        if pred.empty or pred.shape[1] < 2:
            return {}
        trail = None
        if window > 0:
            parts_p = [
                frame_to_wide(frames[s], "pred")
                for s in hist
                if s in frames and not frames[s].empty
            ]
            parts_y = [
                frame_to_wide(frames[s], "y")
                for s in hist
                if s in frames and not frames[s].empty
            ]
            if parts_p and parts_y:
                hp = pd.concat(parts_p).sort_index().groupby(level=0).last()
                hy = pd.concat(parts_y).sort_index().groupby(level=0).last()
                trail = trailing_mean_cs_ic(
                    hp, hy, window=window, min_names=min_names
                )
        return _lo_q20_book(
            pred,
            y,
            min_names=min_names,
            vol_target=vol_target,
            overnight_r=r_on,
            turnover_z=tz,
            vol_level=vol,
            ic_gate_window=window,
            ic_gate_tau=tau,
            ic_gate_trail=trail,
        )

    ic_val_always = _gated_lo("val", ("train", "val"), 0, 0.0)
    ic_val_gated = (
        _gated_lo("val", ("train", "val"), ic_W, ic_tau) if ic_W else dict(ic_val_always)
    )
    ic_test_always = _gated_lo("test", ("train", "val", "test"), 0, 0.0)
    ic_test_gated = (
        _gated_lo("test", ("train", "val", "test"), ic_W, ic_tau)
        if ic_W
        else dict(ic_test_always)
    )
    ic_promo = decide_ic_gate_promote(
        val_always=ic_val_always, val_gated=ic_val_gated, chosen=ic_chosen
    )
    wd_val = weekday_mask_grid(
        frames["val"], min_names=min_names, vol_target=vol_target
    )
    wd_promo = decide_weekday_promote(wd_val)
    wd_test = weekday_mask_grid(
        frames["test"], min_names=min_names, vol_target=vol_target
    )
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
    test_refine_base: dict[str, Any] = {}
    test_refine_best: dict[str, Any] = {}
    spec = lo_promo.get("spec") or {}
    if frames["test"].empty is False:
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
            if spec:
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
            test_refine_base = _run_overnight_book(
                pred_t,
                y_t,
                bundle=LIVE_LONG_ONLY_BUNDLE,
                long_only=True,
                min_names=min_names,
                vol_target=vol_target,
                overnight_r=r_t,
                turnover_z=tz_t,
                vol_level=vol_t,
                quantile=0.2,
                weighting="quantile",
                long_size="equal",
                conf_pctile=0.0,
            )
            best_ref = lo_refine.get("best") or {}
            test_refine_best = _run_overnight_book(
                pred_t,
                y_t,
                bundle=LIVE_LONG_ONLY_BUNDLE,
                long_only=True,
                min_names=min_names,
                vol_target=vol_target,
                overnight_r=r_t,
                turnover_z=tz_t,
                vol_level=vol_t,
                quantile=float(best_ref.get("quantile") or 0.2),
                weighting="quantile",
                long_size=str(best_ref.get("long_size") or "equal"),
                conf_pctile=float(best_ref.get("conf_pctile") or 0.0),
            )
    lo_refine_promo = decide_lo_refine(
        lo_refine, test_baseline=test_refine_base, test_best=test_refine_best
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
        "val_long_only_refine": lo_refine,
        "promotion": promo,
        "lo_promotion": lo_promo,
        "lo_refine_promotion": lo_refine_promo,
        "ic_gate_fit": ic_fit,
        "ic_gate_promotion": ic_promo,
        "ic_gate_val_always": ic_val_always,
        "ic_gate_val_gated": ic_val_gated,
        "ic_gate_test_always": ic_test_always,
        "ic_gate_test_gated": ic_test_gated,
        "weekday_val_grid": wd_val,
        "weekday_test_grid": wd_test,
        "weekday_promotion": wd_promo,
        "ls_experiment": ls_exp,
        "test_long_only_promoted": test_lo_promoted,
        "test_long_only_refine_best": test_refine_best,
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
        f"PROMOTE LONG-ONLY REFINE (q/size/conf, TEST veto)? "
        f"{'YES' if (payload.get('lo_refine_promotion') or {}).get('promote_lo') else 'NO'}",
        f"  spec = {(payload.get('lo_refine_promotion') or {}).get('spec')}",
        f"  {(payload.get('lo_refine_promotion') or {}).get('reason')}",
        "",
        _ic_gate_block(payload),
        "",
        _weekday_block(payload),
        "",
        _lo_refine_block(
            payload.get("val_long_only_refine") or {},
            payload.get("lo_refine_promotion") or {},
        ),
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


def _ic_gate_block(payload: dict[str, Any]) -> str:
    promo = payload.get("ic_gate_promotion") or {}
    fit = payload.get("ic_gate_fit") or {}
    chosen = fit.get("chosen") or promo.get("chosen") or {}
    va = payload.get("ic_gate_val_always") or {}
    vg = payload.get("ic_gate_val_gated") or {}
    ta = payload.get("ic_gate_test_always") or {}
    tg = payload.get("ic_gate_test_gated") or {}

    def _cov(value: Any) -> str:
        x = _as_float(value)
        return "nan%" if not np.isfinite(x) else f"{100.0 * x:.0f}%"

    lines = [
        f"PROMOTE IC-GATE? {'YES' if promo.get('promote_ic_gate') else 'NO'}",
        f"  TRAIN chose W={chosen.get('window')} τ={chosen.get('tau')}  "
        f"cover {_cov(chosen.get('ic_gate_coverage'))}  "
        f"(fit_split={fit.get('fit_split')})",
        f"  {promo.get('reason')}",
        f"  VAL always-on  IR {_fmt(va.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(va.get('unlevered_max_dd'), '+.3f')}  "
        f"cost {_fmt(va.get('mean_cost_unlev_bp'), '.1f')} bp",
        f"  VAL gated      IR {_fmt(vg.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(vg.get('unlevered_max_dd'), '+.3f')}  "
        f"cover {_cov(vg.get('ic_gate_coverage'))}  "
        f"flat {int(_as_float(vg.get('ic_gate_n_flat'), 0.0))}/"
        f"{int(_as_float(vg.get('ic_gate_n_dates'), 0.0))}",
        f"  TEST always-on IR {_fmt(ta.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(ta.get('unlevered_max_dd'), '+.3f')}  (report-only)",
        f"  TEST gated     IR {_fmt(tg.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(tg.get('unlevered_max_dd'), '+.3f')}  "
        f"cover {_cov(tg.get('ic_gate_coverage'))}  (report-only)",
    ]
    return "\n".join(lines)


def _weekday_block(payload: dict[str, Any]) -> str:
    promo = payload.get("weekday_promotion") or {}
    val = payload.get("weekday_val_grid") or {}
    test = payload.get("weekday_test_grid") or {}
    chosen = promo.get("chosen") or {}
    base = val.get("baseline") or promo.get("baseline") or {}

    def _cov(value: Any) -> str:
        x = _as_float(value)
        return "nan%" if not np.isfinite(x) else f"{100.0 * x:.0f}%"

    def _row(r: dict[str, Any]) -> str:
        return (
            f"  {str(r.get('name') or ''):22} "
            f"IR {_fmt(r.get('unlevered_net_ir'), '+.3f')}  "
            f"maxDD {_fmt(r.get('unlevered_max_dd'), '+.3f')}  "
            f"cover {_cov(r.get('weekday_coverage'))}  "
            f"flat {int(_as_float(r.get('weekday_n_flat'), 0.0))}/"
            f"{int(_as_float(r.get('weekday_n_dates'), 0.0))}"
        )

    lines = [
        f"PROMOTE WEEKDAY MASK? {'YES' if promo.get('promote_weekday') else 'NO'}",
        f"  {val.get('note')}",
        f"  {promo.get('reason')}",
        f"  VAL always-on  IR {_fmt(base.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(base.get('unlevered_max_dd'), '+.3f')}  "
        f"cover {_cov(base.get('weekday_coverage'))}",
        f"  VAL best={chosen.get('name')}  "
        f"IR {_fmt(chosen.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(chosen.get('unlevered_max_dd'), '+.3f')}  "
        f"cover {_cov(chosen.get('weekday_coverage'))}",
        "  VAL weekday masks (gate):",
    ]
    for row in list(val.get("rows") or []):
        lines.append(_row(row))
    lines.append("  TEST weekday masks (report-only):")
    for row in list(test.get("rows") or []):
        lines.append(_row(row))
    return "\n".join(lines)


def _lo_refine_block(grid: dict[str, Any], promo: dict[str, Any]) -> str:
    rows = list(grid.get("rows") or [])
    best = grid.get("best") or {}
    base = grid.get("baseline") or {}
    lines = [
        "VAL LONG-ONLY REFINE (q10/15/20/30 × equal/abs_pred/inv_vol × conf; hold_hl=0)",
        f"  {grid.get('note')}",
        f"  baseline = {base.get('name')}  "
        f"IR {_fmt(base.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(base.get('unlevered_max_dd'), '+.3f')}",
        f"  VAL-best = {best.get('name')}  "
        f"IR {_fmt(best.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(best.get('unlevered_max_dd'), '+.3f')}",
        f"  TEST veto  best {_fmt(promo.get('test_best_ir'), '+.3f')} vs "
        f"q20 {_fmt(promo.get('test_baseline_ir'), '+.3f')}  "
        f"delta {_fmt(promo.get('test_ir_delta'), '+.3f')}  "
        f"collapse={promo.get('test_veto')}",
        "  name                           IR      maxDD   cost bp",
    ]
    for row in rows[:8]:
        lines.append(
            f"  {str(row.get('name')):<28}  "
            f"{_fmt(row.get('unlevered_net_ir'), '+.3f'):>7}  "
            f"{_fmt(row.get('unlevered_max_dd'), '+.3f'):>7}  "
            f"{_fmt(row.get('mean_cost_unlev_bp'), '.1f'):>7}"
        )
    if len(rows) > 8:
        lines.append(f"  ... {len(rows) - 8} more rows in JSON")
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
