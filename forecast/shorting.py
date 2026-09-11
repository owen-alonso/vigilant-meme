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
    BOOK_ALIGN_COVER,
    DIR_LIFT,
    PROMOTED_SKIP,
    _frame_for_split,
    cs_stack_mask,
    cs_top_abs_mask,
    fit_book_aligned_on_train,
    fit_promoted_overnight_skip,
    load_split_px,
    overnight_skip_data_config,
    score_eval_frame,
)
from forecast.backtest import (
    book_pnl,
    causal_disp_series,
    last_sticky_held,
    trailing_mean_cs_ic,
)
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
# IDEA 3: SPY-only overnight residual (A) vs default sector-overnight residual (B)
python -m forecast.training --universe liquid --interval daily --skip-only \\
  --label-return overnight --no-sector-residual \\
  --checkpoint-dir checkpoints/forecast_ridge_overnight_spy
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight_spy/best.pt \\
  --holding overnight --live-costs --long-only
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only
# causal CS-dispersion stress gate (TRAIN-fit kind/W/τ; default off until VAL promote)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --disp-gate-kind cc --disp-gate-window 1 --disp-gate-tau 0.02
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --disp-gate-kind on_trail --disp-gate-window 20 --disp-gate-tau 1.0
# IDEA 5: overnight ⊕ close-to-close rank ensemble (TRAIN-chosen α; default α=1)
# A = overnight sector residual (current). B = close-to-close residual skip.
# α grid {0.5, 0.6, 0.7, 0.8, 1.0}; α=1 is overnight-only. VAL gate, TEST report-only.
python -m forecast.training --universe liquid --interval daily --skip-only \\
  --label-return close --checkpoint-dir checkpoints/forecast_ridge
python scripts/overnight_shorting.py --data-dir data --universe liquid \\
    --json checkpoints/forecast_ridge_overnight/shorting.json
# IDEA 8: causal adaptive α_t from trailing CS IC of A vs B (TRAIN W/rule).
# Compare vs fixed α=0.70 and α=1. VAL gate, TEST report-only. Default off.
python scripts/overnight_shorting.py --data-dir data --universe liquid \\
    --json checkpoints/forecast_ridge_overnight/shorting.json
# IDEA 6: sticky long-only enter/exit (TRAIN-chosen; default off / always-rebuild q20)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --sticky-q-enter 0.15 --sticky-q-exit 0.40
# IDEA 7: soft trailing CS-IC gross scale (TRAIN-chosen; default off)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --ic-scale-window 20 --ic-scale-tau 0.04 --ic-scale-smax 1.25
# IDEA F: optional thin high-conviction sleeve (off unless VAL IR gate; default stays q20)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only --conviction-q 0.90 --conf-abs 0.335
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
    conf_abs: float = 0.0,
    conviction_q: float = 0.0,
    stack_long_half: bool = False,
    ic_gate_window: int = 0,
    ic_gate_tau: float = 0.0,
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
    sticky_held0: set | frozenset | None = None,
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
        conf_abs=float(conf_abs or 0.0),
        conviction_q=float(conviction_q or 0.0),
        stack_long_half=bool(stack_long_half),
        ic_gate_window=int(ic_gate_window or 0),
        ic_gate_tau=float(ic_gate_tau or 0.0),
        ic_gate_trail=ic_gate_trail,
        ic_scale_window=int(ic_scale_window or 0),
        ic_scale_tau=float(ic_scale_tau or 0.0),
        ic_scale_trail=ic_scale_trail,
        ic_scale_smax=float(ic_scale_smax or 1.0),
        weekday_mask=str(weekday_mask or "always"),
        disp_gate_trail=disp_gate_trail,
        disp_gate_tau=float(disp_gate_tau),
        disp_gate_kind=str(disp_gate_kind or ""),
        disp_gate_window=int(disp_gate_window or 0),
        close_px=close_px,
        overnight_r=overnight_r,
        turnover_z=turnover_z,
        vol_level=vol_level,
        sticky_q_enter=float(sticky_q_enter or 0.0),
        sticky_q_exit=float(sticky_q_exit or 0.0),
        sticky_held0=sticky_held0,
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


def _sleeve_coverage(df: pd.DataFrame, *, q: float, abs_tau: float, min_names: int) -> dict[str, float]:
    mask = cs_top_abs_mask(
        df, q=float(q), abs_tau=float(abs_tau), score_col="pred", min_names=min_names
    )
    cover = float(mask.mean()) if mask.size else float("nan")
    n = float(int(mask.sum()))
    n_dates = 0.0
    if int(mask.sum()) and "date" in df.columns:
        n_dates = float(pd.Series(df["date"].to_numpy()[mask]).nunique())
    n_all = float(df["date"].nunique()) if not df.empty and "date" in df.columns else 0.0
    return {
        "coverage": cover,
        "n": n,
        "n_dates": n_dates,
        "date_coverage": (float(n_dates / n_all) if n_all else float("nan")),
    }


def score_conviction_live_book(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float,
    conviction_q: float = 0.0,
    conf_abs: float = 0.0,
    quantile: float = 0.2,
    stack_long_half: bool = False,
    name: str = "",
) -> dict[str, Any]:
    """``--live-costs --long-only`` unlev net IR / DD / turnover on one split."""
    empty = {
        "name": name or "empty",
        "unlevered_net_ir": float("nan"),
        "unlevered_max_dd": float("nan"),
        "mean_turnover": float("nan"),
        "mean_cost_unlev_bp": float("nan"),
        "n_book_dates": float("nan"),
        "coverage": float("nan"),
        "date_coverage": float("nan"),
        "n": 0.0,
        "n_dates": 0.0,
        "conviction_q": float(conviction_q),
        "conf_abs": float(conf_abs),
        "quantile": float(quantile),
    }
    if df.empty:
        return empty
    pred, y, r_on, tz, vol = _wide_from_frame(df)
    if pred.empty or pred.shape[1] < 2:
        return empty
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
        quantile=float(quantile),
        weighting="quantile",
        long_size="equal",
        conviction_q=float(conviction_q or 0.0),
        conf_abs=float(conf_abs or 0.0),
        stack_long_half=bool(stack_long_half),
    )
    acc_q = float(conviction_q) if float(conviction_q or 0.0) > 0 else 0.80
    if stack_long_half:
        mask = cs_stack_mask(
            df,
            q=acc_q,
            abs_tau=float(conf_abs or 0.0),
            score_col="pred",
            min_names=min_names,
        )
        cover = float(mask.mean()) if mask.size else float("nan")
        n = float(int(mask.sum()))
        n_dates = (
            float(pd.Series(df["date"].to_numpy()[mask]).nunique())
            if int(mask.sum()) and "date" in df.columns
            else 0.0
        )
        n_all = (
            float(df["date"].nunique())
            if not df.empty and "date" in df.columns
            else 0.0
        )
        cov = {
            "coverage": cover,
            "n": n,
            "n_dates": n_dates,
            "date_coverage": (float(n_dates / n_all) if n_all else float("nan")),
        }
    else:
        cov = _sleeve_coverage(
            df, q=acc_q, abs_tau=float(conf_abs or 0.0), min_names=min_names
        )
    return {
        "name": name or ("conviction" if conviction_q else "live_long_only_q20"),
        "unlevered_net_ir": stats.get("unlevered_net_ir"),
        "unlevered_max_dd": stats.get("unlevered_max_dd"),
        "mean_turnover": stats.get("mean_turnover"),
        "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
        "n_book_dates": stats.get("n_dates"),
        "coverage": cov["coverage"],
        "date_coverage": cov["date_coverage"],
        "n": cov["n"],
        "n_dates": cov["n_dates"],
        "conviction_q": float(conviction_q or 0.0),
        "conf_abs": float(conf_abs or 0.0),
        "quantile": float(quantile),
        "stack_long_half": bool(stack_long_half),
        "long_only": True,
        "cost_bundle": "live_long_only",
    }


def compare_conviction_live(
    frames: dict[str, pd.DataFrame],
    *,
    chosen: dict[str, Any],
    min_names: int,
    vol_target: float,
) -> dict[str, Any]:
    """Score q20 vs E's TRAIN sleeve vs optional q=0.90 (no |pred| floor)."""
    q = float((chosen or {}).get("q") or 0.80)
    abs_tau = float((chosen or {}).get("abs_tau") or 0.0)
    abs_q = float((chosen or {}).get("abs_q") or 0.0)
    out: dict[str, Any] = {
        "train_q": q,
        "train_abs_tau": abs_tau,
        "train_abs_q": abs_q,
        "fit_split": "train",
        "note": (
            "live_long_only unlev net IR / max DD / turnover. "
            "q20 equal is the default book. Chosen sleeve uses IDEA E's "
            "TRAIN (q, |pred| floor). TEST is report-only."
        ),
    }
    for split, df in frames.items():
        q20 = score_conviction_live_book(
            df,
            min_names=min_names,
            vol_target=vol_target,
            quantile=0.2,
            name="q20_equal",
        )
        chosen_row = score_conviction_live_book(
            df,
            min_names=min_names,
            vol_target=vol_target,
            conviction_q=q,
            conf_abs=abs_tau,
            name=f"e_q{q:.2f}_abs{abs_q:.2f}",
        )
        q90 = score_conviction_live_book(
            df,
            min_names=min_names,
            vol_target=vol_target,
            conviction_q=0.90,
            conf_abs=0.0,
            name="q90_no_abs",
        )
        out[split] = {"q20": q20, "chosen": chosen_row, "q90": q90}
    return out


def decide_conviction_live_promote(
    *,
    val_q20: dict[str, Any],
    val_chosen: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only optional live path vs q20. TEST never enters."""
    ir20 = _as_float((val_q20 or {}).get("unlevered_net_ir"))
    ir = _as_float((val_chosen or {}).get("unlevered_net_ir"))
    dd20 = _as_float((val_q20 or {}).get("unlevered_max_dd"))
    dd = _as_float((val_chosen or {}).get("unlevered_max_dd"))
    cover = _as_float((val_chosen or {}).get("coverage"))
    to20 = _as_float((val_q20 or {}).get("mean_turnover"))
    to = _as_float((val_chosen or {}).get("mean_turnover"))
    q = _as_float((chosen or {}).get("q"), default=0.80)
    abs_tau = _as_float((chosen or {}).get("abs_tau"), default=0.0)
    abs_q = _as_float((chosen or {}).get("abs_q"), default=0.0)
    same = abs(q - 0.80) < 1e-12 and abs(abs_tau) <= 1e-15
    ir_delta = (
        float(ir - ir20) if np.isfinite(ir) and np.isfinite(ir20) else float("nan")
    )
    dd_delta = (
        float(dd - dd20) if np.isfinite(dd) and np.isfinite(dd20) else float("nan")
    )
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    cover_ok = bool(np.isfinite(cover) and cover >= BOOK_ALIGN_COVER)
    promote = bool((not same) and ir_ok and dd_ok and cover_ok)
    spec = {"quantile": 0.2, "conviction_q": 0.0, "conf_abs": 0.0}
    if same:
        reason = (
            "NO PROMOTE optional live path: TRAIN sleeve is q20 / no |pred| floor. "
            f"Keep q20 default. IDEA E remains hit-rate-only. VAL IR {ir:+.3f} vs "
            f"q20 {ir20:+.3f}."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE optional live path: VAL cover {100.0 * cover:.1f}% "
            f"< {100.0 * BOOK_ALIGN_COVER:.0f}%. Keep q20 default. "
            "IDEA E remains hit-rate-only."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE optional live path: VAL unlev net IR {ir:+.3f} vs q20 "
            f"{ir20:+.3f} (delta {ir_delta:+.3f} < +{LO_IR_LIFT:.2f}). "
            "Keep q20 default. IDEA E remains hit-rate-only."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE optional live path: VAL max DD {dd:+.3f} vs q20 "
            f"{dd20:+.3f} (delta {dd_delta:+.3f} < -{LO_DD_TOL:.2f}). "
            "Keep q20 default. IDEA E remains hit-rate-only."
        )
    else:
        spec = {"quantile": 0.2, "conviction_q": float(q), "conf_abs": float(abs_tau)}
        reason = (
            f"PROMOTE optional live path q={q:.2f} abs_q={abs_q:.2f}: VAL unlev "
            f"net IR {ir:+.3f} vs q20 {ir20:+.3f} (delta {ir_delta:+.3f}) and "
            f"max DD {dd:+.3f} vs {dd20:+.3f} (delta {dd_delta:+.3f}), "
            f"cover {100.0 * cover:.1f}%. Default CLI stays q20 until liquid."
        )
    return {
        "promote_conviction_live": promote,
        "gated_on": "val",
        "reason": reason,
        "spec": spec,
        "q20": dict(val_q20 or {}),
        "chosen": dict(val_chosen or {}),
        "train_q": float(q),
        "train_abs_q": float(abs_q),
        "train_abs_tau": float(abs_tau),
        "val_ir": ir,
        "val_ir_q20": ir20,
        "val_ir_delta": ir_delta,
        "val_dd": dd,
        "val_dd_q20": dd20,
        "val_dd_delta": dd_delta,
        "val_turnover": to,
        "val_turnover_q20": to20,
        "coverage": cover,
        "ir_lift": LO_IR_LIFT,
        "dd_tol": LO_DD_TOL,
        "cover_floor": BOOK_ALIGN_COVER,
        "default_book_unchanged": True,
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
    sticky_held0: set | frozenset | None = None,
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
        ic_scale_window=int(ic_scale_window or 0),
        ic_scale_tau=float(ic_scale_tau or 0.0),
        ic_scale_trail=ic_scale_trail,
        ic_scale_smax=float(ic_scale_smax or 1.0),
        weekday_mask=str(weekday_mask or "always"),
        disp_gate_trail=disp_gate_trail,
        disp_gate_tau=float(disp_gate_tau),
        disp_gate_kind=str(disp_gate_kind or ""),
        disp_gate_window=int(disp_gate_window or 0),
        close_px=close_px,
        sticky_q_enter=float(sticky_q_enter or 0.0),
        sticky_q_exit=float(sticky_q_exit or 0.0),
        sticky_held0=sticky_held0,
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


IC_SCALE_WINDOWS = (20, 60, 120)
IC_SCALE_TAUS = (0.02, 0.04, 0.06, 0.08)
IC_SCALE_SMAX = (1.0, 1.25)


def _train_ic_scale_taus(
    pred: pd.DataFrame,
    y: pd.DataFrame,
    *,
    window: int,
    min_names: int,
) -> list[float]:
    """Fixed taus plus TRAIN trail percentiles so s_t can actually vary."""
    taus = [float(t) for t in IC_SCALE_TAUS]
    trail = trailing_mean_cs_ic(pred, y, window=int(window), min_names=int(min_names))
    finite = trail.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size >= 8:
        for q in (0.25, 0.50, 0.75):
            t = float(np.quantile(finite, q))
            if t > 1e-4:
                taus.append(t)
    out: list[float] = []
    seen: set[float] = set()
    for t in taus:
        key = round(float(t), 4)
        if key > 0 and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _ic_scale_row(stats: dict[str, Any], *, window: int, tau: float, s_max: float) -> dict[str, Any]:
    return {
        "name": f"ic_scale_W{window}_t{tau:.3f}_x{s_max:.2f}",
        "window": window,
        "tau": tau,
        "s_max": s_max,
        "unlevered_net_ir": stats.get("unlevered_net_ir"),
        "unlevered_max_dd": stats.get("unlevered_max_dd"),
        "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
        "mean_turnover": stats.get("mean_turnover"),
        "mean_name_churn": stats.get("mean_name_churn"),
        "mean_ic_scale": stats.get("mean_ic_scale"),
        "ic_scale_n_partial": stats.get("ic_scale_n_partial"),
        "ic_scale_n_boost": stats.get("ic_scale_n_boost"),
        "ic_scale_n_flat": stats.get("ic_scale_n_flat"),
        "coverage": stats.get("ic_scale_coverage", stats.get("sticky_coverage")),
        "n_dates": stats.get("n_dates"),
    }


def fit_ic_scale_on_train(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Select soft-scale (W, τ, s_max) on TRAIN only. VAL/TEST must never enter."""
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
    for window in IC_SCALE_WINDOWS:
        taus = _train_ic_scale_taus(pred, y, window=window, min_names=min_names)
        for tau in taus:
            for s_max in IC_SCALE_SMAX:
                stats = _lo_q20_book(
                    pred, y, min_names=min_names, vol_target=vol_target,
                    overnight_r=r_on, turnover_z=tz, vol_level=vol,
                    ic_scale_window=window, ic_scale_tau=tau, ic_scale_smax=s_max,
                )
                ir = _as_float(stats.get("unlevered_net_ir"))
                row = _ic_scale_row(stats, window=window, tau=tau, s_max=s_max)
                rows.append(row)
                if not np.isfinite(ir):
                    continue
                if ir > best_ir + 1e-12:
                    best_ir = ir
                    chosen = dict(row)
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            str(r.get("name")),
        )
    )
    return {
        "rows": rows,
        "chosen": chosen,
        "baseline": {
            "name": "live_long_only_q20",
            "unlevered_net_ir": base.get("unlevered_net_ir"),
            "unlevered_max_dd": base.get("unlevered_max_dd"),
            "mean_cost_unlev_bp": base.get("mean_cost_unlev_bp"),
            "mean_turnover": base.get("mean_turnover"),
            "mean_name_churn": base.get("mean_name_churn"),
            "mean_ic_scale": 1.0,
            "coverage": 1.0,
        },
        "fit_split": "train",
        "note": (
            "s_t = clip(trail_IC_{t-}/τ, 0, s_max); s_max in {1.00, 1.25}; "
            "trade every night; NaN warmup = full gross. Causal dates < t only. "
            "τ grid = fixed + TRAIN trail p25/p50/p75."
        ),
    }


def decide_ic_scale_promote(
    *,
    val_always: dict[str, Any],
    val_scaled: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only vs ungated q20. TEST never enters."""
    ir_on = _as_float(val_always.get("unlevered_net_ir"))
    ir_g = _as_float(val_scaled.get("unlevered_net_ir"))
    dd_on = _as_float(val_always.get("unlevered_max_dd"))
    dd_g = _as_float(val_scaled.get("unlevered_max_dd"))
    ir_delta = (
        float(ir_g - ir_on) if np.isfinite(ir_g) and np.isfinite(ir_on) else float("nan")
    )
    dd_delta = (
        float(dd_g - dd_on) if np.isfinite(dd_g) and np.isfinite(dd_on) else float("nan")
    )
    have_spec = bool(chosen.get("window"))
    s_max = _as_float(chosen.get("s_max"), default=1.0)
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    promote = bool(have_spec and ir_ok and dd_ok)
    if not have_spec:
        reason = (
            "NO PROMOTE: TRAIN did not select a soft-scale (W, τ, s_max). "
            "Keep ungated q20."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE: IC scale W={chosen.get('window')} τ={chosen.get('tau'):.3f} "
            f"s_max={s_max:.2f} VAL unlev net IR {ir_g:+.3f} vs ungated {ir_on:+.3f} "
            f"(delta {ir_delta:+.3f} < {LO_IR_LIFT:.2f}). Keep ungated q20."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE: IC scale IR lift {ir_delta:+.3f} but VAL max DD "
            f"{dd_g:+.3f} vs ungated {dd_on:+.3f} exceeds {LO_DD_TOL:.2f}. "
            "Keep ungated q20."
        )
    else:
        reason = (
            f"PROMOTE IC scale W={chosen.get('window')} τ={chosen.get('tau'):.3f} "
            f"s_max={s_max:.2f}: VAL IR {ir_g:+.3f} vs {ir_on:+.3f} "
            f"(delta {ir_delta:+.3f}), max DD {dd_g:+.3f} vs {dd_on:+.3f}, "
            f"mean_s {_as_float(val_scaled.get('mean_ic_scale')):.2f}."
        )
    return {
        "promote_ic_scale": promote,
        "gated_on": "val",
        "fit_split": "train",
        "reason": reason,
        "spec": (
            {
                "window": chosen.get("window", 0),
                "tau": chosen.get("tau", 0.0),
                "s_max": float(s_max) if np.isfinite(s_max) else 1.0,
            }
            if promote
            else {"window": 0, "tau": 0.0, "s_max": 1.0}
        ),
        "chosen": chosen,
        "ir_always": ir_on,
        "ir_scaled": ir_g,
        "ir_delta": ir_delta,
        "dd_always": dd_on,
        "dd_scaled": dd_g,
        "dd_delta": dd_delta,
        "mean_ic_scale": _as_float(val_scaled.get("mean_ic_scale")),
        "name_churn_always": _as_float(val_always.get("mean_name_churn")),
        "name_churn_scaled": _as_float(val_scaled.get("mean_name_churn")),
        "coverage_scaled": _as_float(val_scaled.get("ic_scale_coverage", val_scaled.get("coverage"))),
        "ir_lift": LO_IR_LIFT,
    }


DISP_SPECS = (("cc", 1), ("cc", 20), ("on_trail", 20))
DISP_QS = (0.70, 0.80, 0.90)
DISP_COVER_TRAIN = 0.40
DISP_COVER_VAL = 0.30


def _close_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    return frame_to_wide(df, "close")


def _hist_disp_series(
    frames: dict[str, pd.DataFrame],
    hist: tuple[str, ...],
    kind: str,
    window: int,
    min_names: int,
) -> pd.Series:
    parts_c = [
        _close_wide(frames[s])
        for s in hist
        if s in frames and not frames[s].empty
    ]
    parts_y = [
        frame_to_wide(frames[s], "y")
        for s in hist
        if s in frames and not frames[s].empty
    ]
    close = (
        pd.concat(parts_c).sort_index().groupby(level=0).last() if parts_c else None
    )
    resid = (
        pd.concat(parts_y).sort_index().groupby(level=0).last() if parts_y else None
    )
    return causal_disp_series(
        kind, close=close, resid=resid, window=int(window), min_names=int(min_names)
    )


def fit_disp_gate_on_train(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Select (kind, W, τ) on TRAIN only. VAL/TEST must never enter."""
    empty = {"rows": [], "chosen": {}, "baseline": {}, "fit_split": "train"}
    if df.empty:
        return empty
    pred, y, r_on, tz, vol = _wide_from_frame(df)
    if pred.empty or pred.shape[1] < 2:
        return empty
    close = _close_wide(df)
    base = _lo_q20_book(
        pred, y, min_names=min_names, vol_target=vol_target,
        overnight_r=r_on, turnover_z=tz, vol_level=vol,
    )
    rows: list[dict[str, Any]] = []
    chosen: dict[str, Any] = {}
    best_ir = -1e18
    for kind, window in DISP_SPECS:
        series = causal_disp_series(
            kind, close=close, resid=y, window=window, min_names=min_names
        )
        finite = series.to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size < 8:
            continue
        for q in DISP_QS:
            tau = float(np.quantile(finite, q))
            stats = _lo_q20_book(
                pred, y, min_names=min_names, vol_target=vol_target,
                overnight_r=r_on, turnover_z=tz, vol_level=vol,
                disp_gate_trail=series, disp_gate_tau=tau,
                disp_gate_kind=kind, disp_gate_window=window,
            )
            cover = _as_float(stats.get("disp_gate_coverage"))
            ir = _as_float(stats.get("unlevered_net_ir"))
            row = {
                "name": f"disp_{kind}_W{window}_q{int(100 * q)}",
                "kind": kind,
                "window": window,
                "q": q,
                "tau": tau,
                "unlevered_net_ir": stats.get("unlevered_net_ir"),
                "unlevered_max_dd": stats.get("unlevered_max_dd"),
                "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
                "disp_gate_coverage": cover,
                "disp_gate_n_flat": stats.get("disp_gate_n_flat"),
            }
            rows.append(row)
            if (not np.isfinite(ir)) or (not np.isfinite(cover)):
                continue
            if cover < DISP_COVER_TRAIN:
                continue
            if ir > best_ir + 1e-12:
                best_ir = ir
                chosen = dict(row)
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            str(r.get("name")),
        )
    )
    return {
        "rows": rows,
        "chosen": chosen,
        "baseline": {
            "name": "live_long_only_q20",
            "unlevered_net_ir": base.get("unlevered_net_ir"),
            "unlevered_max_dd": base.get("unlevered_max_dd"),
            "mean_cost_unlev_bp": base.get("mean_cost_unlev_bp"),
            "disp_gate_coverage": 1.0,
        },
        "fit_split": "train",
        "cover_min_train": DISP_COVER_TRAIN,
        "note": (
            "cc = same-day close-to-close CS std (dates ≤ t); "
            "on_trail = overnight residual CS std on dates < t only"
        ),
    }


def decide_disp_gate_promote(
    *,
    val_always: dict[str, Any],
    val_gated: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only vs ungated q20. TEST never enters."""
    ir_on = _as_float(val_always.get("unlevered_net_ir"))
    ir_g = _as_float(val_gated.get("unlevered_net_ir"))
    dd_on = _as_float(val_always.get("unlevered_max_dd"))
    dd_g = _as_float(val_gated.get("unlevered_max_dd"))
    cover = _as_float(val_gated.get("disp_gate_coverage"))
    ir_delta = (
        float(ir_g - ir_on) if np.isfinite(ir_g) and np.isfinite(ir_on) else float("nan")
    )
    dd_delta = (
        float(dd_g - dd_on) if np.isfinite(dd_g) and np.isfinite(dd_on) else float("nan")
    )
    have_spec = bool(chosen.get("kind")) and np.isfinite(_as_float(chosen.get("tau")))
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    cover_ok = bool(np.isfinite(cover) and cover >= DISP_COVER_VAL)
    promote = bool(have_spec and ir_ok and dd_ok and cover_ok)
    kind = chosen.get("kind")
    window = chosen.get("window")
    tau = chosen.get("tau")
    if not have_spec:
        reason = (
            "NO PROMOTE: TRAIN did not select a dispersion (kind, W, τ) with "
            f"coverage >= {DISP_COVER_TRAIN:.0%}. Keep ungated q20."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE: TRAIN chose {kind} W={window} τ={tau} "
            f"but VAL coverage {100 * cover:.0f}% < {100 * DISP_COVER_VAL:.0f}% "
            "(catastrophic flatten). Keep ungated q20."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE: disp gate {kind} W={window} τ={_fmt(tau, '.4f')} "
            f"VAL unlev net IR {ir_g:+.3f} vs ungated {ir_on:+.3f} "
            f"(delta {ir_delta:+.3f} < {LO_IR_LIFT:.2f}). Keep ungated q20."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE: disp gate IR lift {ir_delta:+.3f} but VAL max DD "
            f"{dd_g:+.3f} vs ungated {dd_on:+.3f} exceeds {LO_DD_TOL:.2f}. "
            "Keep ungated q20."
        )
    else:
        reason = (
            f"PROMOTE disp gate {kind} W={window} τ={_fmt(tau, '.4f')}: "
            f"VAL IR {ir_g:+.3f} vs {ir_on:+.3f} (delta {ir_delta:+.3f}), "
            f"max DD {dd_g:+.3f} vs {dd_on:+.3f}, coverage {100 * cover:.0f}%."
        )
    return {
        "promote_disp_gate": promote,
        "gated_on": "val",
        "fit_split": "train",
        "reason": reason,
        "spec": {
            "kind": kind,
            "window": window,
            "tau": tau,
            "q": chosen.get("q"),
        }
        if promote
        else {"kind": "", "window": 0, "tau": 0.0, "q": 0.0},
        "chosen": chosen,
        "ir_always": ir_on,
        "ir_gated": ir_g,
        "ir_delta": ir_delta,
        "dd_always": dd_on,
        "dd_gated": dd_g,
        "dd_delta": dd_delta,
        "coverage": cover,
        "cover_min_val": DISP_COVER_VAL,
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


def decide_sector_promote(
    *,
    val_spy: dict[str, Any],
    val_sector: dict[str, Any],
    n_sector_hedges: int = 0,
    n_names: int = 0,
) -> dict[str, Any]:
    """VAL-only: promote sector-overnight residual vs SPY/market baseline."""
    ir_a = _as_float(val_spy.get("unlevered_net_ir"))
    ir_b = _as_float(val_sector.get("unlevered_net_ir"))
    dd_a = _as_float(val_spy.get("unlevered_max_dd"))
    dd_b = _as_float(val_sector.get("unlevered_max_dd"))
    ir_delta = (
        float(ir_b - ir_a) if np.isfinite(ir_b) and np.isfinite(ir_a) else float("nan")
    )
    dd_delta = (
        float(dd_b - dd_a) if np.isfinite(dd_b) and np.isfinite(dd_a) else float("nan")
    )
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    used_sector = int(n_sector_hedges) > 0
    promote = bool(ir_ok and dd_ok and used_sector)
    if not used_sector:
        reason = (
            "NO PROMOTE: sector ETFs missing (all names fell back to SPY). "
            "A and B are the same overnight residual. Keep current skip "
            "(sector_residual=True with SPY fallback)."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE: sector-overnight residual VAL unlev net IR {ir_b:+.3f} "
            f"vs SPY/market {ir_a:+.3f} (delta {ir_delta:+.3f} < {LO_IR_LIFT:.2f}). "
            "Do not prefer sector hedge on this tape."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE: sector IR lift {ir_delta:+.3f} but VAL max DD "
            f"{dd_b:+.3f} vs SPY {dd_a:+.3f} exceeds {LO_DD_TOL:.2f}."
        )
    else:
        reason = (
            f"PROMOTE sector-overnight residual: VAL IR {ir_b:+.3f} vs SPY "
            f"{ir_a:+.3f} (delta {ir_delta:+.3f}), max DD {dd_b:+.3f} vs "
            f"{dd_a:+.3f}. {n_sector_hedges}/{n_names} names used a sector ETF."
        )
    return {
        "promote_sector": promote,
        "gated_on": "val",
        "reason": reason,
        "spec": {"sector_residual": True} if promote else {"sector_residual": True},
        "ir_spy": ir_a,
        "ir_sector": ir_b,
        "ir_delta": ir_delta,
        "dd_spy": dd_a,
        "dd_sector": dd_b,
        "dd_delta": dd_delta,
        "n_sector_hedges": int(n_sector_hedges),
        "n_names": int(n_names),
        "ir_lift": LO_IR_LIFT,
        "fallback": "SPY when mapped sector ETF parquet is missing",
    }


def _split_lo_q20(
    df: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float,
) -> dict[str, Any]:
    if df.empty:
        return {}
    pred, y, r_on, tz, vol = _wide_from_frame(df)
    if pred.empty or pred.shape[1] < 2:
        return {}
    return _lo_q20_book(
        pred,
        y,
        min_names=min_names,
        vol_target=vol_target,
        overnight_r=r_on,
        turnover_z=tz,
        vol_level=vol,
    )


def _overnight_skip_frames(
    data_dir: str,
    universe: str,
    *,
    sector_residual: bool,
    log_fn: Any | None = None,
    label_return: str = "overnight",
) -> tuple[dict[str, pd.DataFrame], int, float, list[dict[str, Any]]]:
    """Fit the promoted skip on TRAIN for one hedge / label mode."""
    cfg = overnight_skip_data_config(
        data_dir,
        universe,
        sector_residual=bool(sector_residual),
        label_return=str(label_return or "overnight"),
    )
    if log_fn:
        hedge = (
            "sector ETF (SPY fallback if parquet missing)"
            if sector_residual
            else "SPY/market"
        )
        log_fn(f"{label_return} residual skip hedge={hedge}")
    bundle = build_datasets(cfg, log_fn=log_fn)
    weights, bias, train_ic = fit_promoted_overnight_skip(bundle)
    min_names = int(bundle.get("cs_min_names", 30))
    px, _missing = load_split_px(bundle, cfg, log_fn=log_fn)
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        df, _x = _frame_for_split(bundle, split, px, weights, bias, min_names)
        frames[split] = df
    hedges = [
        {"symbol": m.get("symbol"), "hedge": m.get("hedge")}
        for m in (bundle.get("meta") or [])
    ]
    return frames, min_names, float(train_ic), hedges


# Overnight weight α. α=1 is pure overnight (A). TRAIN-only pick; no α<0.5.
ENSEMBLE_ALPHAS = (0.5, 0.6, 0.7, 0.8, 1.0)
ENSEMBLE_COVER_VAL = 0.30
# IDEA 5 locked fixed blend (compare target for adaptive α).
FIXED_ENSEMBLE_ALPHA = 0.70
# Causal adaptive α: TRAIN grid of trail windows × IC→α rules.
ADAPTIVE_WINDOWS = (20, 60, 120)
ADAPTIVE_RULES = ("relu_ratio", "signed_ratio", "softmax")
ADAPTIVE_WARMUP_ALPHA = 0.70


def within_date_z(pred: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score of scores on each date. Causal (no labels)."""
    if pred is None or pred.empty:
        return pd.DataFrame()
    p = pred.astype(np.float64)
    mu = p.mean(axis=1, skipna=True)
    sd = p.std(axis=1, skipna=True, ddof=0)
    sd = sd.mask((~np.isfinite(sd)) | (sd <= 1e-12))
    return p.sub(mu, axis=0).div(sd, axis=0)


def blend_cs_scores(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    weight: float,
) -> pd.DataFrame:
    """``w * z(A) + (1-w) * z(B)`` within date. Inner-join dates and names."""
    if pred_a.empty or pred_b.empty:
        return pd.DataFrame()
    a, b = pred_a.align(pred_b, join="inner")
    if a.empty:
        return pd.DataFrame()
    za = within_date_z(a)
    zb = within_date_z(b)
    w = float(weight)
    return w * za + (1.0 - w) * zb


def score_ensemble_alpha(
    frame_on: pd.DataFrame,
    frame_cc: pd.DataFrame,
    alpha: float,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Overnight live long-only q20 on ``α z(A) + (1-α) z(B)``. A = overnight."""
    empty = {
        "name": f"ens_a{float(alpha):.2f}",
        "weight": float(alpha),
        "alpha": float(alpha),
        "unlevered_net_ir": float("nan"),
        "net_ir": float("nan"),
        "unlevered_max_dd": float("nan"),
        "mean_cost_unlev_bp": float("nan"),
        "n_dates": 0.0,
        "coverage": float("nan"),
    }
    if frame_on.empty or frame_cc.empty:
        return empty
    pred_a = frame_to_wide(frame_on, "pred")
    pred_b = frame_to_wide(frame_cc, "pred")
    _p, y, r_on, tz, vol = _wide_from_frame(frame_on)
    if pred_a.empty or pred_b.empty or y.empty or pred_a.shape[1] < 2:
        return empty
    pred = blend_cs_scores(pred_a, pred_b, float(alpha))
    if pred.empty or pred.shape[1] < 2:
        return empty
    idx = pred.index.intersection(y.index)
    if idx.empty:
        return empty
    stats = _lo_q20_book(
        pred.loc[idx],
        y.loc[idx],
        min_names=min_names,
        vol_target=vol_target,
        overnight_r=r_on.reindex(idx) if r_on is not None else None,
        turnover_z=tz.reindex(idx) if tz is not None else None,
        vol_level=vol.reindex(idx) if vol is not None else None,
    )
    n_cal = float(len(idx))
    n_book = _as_float(stats.get("n_dates"))
    cover = (
        float(n_book / n_cal)
        if n_cal > 0 and np.isfinite(n_book)
        else float("nan")
    )
    return {
        "name": f"ens_a{float(alpha):.2f}",
        "weight": float(alpha),
        "alpha": float(alpha),
        "unlevered_net_ir": stats.get("unlevered_net_ir"),
        "net_ir": stats.get("net_ir"),
        "unlevered_max_dd": stats.get("unlevered_max_dd"),
        "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
        "n_dates": n_book,
        "coverage": cover,
    }


def ensemble_on_cc_grid(
    frame_on: pd.DataFrame,
    frame_cc: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Score the TRAIN α grid on one split. Does not pick α."""
    empty = {
        "rows": [],
        "best": {},
        "baseline": {},
        "note": (
            "s = α z(overnight sector residual) + (1-α) z(close-to-close residual); "
            "book is --live-costs --long-only q20 overnight. α=1 is pure A. "
            "α is fit on TRAIN only."
        ),
    }
    if frame_on.empty or frame_cc.empty:
        return empty
    rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] = {}
    for alpha in ENSEMBLE_ALPHAS:
        row = score_ensemble_alpha(
            frame_on,
            frame_cc,
            alpha,
            min_names=min_names,
            vol_target=vol_target,
        )
        if not np.isfinite(_as_float(row.get("unlevered_net_ir"))):
            continue
        rows.append(row)
        if abs(float(alpha) - 1.0) < 1e-12:
            baseline = dict(row)
    if not rows:
        return empty
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            -_as_float(r.get("alpha"), default=0.0),
            str(r.get("name")),
        )
    )
    return {
        "rows": rows,
        "best": dict(rows[0]),
        "baseline": baseline,
        "note": empty["note"],
    }


def fit_ensemble_on_train(
    frame_on: pd.DataFrame,
    frame_cc: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Select α on TRAIN only. VAL/TEST must never enter."""
    grid = ensemble_on_cc_grid(
        frame_on, frame_cc, min_names=min_names, vol_target=vol_target
    )
    rows = list(grid.get("rows") or [])
    baseline = dict(grid.get("baseline") or {})
    chosen: dict[str, Any] = {}
    best_ir = -1e18
    for row in rows:
        ir = _as_float(row.get("unlevered_net_ir"))
        cover = _as_float(row.get("coverage"))
        if not np.isfinite(ir):
            continue
        if np.isfinite(cover) and cover < ENSEMBLE_COVER_VAL:
            continue
        # Prefer higher IR; ties go to larger α (closer to overnight-only).
        alpha = _as_float(row.get("alpha"), default=0.0)
        better = ir > best_ir + 1e-12
        tie = abs(ir - best_ir) <= 1e-12 and alpha > _as_float(
            chosen.get("alpha"), default=-1.0
        )
        if better or tie:
            best_ir = ir
            chosen = dict(row)
    if not chosen:
        chosen = dict(baseline)
    return {
        "rows": rows,
        "chosen": chosen,
        "baseline": baseline,
        "fit_split": "train",
        "alphas": list(ENSEMBLE_ALPHAS),
        "note": grid.get("note"),
    }


def decide_ensemble_promote(
    *,
    val_overnight: dict[str, Any],
    val_chosen: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only vs pure overnight (α=1). TEST never enters. α is TRAIN-chosen."""
    baseline = dict(val_overnight or {})
    scored = dict(val_chosen or {})
    ir_a = _as_float(baseline.get("unlevered_net_ir"))
    ir_b = _as_float(scored.get("unlevered_net_ir"))
    dd_a = _as_float(baseline.get("unlevered_max_dd"))
    dd_b = _as_float(scored.get("unlevered_max_dd"))
    cover = _as_float(scored.get("coverage"))
    ir_delta = (
        float(ir_b - ir_a) if np.isfinite(ir_b) and np.isfinite(ir_a) else float("nan")
    )
    dd_delta = (
        float(dd_b - dd_a) if np.isfinite(dd_b) and np.isfinite(dd_a) else float("nan")
    )
    alpha = _as_float(chosen.get("alpha", chosen.get("weight")), default=1.0)
    same = (not chosen) or abs(alpha - 1.0) < 1e-12
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    cover_ok = bool(not np.isfinite(cover) or cover >= ENSEMBLE_COVER_VAL)
    promote = bool((not same) and ir_ok and dd_ok and cover_ok)
    if same:
        reason = (
            "NO PROMOTE: TRAIN chose α=1.0 (overnight-only). "
            f"VAL A IR {ir_a:+.3f}. Keep overnight sector ranks."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE: TRAIN α={alpha:.2f} but VAL coverage "
            f"{100 * cover:.0f}% < {100 * ENSEMBLE_COVER_VAL:.0f}%. "
            "Keep overnight sector ranks."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE: TRAIN α={alpha:.2f} VAL IR {ir_b:+.3f} vs overnight "
            f"{ir_a:+.3f} (delta {ir_delta:+.3f} < {LO_IR_LIFT:.2f}). "
            "Keep overnight sector ranks."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE: ensemble α={alpha:.2f} IR lift {ir_delta:+.3f} but VAL "
            f"max DD {dd_b:+.3f} vs A {dd_a:+.3f} exceeds {LO_DD_TOL:.2f}."
        )
    else:
        reason = (
            f"PROMOTE ensemble α={alpha:.2f}: VAL IR {ir_b:+.3f} vs overnight "
            f"{ir_a:+.3f} (delta {ir_delta:+.3f}), max DD {dd_b:+.3f} vs "
            f"{dd_a:+.3f}, coverage {100 * cover:.0f}%."
        )
    return {
        "promote_ensemble": promote,
        "gated_on": "val",
        "reason": reason,
        "spec": {"alpha": float(alpha) if promote else 1.0, "weight": float(alpha) if promote else 1.0},
        "chosen": scored,
        "baseline": baseline,
        "train_alpha": float(alpha) if np.isfinite(alpha) else 1.0,
        "ir_overnight": ir_a,
        "ir_ensemble": ir_b,
        "ir_delta": ir_delta,
        "dd_overnight": dd_a,
        "dd_ensemble": dd_b,
        "dd_delta": dd_delta,
        "coverage": cover,
        "cover_min_val": ENSEMBLE_COVER_VAL,
        "ir_lift": LO_IR_LIFT,
        "net_ir_overnight": _as_float(baseline.get("net_ir")),
        "net_ir_ensemble": _as_float(scored.get("net_ir")),
    }


def _concat_eval_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Stack name-date eval frames; last row wins on (date, symbol) overlap."""
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "date" in out.columns and "symbol" in out.columns:
        out = (
            out.sort_values(["date", "symbol"])
            .drop_duplicates(["date", "symbol"], keep="last")
            .reset_index(drop=True)
        )
    return out


def map_ics_to_alpha(ic_a: float, ic_b: float, rule: str) -> float:
    """Map trailing CS ICs of A (overnight) and B (c2c) → α in [0, 1].

    NaN ICs (warmup) stay NaN so the caller can fill with the warmup α.
    """
    a = float(ic_a) if ic_a is not None else float("nan")
    b = float(ic_b) if ic_b is not None else float("nan")
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    eps = 1e-8
    kind = str(rule or "relu_ratio")
    if kind == "relu_ratio":
        pa, pb = max(a, 0.0), max(b, 0.0)
        return float(np.clip(pa / (pa + pb + eps), 0.0, 1.0))
    if kind == "signed_ratio":
        # Owen example: clip(IC_A / (IC_A+IC_B+eps), 0, 1). Both weak → overnight.
        s = a + b
        if s <= 0.0:
            return 1.0
        return float(np.clip(a / (s + eps), 0.0, 1.0))
    if kind == "softmax":
        ea = float(np.exp(np.clip(a, -20.0, 20.0)))
        eb = float(np.exp(np.clip(b, -20.0, 20.0)))
        return float(ea / (ea + eb + eps))
    raise ValueError(f"unknown adaptive blend rule {rule!r}")


def adaptive_alpha_series(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    y: pd.DataFrame,
    *,
    window: int,
    rule: str,
    min_names: int = 3,
    warmup_alpha: float = ADAPTIVE_WARMUP_ALPHA,
) -> pd.Series:
    """Causal per-date α_t from trailing mean CS IC of A vs B (dates < t).

    Both ICs are vs overnight residual ``y`` (the live book target). Date
    ``t``'s overnight never enters α_t. Warmup (too few prior ICs) uses
    ``warmup_alpha`` (IDEA 5 locked 0.70). Next open is never a feature.
    """
    if pred_a is None or pred_b is None or y is None:
        return pd.Series(dtype=np.float64)
    if pred_a.empty or pred_b.empty or y.empty:
        return pd.Series(dtype=np.float64)
    a, b = pred_a.align(pred_b, join="inner")
    a, y2 = a.align(y, join="inner")
    b = b.reindex(index=a.index, columns=a.columns)
    if a.empty or a.shape[1] < 2:
        return pd.Series(dtype=np.float64)
    trail_a = trailing_mean_cs_ic(a, y2, window=int(window), min_names=int(min_names))
    trail_b = trailing_mean_cs_ic(b, y2, window=int(window), min_names=int(min_names))
    idx = a.index
    out = pd.Series(np.nan, index=idx, dtype=np.float64)
    for t in idx:
        ia = trail_a.loc[t] if t in trail_a.index else float("nan")
        ib = trail_b.loc[t] if t in trail_b.index else float("nan")
        out.loc[t] = map_ics_to_alpha(ia, ib, rule)
    fill = float(warmup_alpha)
    if not np.isfinite(fill):
        fill = ADAPTIVE_WARMUP_ALPHA
    return out.where(np.isfinite(out), fill)


def blend_cs_scores_adaptive(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    alpha_t: pd.Series,
    *,
    warmup_alpha: float = ADAPTIVE_WARMUP_ALPHA,
) -> pd.DataFrame:
    """``α_t * z(A) + (1-α_t) * z(B)`` with a per-date causal α."""
    if pred_a is None or pred_b is None or pred_a.empty or pred_b.empty:
        return pd.DataFrame()
    a, b = pred_a.align(pred_b, join="inner")
    if a.empty:
        return pd.DataFrame()
    za = within_date_z(a)
    zb = within_date_z(b)
    fill = float(warmup_alpha) if np.isfinite(float(warmup_alpha)) else ADAPTIVE_WARMUP_ALPHA
    if alpha_t is None or len(alpha_t) == 0:
        al = pd.Series(fill, index=za.index, dtype=np.float64)
    else:
        al = alpha_t.reindex(za.index).astype(np.float64)
        al = al.where(np.isfinite(al), fill)
    al = al.clip(0.0, 1.0)
    return za.mul(al, axis=0) + zb.mul(1.0 - al, axis=0)


def score_adaptive_ensemble(
    frame_on: pd.DataFrame,
    frame_cc: pd.DataFrame,
    *,
    window: int,
    rule: str,
    min_names: int,
    vol_target: float = 0.15,
    hist_on: pd.DataFrame | None = None,
    hist_cc: pd.DataFrame | None = None,
    warmup_alpha: float = ADAPTIVE_WARMUP_ALPHA,
) -> dict[str, Any]:
    """Overnight live long-only q20 on causal α_t z(A) + (1-α_t) z(B)."""
    name = f"adp_W{int(window)}_{rule}"
    empty = {
        "name": name,
        "window": int(window),
        "rule": str(rule),
        "unlevered_net_ir": float("nan"),
        "net_ir": float("nan"),
        "unlevered_max_dd": float("nan"),
        "mean_cost_unlev_bp": float("nan"),
        "n_dates": 0.0,
        "coverage": float("nan"),
        "mean_alpha": float("nan"),
        "p25_alpha": float("nan"),
        "p50_alpha": float("nan"),
        "p75_alpha": float("nan"),
    }
    if frame_on is None or frame_cc is None or frame_on.empty or frame_cc.empty:
        return empty
    if int(window) <= 0 or not str(rule):
        return empty
    book_pred_a = frame_to_wide(frame_on, "pred")
    book_pred_b = frame_to_wide(frame_cc, "pred")
    _p, y_book, r_on, tz, vol = _wide_from_frame(frame_on)
    hist_a_df = hist_on if hist_on is not None and not hist_on.empty else frame_on
    hist_b_df = hist_cc if hist_cc is not None and not hist_cc.empty else frame_cc
    pred_a = frame_to_wide(hist_a_df, "pred")
    pred_b = frame_to_wide(hist_b_df, "pred")
    y_hist = frame_to_wide(hist_a_df, "y")
    if (
        book_pred_a.empty
        or book_pred_b.empty
        or pred_a.empty
        or pred_b.empty
        or y_hist.empty
        or y_book.empty
        or book_pred_a.shape[1] < 2
    ):
        return empty
    alpha_all = adaptive_alpha_series(
        pred_a,
        pred_b,
        y_hist,
        window=int(window),
        rule=str(rule),
        min_names=int(min_names),
        warmup_alpha=float(warmup_alpha),
    )
    pred = blend_cs_scores_adaptive(
        book_pred_a, book_pred_b, alpha_all, warmup_alpha=float(warmup_alpha)
    )
    if pred.empty or pred.shape[1] < 2:
        return empty
    idx = pred.index.intersection(y_book.index)
    if idx.empty:
        return empty
    stats = _lo_q20_book(
        pred.loc[idx],
        y_book.loc[idx],
        min_names=min_names,
        vol_target=vol_target,
        overnight_r=r_on.reindex(idx) if r_on is not None else None,
        turnover_z=tz.reindex(idx) if tz is not None else None,
        vol_level=vol.reindex(idx) if vol is not None else None,
    )
    n_cal = float(len(idx))
    n_book = _as_float(stats.get("n_dates"))
    cover = (
        float(n_book / n_cal)
        if n_cal > 0 and np.isfinite(n_book)
        else float("nan")
    )
    al_book = alpha_all.reindex(idx).astype(np.float64)
    al_book = al_book[np.isfinite(al_book.to_numpy())]
    return {
        "name": name,
        "window": int(window),
        "rule": str(rule),
        "unlevered_net_ir": stats.get("unlevered_net_ir"),
        "net_ir": stats.get("net_ir"),
        "unlevered_max_dd": stats.get("unlevered_max_dd"),
        "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
        "n_dates": n_book,
        "coverage": cover,
        "mean_alpha": float(al_book.mean()) if len(al_book) else float("nan"),
        "p25_alpha": float(al_book.quantile(0.25)) if len(al_book) else float("nan"),
        "p50_alpha": float(al_book.quantile(0.50)) if len(al_book) else float("nan"),
        "p75_alpha": float(al_book.quantile(0.75)) if len(al_book) else float("nan"),
    }


def adaptive_ensemble_grid(
    frame_on: pd.DataFrame,
    frame_cc: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
    hist_on: pd.DataFrame | None = None,
    hist_cc: pd.DataFrame | None = None,
    warmup_alpha: float = ADAPTIVE_WARMUP_ALPHA,
) -> dict[str, Any]:
    """Score the TRAIN (W, rule) grid on one split. Does not pick."""
    empty = {
        "rows": [],
        "best": {},
        "note": (
            "s = α_t z(overnight sector residual) + (1-α_t) z(c2c residual); "
            "α_t from trailing mean CS IC of A vs B on dates < t only "
            f"(warmup α={ADAPTIVE_WARMUP_ALPHA:.2f}). Book is --live-costs "
            "--long-only q20 overnight. W/rule fit on TRAIN only."
        ),
    }
    if frame_on is None or frame_cc is None or frame_on.empty or frame_cc.empty:
        return empty
    rows: list[dict[str, Any]] = []
    for window in ADAPTIVE_WINDOWS:
        for rule in ADAPTIVE_RULES:
            row = score_adaptive_ensemble(
                frame_on,
                frame_cc,
                window=int(window),
                rule=str(rule),
                min_names=min_names,
                vol_target=vol_target,
                hist_on=hist_on,
                hist_cc=hist_cc,
                warmup_alpha=float(warmup_alpha),
            )
            if not np.isfinite(_as_float(row.get("unlevered_net_ir"))):
                continue
            rows.append(row)
    if not rows:
        return empty
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            -_as_float(r.get("mean_alpha"), default=0.0),
            -int(r.get("window") or 0),
            str(r.get("rule") or ""),
        )
    )
    return {"rows": rows, "best": dict(rows[0]), "note": empty["note"]}


def fit_adaptive_ensemble_on_train(
    frame_on: pd.DataFrame,
    frame_cc: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
    warmup_alpha: float = ADAPTIVE_WARMUP_ALPHA,
) -> dict[str, Any]:
    """Select (W, rule) on TRAIN only. VAL/TEST must never enter."""
    grid = adaptive_ensemble_grid(
        frame_on,
        frame_cc,
        min_names=min_names,
        vol_target=vol_target,
        hist_on=frame_on,
        hist_cc=frame_cc,
        warmup_alpha=float(warmup_alpha),
    )
    rows = list(grid.get("rows") or [])
    chosen: dict[str, Any] = {}
    best_ir = -1e18
    for row in rows:
        ir = _as_float(row.get("unlevered_net_ir"))
        cover = _as_float(row.get("coverage"))
        if not np.isfinite(ir):
            continue
        if np.isfinite(cover) and cover < ENSEMBLE_COVER_VAL:
            continue
        mean_a = _as_float(row.get("mean_alpha"), default=0.0)
        window = int(row.get("window") or 0)
        better = ir > best_ir + 1e-12
        tie = abs(ir - best_ir) <= 1e-12 and (
            mean_a > _as_float(chosen.get("mean_alpha"), default=-1.0) + 1e-12
            or (
                abs(mean_a - _as_float(chosen.get("mean_alpha"), default=-1.0)) <= 1e-12
                and window > int(chosen.get("window") or 0)
            )
        )
        if better or tie:
            best_ir = ir
            chosen = dict(row)
    return {
        "rows": rows,
        "chosen": chosen,
        "fit_split": "train",
        "windows": list(ADAPTIVE_WINDOWS),
        "rules": list(ADAPTIVE_RULES),
        "warmup_alpha": float(warmup_alpha),
        "note": grid.get("note"),
    }


def decide_adaptive_ensemble_promote(
    *,
    val_adaptive: dict[str, Any],
    val_fixed_070: dict[str, Any],
    val_alpha1: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only vs the better of fixed α=0.70 and α=1. TEST never enters."""
    scored = dict(val_adaptive or {})
    fix070 = dict(val_fixed_070 or {})
    fix1 = dict(val_alpha1 or {})
    ir_070 = _as_float(fix070.get("unlevered_net_ir"))
    ir_1 = _as_float(fix1.get("unlevered_net_ir"))
    dd_070 = _as_float(fix070.get("unlevered_max_dd"))
    dd_1 = _as_float(fix1.get("unlevered_max_dd"))
    # Best fixed baseline: higher VAL IR; ties prefer α=1 (pure overnight).
    use_070 = bool(
        np.isfinite(ir_070)
        and ((not np.isfinite(ir_1)) or ir_070 > ir_1 + 1e-12)
    )
    if use_070:
        baseline = fix070
        base_name = "fixed_a0.70"
        base_alpha = FIXED_ENSEMBLE_ALPHA
        ir_base = ir_070
        dd_base = dd_070
    else:
        baseline = fix1
        base_name = "fixed_a1.00"
        base_alpha = 1.0
        ir_base = ir_1
        dd_base = dd_1
    ir_b = _as_float(scored.get("unlevered_net_ir"))
    dd_b = _as_float(scored.get("unlevered_max_dd"))
    cover = _as_float(scored.get("coverage"))
    ir_delta = (
        float(ir_b - ir_base) if np.isfinite(ir_b) and np.isfinite(ir_base) else float("nan")
    )
    dd_delta = (
        float(dd_b - dd_base) if np.isfinite(dd_b) and np.isfinite(dd_base) else float("nan")
    )
    window = int((chosen or {}).get("window") or 0)
    rule = str((chosen or {}).get("rule") or "")
    have = bool(chosen) and window > 0 and bool(rule)
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    cover_ok = bool(not np.isfinite(cover) or cover >= ENSEMBLE_COVER_VAL)
    promote = bool(have and ir_ok and dd_ok and cover_ok)
    if not have:
        reason = (
            "NO PROMOTE: TRAIN did not pick a (W, rule). "
            f"Keep best fixed baseline {base_name} "
            f"(IR {ir_base:+.3f}; α=0.70 vs α=1)."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE: TRAIN W={window} {rule} but VAL coverage "
            f"{100 * cover:.0f}% < {100 * ENSEMBLE_COVER_VAL:.0f}%. "
            f"Keep {base_name}."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE: TRAIN W={window} {rule} VAL IR {ir_b:+.3f} vs best "
            f"fixed {base_name} {ir_base:+.3f} (delta {ir_delta:+.3f} < "
            f"{LO_IR_LIFT:.2f}). Keep {base_name}."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE: adaptive W={window} {rule} IR lift {ir_delta:+.3f} "
            f"but VAL max DD {dd_b:+.3f} vs {base_name} {dd_base:+.3f} "
            f"exceeds {LO_DD_TOL:.2f}."
        )
    else:
        reason = (
            f"PROMOTE adaptive α W={window} {rule}: VAL IR {ir_b:+.3f} vs "
            f"{base_name} {ir_base:+.3f} (delta {ir_delta:+.3f}), max DD "
            f"{dd_b:+.3f} vs {dd_base:+.3f}, coverage {100 * cover:.0f}%."
        )
    return {
        "promote_adaptive_ensemble": promote,
        "gated_on": "val",
        "reason": reason,
        "spec": (
            {"window": window, "rule": rule, "adaptive": True}
            if promote
            else {"alpha": float(base_alpha), "adaptive": False}
        ),
        "chosen": scored,
        "baseline": baseline,
        "baseline_name": base_name,
        "baseline_alpha": float(base_alpha),
        "train_window": window,
        "train_rule": rule,
        "ir_baseline": ir_base,
        "ir_adaptive": ir_b,
        "ir_delta": ir_delta,
        "ir_fixed_070": ir_070,
        "ir_alpha1": ir_1,
        "dd_baseline": dd_base,
        "dd_adaptive": dd_b,
        "dd_delta": dd_delta,
        "dd_fixed_070": dd_070,
        "dd_alpha1": dd_1,
        "coverage": cover,
        "cover_min_val": ENSEMBLE_COVER_VAL,
        "ir_lift": LO_IR_LIFT,
        "dd_tol": LO_DD_TOL,
        "mean_alpha": _as_float(scored.get("mean_alpha")),
        "net_ir_baseline": _as_float(baseline.get("net_ir")),
        "net_ir_adaptive": _as_float(scored.get("net_ir")),
    }


STICKY_ENTERS = (0.10, 0.15, 0.20)
STICKY_EXITS = (0.30, 0.40, 0.50)
STICKY_COVER_VAL = 0.30


def score_sticky_book(
    frame: pd.DataFrame,
    *,
    q_enter: float,
    q_exit: float,
    min_names: int,
    vol_target: float = 0.15,
    held0: set | frozenset | None = None,
) -> dict[str, Any]:
    """Overnight live long-only sticky vs flatten-every-night q20 costs."""
    empty = {
        "name": f"sticky_e{int(round(100 * float(q_enter)))}_x{int(round(100 * float(q_exit)))}",
        "q_enter": float(q_enter),
        "q_exit": float(q_exit),
        "unlevered_net_ir": float("nan"),
        "net_ir": float("nan"),
        "unlevered_max_dd": float("nan"),
        "mean_cost_unlev_bp": float("nan"),
        "mean_turnover": float("nan"),
        "mean_name_churn": float("nan"),
        "mean_n_held": float("nan"),
        "n_dates": 0.0,
        "coverage": float("nan"),
        "sticky_coverage": float("nan"),
    }
    if frame is None or frame.empty:
        return empty
    pred, y, r_on, tz, vol = _wide_from_frame(frame)
    if pred.empty or pred.shape[1] < 2:
        return empty
    stats = _lo_q20_book(
        pred,
        y,
        min_names=min_names,
        vol_target=vol_target,
        overnight_r=r_on,
        turnover_z=tz,
        vol_level=vol,
        sticky_q_enter=float(q_enter),
        sticky_q_exit=float(q_exit),
        sticky_held0=held0,
    )
    n_cal = float(len(pred.index.intersection(y.index)))
    n_book = _as_float(stats.get("n_dates"))
    cover = (
        float(n_book / n_cal)
        if n_cal > 0 and np.isfinite(n_book)
        else _as_float(stats.get("sticky_coverage"))
    )
    invested = _as_float(stats.get("sticky_coverage"))
    if np.isfinite(invested):
        cover = invested
    return {
        "name": empty["name"],
        "q_enter": float(q_enter),
        "q_exit": float(q_exit),
        "unlevered_net_ir": stats.get("unlevered_net_ir"),
        "net_ir": stats.get("net_ir"),
        "unlevered_max_dd": stats.get("unlevered_max_dd"),
        "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
        "mean_turnover": stats.get("mean_turnover"),
        "mean_name_churn": stats.get("mean_name_churn"),
        "mean_n_held": stats.get("mean_n_held"),
        "n_dates": n_book,
        "coverage": cover,
        "sticky_coverage": invested,
    }


def score_q20_rebuild(
    frame: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Always-rebuild q20 equal — the live default baseline."""
    row = score_sticky_book(
        frame,
        q_enter=0.0,
        q_exit=0.0,
        min_names=min_names,
        vol_target=vol_target,
    )
    if frame is None or frame.empty:
        row["name"] = "q20_rebuild"
        return row
    pred, y, r_on, tz, vol = _wide_from_frame(frame)
    if pred.empty or pred.shape[1] < 2:
        row["name"] = "q20_rebuild"
        return row
    stats = _lo_q20_book(
        pred,
        y,
        min_names=min_names,
        vol_target=vol_target,
        overnight_r=r_on,
        turnover_z=tz,
        vol_level=vol,
    )
    n_cal = float(len(pred.index.intersection(y.index)))
    n_book = _as_float(stats.get("n_dates"))
    cover = (
        float(n_book / n_cal)
        if n_cal > 0 and np.isfinite(n_book)
        else 1.0
    )
    invested = _as_float(stats.get("sticky_coverage"))
    if np.isfinite(invested):
        cover = invested
    return {
        "name": "q20_rebuild",
        "q_enter": 0.20,
        "q_exit": 0.20,
        "unlevered_net_ir": stats.get("unlevered_net_ir"),
        "net_ir": stats.get("net_ir"),
        "unlevered_max_dd": stats.get("unlevered_max_dd"),
        "mean_cost_unlev_bp": stats.get("mean_cost_unlev_bp"),
        "mean_turnover": stats.get("mean_turnover"),
        "mean_name_churn": stats.get("mean_name_churn"),
        "mean_n_held": stats.get("mean_n_held"),
        "n_dates": n_book,
        "coverage": cover,
        "sticky_coverage": invested,
    }


def sticky_grid(
    frame: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
    held0: set | frozenset | None = None,
) -> dict[str, Any]:
    """Score the TRAIN (q_enter, q_exit) grid on one split. Does not pick."""
    note = (
        "sticky long-only: enter top q_enter, hold while in top q_exit; "
        "equal-weight active set; empty = flat (still in IR). "
        "Ranks = sector-overnight residual skip. α-ensemble is not the default path. "
        "(q_enter, q_exit) fit on TRAIN only."
    )
    empty = {"rows": [], "best": {}, "baseline": {}, "note": note}
    if frame is None or frame.empty:
        return empty
    rows: list[dict[str, Any]] = []
    for qe in STICKY_ENTERS:
        for qx in STICKY_EXITS:
            if qx <= qe:
                continue
            row = score_sticky_book(
                frame,
                q_enter=qe,
                q_exit=qx,
                min_names=min_names,
                vol_target=vol_target,
                held0=held0,
            )
            if not np.isfinite(_as_float(row.get("unlevered_net_ir"))):
                continue
            rows.append(row)
    baseline = score_q20_rebuild(frame, min_names=min_names, vol_target=vol_target)
    if not rows:
        return {**empty, "baseline": baseline}
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("unlevered_net_ir"), default=-1e9),
            _as_float(r.get("mean_name_churn"), default=1e9),
            str(r.get("name")),
        )
    )
    return {
        "rows": rows,
        "best": dict(rows[0]),
        "baseline": baseline,
        "note": note,
    }


def fit_sticky_on_train(
    frame: pd.DataFrame,
    *,
    min_names: int,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Select (q_enter, q_exit) on TRAIN only. VAL/TEST must never enter."""
    grid = sticky_grid(frame, min_names=min_names, vol_target=vol_target, held0=None)
    rows = list(grid.get("rows") or [])
    baseline = dict(grid.get("baseline") or {})
    chosen: dict[str, Any] = {}
    best_ir = -1e18
    for row in rows:
        ir = _as_float(row.get("unlevered_net_ir"))
        cover = _as_float(row.get("coverage"))
        if not np.isfinite(ir):
            continue
        if np.isfinite(cover) and cover < STICKY_COVER_VAL:
            continue
        churn = _as_float(row.get("mean_name_churn"), default=1e9)
        better = ir > best_ir + 1e-12
        tie = abs(ir - best_ir) <= 1e-12 and churn < _as_float(
            chosen.get("mean_name_churn"), default=1e9
        )
        if better or tie:
            best_ir = ir
            chosen = dict(row)
    return {
        "rows": rows,
        "chosen": chosen,
        "baseline": baseline,
        "fit_split": "train",
        "enters": list(STICKY_ENTERS),
        "exits": list(STICKY_EXITS),
        "note": grid.get("note"),
    }


def decide_sticky_promote(
    *,
    val_baseline: dict[str, Any],
    val_chosen: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """VAL-only vs always-rebuild q20. TEST never enters. Spec is TRAIN-chosen."""
    base = dict(val_baseline or {})
    scored = dict(val_chosen or {})
    ir_a = _as_float(base.get("unlevered_net_ir"))
    ir_b = _as_float(scored.get("unlevered_net_ir"))
    dd_a = _as_float(base.get("unlevered_max_dd"))
    dd_b = _as_float(scored.get("unlevered_max_dd"))
    cover = _as_float(scored.get("coverage"))
    ir_delta = (
        float(ir_b - ir_a) if np.isfinite(ir_b) and np.isfinite(ir_a) else float("nan")
    )
    dd_delta = (
        float(dd_b - dd_a) if np.isfinite(dd_b) and np.isfinite(dd_a) else float("nan")
    )
    qe = _as_float(chosen.get("q_enter"))
    qx = _as_float(chosen.get("q_exit"))
    have = bool(chosen) and np.isfinite(qe) and np.isfinite(qx) and qx > qe
    ir_ok = bool(np.isfinite(ir_delta) and ir_delta >= LO_IR_LIFT)
    dd_ok = bool(not np.isfinite(dd_delta) or dd_delta >= -LO_DD_TOL)
    cover_ok = bool(not np.isfinite(cover) or cover >= STICKY_COVER_VAL)
    promote = bool(have and ir_ok and dd_ok and cover_ok)
    if not have:
        reason = (
            "NO PROMOTE: TRAIN did not select a (q_enter, q_exit) with "
            f"coverage >= {STICKY_COVER_VAL:.0%}. Keep always-rebuild q20."
        )
    elif not cover_ok:
        reason = (
            f"NO PROMOTE: TRAIN sticky e{100 * qe:.0f}/x{100 * qx:.0f} but VAL "
            f"coverage {100 * cover:.0f}% < {100 * STICKY_COVER_VAL:.0f}%. "
            "Keep always-rebuild q20."
        )
    elif not ir_ok:
        reason = (
            f"NO PROMOTE: TRAIN sticky e{100 * qe:.0f}/x{100 * qx:.0f} VAL IR "
            f"{ir_b:+.3f} vs q20 {ir_a:+.3f} (delta {ir_delta:+.3f} < "
            f"{LO_IR_LIFT:.2f}). Keep always-rebuild q20."
        )
    elif not dd_ok:
        reason = (
            f"NO PROMOTE: sticky e{100 * qe:.0f}/x{100 * qx:.0f} IR lift "
            f"{ir_delta:+.3f} but VAL max DD {dd_b:+.3f} vs q20 {dd_a:+.3f} "
            f"exceeds {LO_DD_TOL:.2f}."
        )
    else:
        reason = (
            f"PROMOTE sticky e{100 * qe:.0f}/x{100 * qx:.0f}: VAL IR {ir_b:+.3f} "
            f"vs q20 {ir_a:+.3f} (delta {ir_delta:+.3f}), max DD {dd_b:+.3f} vs "
            f"{dd_a:+.3f}, coverage {100 * cover:.0f}%, name_churn "
            f"{_as_float(scored.get('mean_name_churn')):.3f} vs "
            f"{_as_float(base.get('mean_name_churn')):.3f}."
        )
    return {
        "promote_sticky": promote,
        "gated_on": "val",
        "reason": reason,
        "spec": (
            {"q_enter": float(qe), "q_exit": float(qx)}
            if promote
            else {"q_enter": 0.20, "q_exit": 0.20}
        ),
        "chosen": scored,
        "baseline": base,
        "train_q_enter": float(qe) if np.isfinite(qe) else 0.20,
        "train_q_exit": float(qx) if np.isfinite(qx) else 0.20,
        "ir_q20": ir_a,
        "ir_sticky": ir_b,
        "ir_delta": ir_delta,
        "dd_q20": dd_a,
        "dd_sticky": dd_b,
        "dd_delta": dd_delta,
        "coverage": cover,
        "cover_min_val": STICKY_COVER_VAL,
        "ir_lift": LO_IR_LIFT,
        "name_churn_q20": _as_float(base.get("mean_name_churn")),
        "name_churn_sticky": _as_float(scored.get("mean_name_churn")),
        "mean_turnover_q20": _as_float(base.get("mean_turnover")),
        "mean_turnover_sticky": _as_float(scored.get("mean_turnover")),
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
    book_aligned_fit = fit_book_aligned_on_train(frames["train"], min_names=min_names)
    conviction_live = compare_conviction_live(
        {k: frames[k] for k in ("train", "val", "test") if k in frames},
        chosen=book_aligned_fit.get("chosen") or {},
        min_names=min_names,
        vol_target=vol_target,
    )
    conviction_live_promotion = decide_conviction_live_promote(
        val_q20=(conviction_live.get("val") or {}).get("q20") or {},
        val_chosen=(conviction_live.get("val") or {}).get("chosen") or {},
        chosen=book_aligned_fit.get("chosen") or {},
    )
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
    ics_fit = fit_ic_scale_on_train(
        frames["train"], min_names=min_names, vol_target=vol_target
    )
    ics_chosen = ics_fit.get("chosen") or {}
    ics_W = int(ics_chosen.get("window") or 0)
    ics_tau = float(ics_chosen.get("tau") or 0.0)
    ics_smax = float(ics_chosen.get("s_max") or 1.0)

    def _scaled_lo(
        split: str,
        hist: tuple[str, ...],
        window: int,
        tau: float,
        s_max: float = 1.0,
    ) -> dict[str, Any]:
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
            ic_scale_window=window,
            ic_scale_tau=tau,
            ic_scale_trail=trail,
            ic_scale_smax=s_max,
        )

    ics_val_always = _scaled_lo("val", ("train", "val"), 0, 0.0)
    ics_val_scaled = (
        _scaled_lo("val", ("train", "val"), ics_W, ics_tau, ics_smax)
        if ics_W
        else dict(ics_val_always)
    )
    ics_test_always = _scaled_lo("test", ("train", "val", "test"), 0, 0.0)
    ics_test_scaled = (
        _scaled_lo("test", ("train", "val", "test"), ics_W, ics_tau, ics_smax)
        if ics_W
        else dict(ics_test_always)
    )
    ics_promo = decide_ic_scale_promote(
        val_always=ics_val_always, val_scaled=ics_val_scaled, chosen=ics_chosen
    )
    disp_fit = fit_disp_gate_on_train(
        frames["train"], min_names=min_names, vol_target=vol_target
    )
    disp_chosen = disp_fit.get("chosen") or {}
    disp_kind = str(disp_chosen.get("kind") or "")
    disp_W = int(disp_chosen.get("window") or 0)
    disp_tau = float(disp_chosen.get("tau") or float("nan"))

    def _disp_lo(split: str, hist: tuple[str, ...], kind: str, window: int, tau: float) -> dict[str, Any]:
        if frames[split].empty:
            return {}
        pred, y, r_on, tz, vol = _wide_from_frame(frames[split])
        if pred.empty or pred.shape[1] < 2:
            return {}
        trail = None
        if kind:
            trail = _hist_disp_series(frames, hist, kind, window, min_names)
        return _lo_q20_book(
            pred,
            y,
            min_names=min_names,
            vol_target=vol_target,
            overnight_r=r_on,
            turnover_z=tz,
            vol_level=vol,
            disp_gate_trail=trail,
            disp_gate_tau=tau,
            disp_gate_kind=kind,
            disp_gate_window=window,
        )

    disp_val_always = _disp_lo("val", ("train", "val"), "", 0, float("nan"))
    disp_val_gated = (
        _disp_lo("val", ("train", "val"), disp_kind, disp_W, disp_tau)
        if disp_kind
        else dict(disp_val_always)
    )
    disp_test_always = _disp_lo("test", ("train", "val", "test"), "", 0, float("nan"))
    disp_test_gated = (
        _disp_lo("test", ("train", "val", "test"), disp_kind, disp_W, disp_tau)
        if disp_kind
        else dict(disp_test_always)
    )
    disp_promo = decide_disp_gate_promote(
        val_always=disp_val_always, val_gated=disp_val_gated, chosen=disp_chosen
    )
    wd_val = weekday_mask_grid(
        frames["val"], min_names=min_names, vol_target=vol_target
    )
    wd_promo = decide_weekday_promote(wd_val)
    wd_test = weekday_mask_grid(
        frames["test"], min_names=min_names, vol_target=vol_target
    )
    if log_fn:
        log_fn("IDEA 3: SPY/market overnight residual (A) vs sector-overnight (B)")
    spy_frames, _spy_min, spy_train_ic, _spy_hedges = _overnight_skip_frames(
        data_dir, universe, sector_residual=False, log_fn=None
    )
    spy_val_lo = _split_lo_q20(
        spy_frames.get("val", pd.DataFrame()),
        min_names=min_names,
        vol_target=vol_target,
    )
    spy_test_lo = _split_lo_q20(
        spy_frames.get("test", pd.DataFrame()),
        min_names=min_names,
        vol_target=vol_target,
    )
    sector_val_lo = (val.get("books") or {}).get("live_long_only") or {}
    sector_test_lo = (test.get("books") or {}).get("live_long_only") or {}
    bench = str(getattr(cfg, "benchmark_symbol", None) or "SPY").upper()
    hedge_rows = [
        {"symbol": m.get("symbol"), "hedge": m.get("hedge")}
        for m in (bundle.get("meta") or [])
    ]
    n_sector_hedges = sum(
        1
        for row in hedge_rows
        if str(row.get("hedge") or "").upper() not in ("", bench)
    )
    if log_fn:
        log_fn("IDEA 5: overnight ⊕ close-to-close rank ensemble (TRAIN-chosen α)")
    cc_frames, _cc_min, cc_train_ic, _cc_hedges = _overnight_skip_frames(
        data_dir,
        universe,
        sector_residual=True,
        log_fn=None,
        label_return="close",
    )
    ens_fit = fit_ensemble_on_train(
        frames["train"],
        cc_frames.get("train", pd.DataFrame()),
        min_names=min_names,
        vol_target=vol_target,
    )
    ens_val = ensemble_on_cc_grid(
        frames["val"],
        cc_frames.get("val", pd.DataFrame()),
        min_names=min_names,
        vol_target=vol_target,
    )
    ens_test = ensemble_on_cc_grid(
        frames["test"],
        cc_frames.get("test", pd.DataFrame()),
        min_names=min_names,
        vol_target=vol_target,
    )
    ens_alpha = _as_float(
        (ens_fit.get("chosen") or {}).get("alpha", (ens_fit.get("chosen") or {}).get("weight")),
        default=1.0,
    )
    if not np.isfinite(ens_alpha):
        ens_alpha = 1.0
    ens_val_chosen = score_ensemble_alpha(
        frames["val"],
        cc_frames.get("val", pd.DataFrame()),
        ens_alpha,
        min_names=min_names,
        vol_target=vol_target,
    )
    ens_val_base = score_ensemble_alpha(
        frames["val"],
        cc_frames.get("val", pd.DataFrame()),
        1.0,
        min_names=min_names,
        vol_target=vol_target,
    )
    ens_test_chosen = score_ensemble_alpha(
        frames["test"],
        cc_frames.get("test", pd.DataFrame()),
        ens_alpha,
        min_names=min_names,
        vol_target=vol_target,
    )
    ens_test_base = score_ensemble_alpha(
        frames["test"],
        cc_frames.get("test", pd.DataFrame()),
        1.0,
        min_names=min_names,
        vol_target=vol_target,
    )
    ens_promo = decide_ensemble_promote(
        val_overnight=ens_val_base,
        val_chosen=ens_val_chosen,
        chosen=ens_fit.get("chosen") or {},
    )
    ens_compare = {
        "train_cs_ic_overnight": float(train_ic),
        "train_cs_ic_c2c": cc_train_ic,
        "train_grid": ens_fit,
        "val_grid": ens_val,
        "test_grid": ens_test,
        "val_chosen": ens_val_chosen,
        "val_overnight": ens_val_base,
        "test_chosen": ens_test_chosen,
        "test_overnight": ens_test_base,
        "train_alpha": ens_alpha,
    }
    if log_fn:
        log_fn("IDEA 8: causal adaptive overnight⊕c2c α (TRAIN W/rule)")
    ens_val_070 = score_ensemble_alpha(
        frames["val"],
        cc_frames.get("val", pd.DataFrame()),
        FIXED_ENSEMBLE_ALPHA,
        min_names=min_names,
        vol_target=vol_target,
    )
    ens_test_070 = score_ensemble_alpha(
        frames["test"],
        cc_frames.get("test", pd.DataFrame()),
        FIXED_ENSEMBLE_ALPHA,
        min_names=min_names,
        vol_target=vol_target,
    )
    adp_fit = fit_adaptive_ensemble_on_train(
        frames["train"],
        cc_frames.get("train", pd.DataFrame()),
        min_names=min_names,
        vol_target=vol_target,
    )
    adp_chosen = adp_fit.get("chosen") or {}
    adp_W = int(adp_chosen.get("window") or 0)
    adp_rule = str(adp_chosen.get("rule") or "")
    hist_on_val = _concat_eval_frames(frames["train"], frames["val"])
    hist_cc_val = _concat_eval_frames(
        cc_frames.get("train", pd.DataFrame()),
        cc_frames.get("val", pd.DataFrame()),
    )
    hist_on_test = _concat_eval_frames(frames["train"], frames["val"], frames["test"])
    hist_cc_test = _concat_eval_frames(
        cc_frames.get("train", pd.DataFrame()),
        cc_frames.get("val", pd.DataFrame()),
        cc_frames.get("test", pd.DataFrame()),
    )
    adp_val_grid = adaptive_ensemble_grid(
        frames["val"],
        cc_frames.get("val", pd.DataFrame()),
        min_names=min_names,
        vol_target=vol_target,
        hist_on=hist_on_val,
        hist_cc=hist_cc_val,
    )
    adp_test_grid = adaptive_ensemble_grid(
        frames["test"],
        cc_frames.get("test", pd.DataFrame()),
        min_names=min_names,
        vol_target=vol_target,
        hist_on=hist_on_test,
        hist_cc=hist_cc_test,
    )
    adp_val_chosen = (
        score_adaptive_ensemble(
            frames["val"],
            cc_frames.get("val", pd.DataFrame()),
            window=adp_W,
            rule=adp_rule,
            min_names=min_names,
            vol_target=vol_target,
            hist_on=hist_on_val,
            hist_cc=hist_cc_val,
        )
        if adp_W and adp_rule
        else {}
    )
    adp_test_chosen = (
        score_adaptive_ensemble(
            frames["test"],
            cc_frames.get("test", pd.DataFrame()),
            window=adp_W,
            rule=adp_rule,
            min_names=min_names,
            vol_target=vol_target,
            hist_on=hist_on_test,
            hist_cc=hist_cc_test,
        )
        if adp_W and adp_rule
        else {}
    )
    adp_promo = decide_adaptive_ensemble_promote(
        val_adaptive=adp_val_chosen,
        val_fixed_070=ens_val_070,
        val_alpha1=ens_val_base,
        chosen=adp_chosen,
    )
    adp_compare = {
        "train_grid": adp_fit,
        "val_grid": adp_val_grid,
        "test_grid": adp_test_grid,
        "val_chosen": adp_val_chosen,
        "val_fixed_070": ens_val_070,
        "val_alpha1": ens_val_base,
        "test_chosen": adp_test_chosen,
        "test_fixed_070": ens_test_070,
        "test_alpha1": ens_test_base,
        "train_window": adp_W,
        "train_rule": adp_rule,
        "warmup_alpha": ADAPTIVE_WARMUP_ALPHA,
        "fixed_alpha": FIXED_ENSEMBLE_ALPHA,
    }
    if log_fn:
        log_fn("IDEA 6: sticky long-only enter/exit hysteresis (TRAIN-chosen q)")
    sticky_fit = fit_sticky_on_train(
        frames["train"], min_names=min_names, vol_target=vol_target
    )
    sticky_qe = _as_float((sticky_fit.get("chosen") or {}).get("q_enter"))
    sticky_qx = _as_float((sticky_fit.get("chosen") or {}).get("q_exit"))
    have_sticky = (
        np.isfinite(sticky_qe) and np.isfinite(sticky_qx) and sticky_qx > sticky_qe
    )
    train_pred = frame_to_wide(frames["train"], "pred")
    val_pred = frame_to_wide(frames["val"], "pred")
    sticky_held_val = (
        last_sticky_held(
            train_pred, q_enter=sticky_qe, q_exit=sticky_qx, min_names=min_names
        )
        if have_sticky
        else set()
    )
    hist_parts = [p for p in (train_pred, val_pred) if p is not None and not p.empty]
    hist_tv = pd.concat(hist_parts) if hist_parts else pd.DataFrame()
    if not hist_tv.empty:
        hist_tv = hist_tv.sort_index().groupby(level=0).last()
    sticky_held_test = (
        last_sticky_held(
            hist_tv, q_enter=sticky_qe, q_exit=sticky_qx, min_names=min_names
        )
        if have_sticky
        else set()
    )
    sticky_val = sticky_grid(
        frames["val"], min_names=min_names, vol_target=vol_target
    )
    sticky_test = sticky_grid(
        frames["test"], min_names=min_names, vol_target=vol_target
    )
    sticky_val_base = score_q20_rebuild(
        frames["val"], min_names=min_names, vol_target=vol_target
    )
    sticky_test_base = score_q20_rebuild(
        frames["test"], min_names=min_names, vol_target=vol_target
    )
    sticky_val_chosen = (
        score_sticky_book(
            frames["val"],
            q_enter=sticky_qe,
            q_exit=sticky_qx,
            min_names=min_names,
            vol_target=vol_target,
            held0=sticky_held_val,
        )
        if have_sticky
        else dict(sticky_val_base)
    )
    sticky_test_chosen = (
        score_sticky_book(
            frames["test"],
            q_enter=sticky_qe,
            q_exit=sticky_qx,
            min_names=min_names,
            vol_target=vol_target,
            held0=sticky_held_test,
        )
        if have_sticky
        else dict(sticky_test_base)
    )
    sticky_promo = decide_sticky_promote(
        val_baseline=sticky_val_base,
        val_chosen=sticky_val_chosen,
        chosen=sticky_fit.get("chosen") or {},
    )
    sticky_compare = {
        "train_grid": sticky_fit,
        "val_grid": sticky_val,
        "test_grid": sticky_test,
        "val_chosen": sticky_val_chosen,
        "val_baseline": sticky_val_base,
        "test_chosen": sticky_test_chosen,
        "test_baseline": sticky_test_base,
        "train_q_enter": sticky_qe if have_sticky else 0.20,
        "train_q_exit": sticky_qx if have_sticky else 0.20,
    }
    sector_promo = decide_sector_promote(
        val_spy=spy_val_lo,
        val_sector=sector_val_lo,
        n_sector_hedges=n_sector_hedges,
        n_names=len(hedge_rows),
    )
    sector_compare = {
        "a": "spy_market_overnight",
        "b": "sector_overnight",
        "fallback": "SPY when mapped sector ETF parquet is missing",
        "train_cs_ic_spy": spy_train_ic,
        "train_cs_ic_sector": float(train_ic),
        "n_sector_hedges": n_sector_hedges,
        "n_names": len(hedge_rows),
        "hedges": hedge_rows,
        "val_spy": spy_val_lo,
        "val_sector": sector_val_lo,
        "test_spy": spy_test_lo,
        "test_sector": sector_test_lo,
    }
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
        "book_aligned_fit": book_aligned_fit,
        "conviction_live": conviction_live,
        "conviction_live_promotion": conviction_live_promotion,
        "ic_gate_fit": ic_fit,
        "ic_gate_promotion": ic_promo,
        "ic_gate_val_always": ic_val_always,
        "ic_gate_val_gated": ic_val_gated,
        "ic_gate_test_always": ic_test_always,
        "ic_gate_test_gated": ic_test_gated,
        "ic_scale_fit": ics_fit,
        "ic_scale_promotion": ics_promo,
        "ic_scale_val_always": ics_val_always,
        "ic_scale_val_scaled": ics_val_scaled,
        "ic_scale_test_always": ics_test_always,
        "ic_scale_test_scaled": ics_test_scaled,
        "disp_gate_fit": disp_fit,
        "disp_gate_promotion": disp_promo,
        "disp_gate_val_always": disp_val_always,
        "disp_gate_val_gated": disp_val_gated,
        "disp_gate_test_always": disp_test_always,
        "disp_gate_test_gated": disp_test_gated,
        "weekday_val_grid": wd_val,
        "weekday_test_grid": wd_test,
        "weekday_promotion": wd_promo,
        "sector_compare": sector_compare,
        "sector_promotion": sector_promo,
        "ensemble_fit": ens_fit,
        "ensemble_compare": ens_compare,
        "ensemble_promotion": ens_promo,
        "adaptive_ensemble_fit": adp_fit,
        "adaptive_ensemble_compare": adp_compare,
        "adaptive_ensemble_promotion": adp_promo,
        "sticky_fit": sticky_fit,
        "sticky_compare": sticky_compare,
        "sticky_promotion": sticky_promo,
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
        _conviction_live_block(payload),
        "",
        f"PROMOTE LONG-ONLY REFINE (q/size/conf, TEST veto)? "
        f"{'YES' if (payload.get('lo_refine_promotion') or {}).get('promote_lo') else 'NO'}",
        f"  spec = {(payload.get('lo_refine_promotion') or {}).get('spec')}",
        f"  {(payload.get('lo_refine_promotion') or {}).get('reason')}",
        "",
        _ic_gate_block(payload),
        "",
        _ic_scale_block(payload),
        "",
        _disp_gate_block(payload),
        "",
        _weekday_block(payload),
        "",
        _sector_block(payload),
        "",
        _ensemble_block(payload),
        "",
        _adaptive_ensemble_block(payload),
        "",
        _sticky_block(payload),
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


def _ic_scale_block(payload: dict[str, Any]) -> str:
    promo = payload.get("ic_scale_promotion") or {}
    fit = payload.get("ic_scale_fit") or {}
    chosen = fit.get("chosen") or promo.get("chosen") or {}
    va = payload.get("ic_scale_val_always") or {}
    vs = payload.get("ic_scale_val_scaled") or {}
    ta = payload.get("ic_scale_test_always") or {}
    ts = payload.get("ic_scale_test_scaled") or {}
    lines = [
        f"PROMOTE IC-SCALE? {'YES' if promo.get('promote_ic_scale') else 'NO'}",
        f"  {fit.get('note')}",
        f"  TRAIN chose W={chosen.get('window')} τ={chosen.get('tau')}  "
        f"s_max={_fmt(chosen.get('s_max'), '.2f')}  "
        f"mean_s {_fmt(chosen.get('mean_ic_scale'), '.2f')}  "
        f"(fit_split={fit.get('fit_split')})",
        f"  {promo.get('reason')}",
        f"  VAL ungated    IR {_fmt(va.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(va.get('unlevered_max_dd'), '+.3f')}  "
        f"churn {_fmt(va.get('mean_name_churn'), '.3f')}",
        f"  VAL scaled     IR {_fmt(vs.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(vs.get('unlevered_max_dd'), '+.3f')}  "
        f"mean_s {_fmt(vs.get('mean_ic_scale'), '.2f')}  "
        f"churn {_fmt(vs.get('mean_name_churn'), '.3f')}  "
        f"cover {_fmt(100.0 * _as_float(vs.get('ic_scale_coverage')), '.0f')}%  "
        f"partial {int(_as_float(vs.get('ic_scale_n_partial'), 0.0))}/"
        f"{int(_as_float(vs.get('ic_scale_n_dates'), 0.0))}",
        f"  TEST ungated   IR {_fmt(ta.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(ta.get('unlevered_max_dd'), '+.3f')}  "
        f"churn {_fmt(ta.get('mean_name_churn'), '.3f')}  (report-only)",
        f"  TEST scaled    IR {_fmt(ts.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(ts.get('unlevered_max_dd'), '+.3f')}  "
        f"mean_s {_fmt(ts.get('mean_ic_scale'), '.2f')}  "
        f"churn {_fmt(ts.get('mean_name_churn'), '.3f')}  (report-only)",
        "  TRAIN (W, τ) grid (fit):",
    ]
    for row in list(fit.get("rows") or [])[:8]:
        lines.append(
            f"  {str(row.get('name') or ''):22} "
            f"IR {_fmt(row.get('unlevered_net_ir'), '+.3f')}  "
            f"maxDD {_fmt(row.get('unlevered_max_dd'), '+.3f')}  "
            f"mean_s {_fmt(row.get('mean_ic_scale'), '.2f')}  "
            f"churn {_fmt(row.get('mean_name_churn'), '.3f')}"
        )
    return "\n".join(lines)


def _disp_gate_block(payload: dict[str, Any]) -> str:
    promo = payload.get("disp_gate_promotion") or {}
    fit = payload.get("disp_gate_fit") or {}
    chosen = fit.get("chosen") or promo.get("chosen") or {}
    va = payload.get("disp_gate_val_always") or {}
    vg = payload.get("disp_gate_val_gated") or {}
    ta = payload.get("disp_gate_test_always") or {}
    tg = payload.get("disp_gate_test_gated") or {}

    def _cov(value: Any) -> str:
        x = _as_float(value)
        return "nan%" if not np.isfinite(x) else f"{100.0 * x:.0f}%"

    lines = [
        f"PROMOTE DISP-GATE? {'YES' if promo.get('promote_disp_gate') else 'NO'}",
        f"  {fit.get('note')}",
        f"  TRAIN chose {chosen.get('kind')} W={chosen.get('window')} "
        f"q={chosen.get('q')} τ={_fmt(chosen.get('tau'), '.4f')}  "
        f"cover {_cov(chosen.get('disp_gate_coverage'))}  "
        f"(fit_split={fit.get('fit_split')})",
        f"  {promo.get('reason')}",
        f"  VAL ungated    IR {_fmt(va.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(va.get('unlevered_max_dd'), '+.3f')}",
        f"  VAL gated      IR {_fmt(vg.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(vg.get('unlevered_max_dd'), '+.3f')}  "
        f"cover {_cov(vg.get('disp_gate_coverage'))}  "
        f"flat {int(_as_float(vg.get('disp_gate_n_flat'), 0.0))}/"
        f"{int(_as_float(vg.get('disp_gate_n_dates'), 0.0))}",
        f"  TEST ungated   IR {_fmt(ta.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(ta.get('unlevered_max_dd'), '+.3f')}  (report-only)",
        f"  TEST gated     IR {_fmt(tg.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(tg.get('unlevered_max_dd'), '+.3f')}  "
        f"cover {_cov(tg.get('disp_gate_coverage'))}  (report-only)",
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


def _sector_block(payload: dict[str, Any]) -> str:
    promo = payload.get("sector_promotion") or {}
    cmp_ = payload.get("sector_compare") or {}
    va = cmp_.get("val_spy") or {}
    vb = cmp_.get("val_sector") or {}
    ta = cmp_.get("test_spy") or {}
    tb = cmp_.get("test_sector") or {}
    n_sec = int(cmp_.get("n_sector_hedges") or 0)
    n_nm = int(cmp_.get("n_names") or 0)
    lines = [
        f"PROMOTE SECTOR-OVERNIGHT RESIDUAL? "
        f"{'YES' if promo.get('promote_sector') else 'NO'}",
        f"  A = SPY/market overnight residual   "
        f"train CS IC {_fmt(cmp_.get('train_cs_ic_spy'), '+.4f')}",
        f"  B = sector ETF overnight residual   "
        f"train CS IC {_fmt(cmp_.get('train_cs_ic_sector'), '+.4f')}  "
        f"({n_sec}/{n_nm} names mapped to a sector ETF; "
        f"{cmp_.get('fallback')})",
        f"  {promo.get('reason')}",
        f"  VAL A (SPY)     IR {_fmt(va.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(va.get('unlevered_max_dd'), '+.3f')}  "
        f"cost {_fmt(va.get('mean_cost_unlev_bp'), '.1f')} bp",
        f"  VAL B (sector)  IR {_fmt(vb.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(vb.get('unlevered_max_dd'), '+.3f')}  "
        f"cost {_fmt(vb.get('mean_cost_unlev_bp'), '.1f')} bp  "
        f"delta {_fmt(promo.get('ir_delta'), '+.3f')}",
        f"  TEST A (SPY)    IR {_fmt(ta.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(ta.get('unlevered_max_dd'), '+.3f')}  (report-only)",
        f"  TEST B (sector) IR {_fmt(tb.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(tb.get('unlevered_max_dd'), '+.3f')}  (report-only)",
    ]
    return "\n".join(lines)


def _ensemble_block(payload: dict[str, Any]) -> str:
    promo = payload.get("ensemble_promotion") or {}
    cmp_ = payload.get("ensemble_compare") or {}
    fit = payload.get("ensemble_fit") or cmp_.get("train_grid") or {}
    val = cmp_.get("val_grid") or {}
    test = cmp_.get("test_grid") or {}
    chosen = cmp_.get("val_chosen") or promo.get("chosen") or {}
    base = cmp_.get("val_overnight") or promo.get("baseline") or {}
    test_ch = cmp_.get("test_chosen") or {}
    test_a = cmp_.get("test_overnight") or {}
    train_ch = fit.get("chosen") or {}

    def _row(r: dict[str, Any]) -> str:
        return (
            f"  {str(r.get('name') or ''):14} "
            f"IR {_fmt(r.get('unlevered_net_ir'), '+.3f')}  "
            f"maxDD {_fmt(r.get('unlevered_max_dd'), '+.3f')}  "
            f"α={_fmt(r.get('alpha', r.get('weight')), '.2f')}  "
            f"cover {_fmt(100.0 * _as_float(r.get('coverage')), '.0f')}%"
        )

    lines = [
        f"PROMOTE OVERNIGHT⊕C2C ENSEMBLE? "
        f"{'YES' if promo.get('promote_ensemble') else 'NO'}",
        f"  {fit.get('note') or val.get('note')}",
        f"  train CS IC overnight {_fmt(cmp_.get('train_cs_ic_overnight'), '+.4f')}  "
        f"c2c {_fmt(cmp_.get('train_cs_ic_c2c'), '+.4f')}",
        f"  TRAIN chose α={_fmt(train_ch.get('alpha', train_ch.get('weight')), '.2f')}  "
        f"IR {_fmt(train_ch.get('unlevered_net_ir'), '+.3f')}  "
        f"(fit_split={fit.get('fit_split')})",
        f"  {promo.get('reason')}",
        f"  VAL A (α=1 overnight) IR {_fmt(base.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(base.get('unlevered_max_dd'), '+.3f')}  "
        f"liveIR {_fmt(base.get('net_ir'), '+.3f')}",
        f"  VAL TRAIN-α={_fmt(chosen.get('alpha', chosen.get('weight')), '.2f')}  "
        f"IR {_fmt(chosen.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(chosen.get('unlevered_max_dd'), '+.3f')}  "
        f"liveIR {_fmt(chosen.get('net_ir'), '+.3f')}  "
        f"cover {_fmt(100.0 * _as_float(chosen.get('coverage')), '.0f')}%",
        "  TRAIN α grid (fit):",
    ]
    for row in list(fit.get("rows") or []):
        lines.append(_row(row))
    lines.append("  VAL α grid (report; α not picked here):")
    for row in list(val.get("rows") or []):
        lines.append(_row(row))
    lines.append(
        f"  TEST A (α=1) IR {_fmt(test_a.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(test_a.get('unlevered_max_dd'), '+.3f')}  (report-only)"
    )
    lines.append(
        f"  TEST TRAIN-α IR {_fmt(test_ch.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(test_ch.get('unlevered_max_dd'), '+.3f')}  (report-only)"
    )
    lines.append("  TEST α grid (report-only):")
    for row in list(test.get("rows") or []):
        lines.append(_row(row))
    return "\n".join(lines)


def _adaptive_ensemble_block(payload: dict[str, Any]) -> str:
    promo = payload.get("adaptive_ensemble_promotion") or {}
    cmp_ = payload.get("adaptive_ensemble_compare") or {}
    fit = payload.get("adaptive_ensemble_fit") or cmp_.get("train_grid") or {}
    val = cmp_.get("val_grid") or {}
    test = cmp_.get("test_grid") or {}
    chosen = cmp_.get("val_chosen") or promo.get("chosen") or {}
    fix070 = cmp_.get("val_fixed_070") or {}
    fix1 = cmp_.get("val_alpha1") or {}
    test_ch = cmp_.get("test_chosen") or {}
    test_070 = cmp_.get("test_fixed_070") or {}
    test_1 = cmp_.get("test_alpha1") or {}
    train_ch = fit.get("chosen") or {}

    def _row(r: dict[str, Any]) -> str:
        return (
            f"  {str(r.get('name') or ''):22} "
            f"IR {_fmt(r.get('unlevered_net_ir'), '+.3f')}  "
            f"maxDD {_fmt(r.get('unlevered_max_dd'), '+.3f')}  "
            f"meanα {_fmt(r.get('mean_alpha'), '.2f')}  "
            f"cover {_fmt(100.0 * _as_float(r.get('coverage')), '.0f')}%"
        )

    lines = [
        f"PROMOTE ADAPTIVE OVERNIGHT⊕C2C α? "
        f"{'YES' if promo.get('promote_adaptive_ensemble') else 'NO'}",
        f"  {fit.get('note') or val.get('note')}",
        f"  TRAIN chose W={train_ch.get('window')} {train_ch.get('rule')}  "
        f"IR {_fmt(train_ch.get('unlevered_net_ir'), '+.3f')}  "
        f"meanα {_fmt(train_ch.get('mean_alpha'), '.2f')}  "
        f"(fit_split={fit.get('fit_split')})",
        f"  {promo.get('reason')}",
        f"  VAL fixed α=0.70 IR {_fmt(fix070.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(fix070.get('unlevered_max_dd'), '+.3f')}",
        f"  VAL fixed α=1.00 IR {_fmt(fix1.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(fix1.get('unlevered_max_dd'), '+.3f')}",
        f"  VAL best-fixed ({promo.get('baseline_name')}) IR "
        f"{_fmt(promo.get('ir_baseline'), '+.3f')}  "
        f"maxDD {_fmt(promo.get('dd_baseline'), '+.3f')}",
        f"  VAL adaptive W={chosen.get('window')} {chosen.get('rule')}  "
        f"IR {_fmt(chosen.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(chosen.get('unlevered_max_dd'), '+.3f')}  "
        f"meanα {_fmt(chosen.get('mean_alpha'), '.2f')}  "
        f"cover {_fmt(100.0 * _as_float(chosen.get('coverage')), '.0f')}%  "
        f"delta {_fmt(promo.get('ir_delta'), '+.3f')}",
        "  TRAIN (W, rule) grid (fit):",
    ]
    for row in list(fit.get("rows") or []):
        lines.append(_row(row))
    lines.append("  VAL (W, rule) grid (report; not picked here):")
    for row in list(val.get("rows") or []):
        lines.append(_row(row))
    lines.append(
        f"  TEST α=0.70 IR {_fmt(test_070.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(test_070.get('unlevered_max_dd'), '+.3f')}  (report-only)"
    )
    lines.append(
        f"  TEST α=1.00 IR {_fmt(test_1.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(test_1.get('unlevered_max_dd'), '+.3f')}  (report-only)"
    )
    lines.append(
        f"  TEST adaptive IR {_fmt(test_ch.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(test_ch.get('unlevered_max_dd'), '+.3f')}  "
        f"meanα {_fmt(test_ch.get('mean_alpha'), '.2f')}  (report-only)"
    )
    lines.append("  TEST (W, rule) grid (report-only):")
    for row in list(test.get("rows") or []):
        lines.append(_row(row))
    return "\n".join(lines)


def _sticky_block(payload: dict[str, Any]) -> str:
    promo = payload.get("sticky_promotion") or {}
    cmp_ = payload.get("sticky_compare") or {}
    fit = payload.get("sticky_fit") or cmp_.get("train_grid") or {}
    val = cmp_.get("val_grid") or {}
    test = cmp_.get("test_grid") or {}
    chosen = cmp_.get("val_chosen") or promo.get("chosen") or {}
    base = cmp_.get("val_baseline") or promo.get("baseline") or {}
    test_ch = cmp_.get("test_chosen") or {}
    test_a = cmp_.get("test_baseline") or {}
    train_ch = fit.get("chosen") or {}

    def _row(r: dict[str, Any]) -> str:
        return (
            f"  {str(r.get('name') or ''):18} "
            f"IR {_fmt(r.get('unlevered_net_ir'), '+.3f')}  "
            f"maxDD {_fmt(r.get('unlevered_max_dd'), '+.3f')}  "
            f"churn {_fmt(r.get('mean_name_churn'), '.3f')}  "
            f"n_held {_fmt(r.get('mean_n_held'), '.1f')}  "
            f"cover {_fmt(100.0 * _as_float(r.get('coverage')), '.0f')}%"
        )

    lines = [
        f"PROMOTE STICKY HYSTERESIS? "
        f"{'YES' if promo.get('promote_sticky') else 'NO'}",
        f"  {fit.get('note') or val.get('note')}",
        f"  TRAIN chose e={_fmt(100.0 * _as_float(train_ch.get('q_enter')), '.0f')}% "
        f"x={_fmt(100.0 * _as_float(train_ch.get('q_exit')), '.0f')}%  "
        f"IR {_fmt(train_ch.get('unlevered_net_ir'), '+.3f')}  "
        f"churn {_fmt(train_ch.get('mean_name_churn'), '.3f')}  "
        f"(fit_split={fit.get('fit_split')})",
        f"  {promo.get('reason')}",
        f"  VAL q20 rebuild IR {_fmt(base.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(base.get('unlevered_max_dd'), '+.3f')}  "
        f"churn {_fmt(base.get('mean_name_churn'), '.3f')}  "
        f"turn {_fmt(base.get('mean_turnover'), '.3f')}",
        f"  VAL TRAIN-sticky IR {_fmt(chosen.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(chosen.get('unlevered_max_dd'), '+.3f')}  "
        f"churn {_fmt(chosen.get('mean_name_churn'), '.3f')}  "
        f"turn {_fmt(chosen.get('mean_turnover'), '.3f')}  "
        f"cover {_fmt(100.0 * _as_float(chosen.get('coverage')), '.0f')}%",
        "  TRAIN (q_enter, q_exit) grid (fit):",
    ]
    for row in list(fit.get("rows") or []):
        lines.append(_row(row))
    lines.append("  VAL grid (report; q not picked here):")
    for row in list(val.get("rows") or []):
        lines.append(_row(row))
    lines.append(
        f"  TEST q20 IR {_fmt(test_a.get('unlevered_net_ir'), '+.3f')}  "
        f"churn {_fmt(test_a.get('mean_name_churn'), '.3f')}  (report-only)"
    )
    lines.append(
        f"  TEST TRAIN-sticky IR {_fmt(test_ch.get('unlevered_net_ir'), '+.3f')}  "
        f"churn {_fmt(test_ch.get('mean_name_churn'), '.3f')}  (report-only)"
    )
    lines.append("  TEST grid (report-only):")
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


def _fmt_conv_row(label: str, row: dict[str, Any]) -> str:
    cover = _as_float(row.get("coverage"))
    cover_s = f"{100.0 * cover:.1f}%" if np.isfinite(cover) else "nan%"
    return (
        f"  {label:<16}  "
        f"IR {_fmt(row.get('unlevered_net_ir'), '+.3f')}  "
        f"maxDD {_fmt(row.get('unlevered_max_dd'), '+.3f')}  "
        f"to {_fmt(row.get('mean_turnover'), '.3f')}  "
        f"cost {_fmt(row.get('mean_cost_unlev_bp'), '.1f')}bp  "
        f"cover {cover_s}  "
        f"n={_n(row.get('n'))}"
    )


def _conviction_live_block(payload: dict[str, Any]) -> str:
    promo = payload.get("conviction_live_promotion") or {}
    cmp = payload.get("conviction_live") or {}
    if not promo and not cmp:
        return ""
    yes = bool(promo.get("promote_conviction_live"))
    va = cmp.get("val") or {}
    te = cmp.get("test") or {}
    q = _as_float(cmp.get("train_q"), default=0.80)
    abs_q = _as_float(cmp.get("train_abs_q"), default=0.0)
    abs_tau = _as_float(cmp.get("train_abs_tau"), default=0.0)
    return "\n".join(
        [
            f"PROMOTE CONVICTION LIVE? {'YES' if yes else 'NO'}",
            "  IDEA F: --live-costs --long-only unlev net IR / max DD / turnover "
            "for E's TRAIN sleeve vs q20 equal. Optional live path only. "
            "Default CLI stays q20. TEST report-only.",
            f"  TRAIN sleeve q={q:.2f} abs_q={abs_q:.2f} |pred|>={abs_tau:.5f}  "
            f"(fit_split=train)",
            "  VAL (gate):",
            _fmt_conv_row("q20 equal", va.get("q20") or {}),
            _fmt_conv_row("E chosen", va.get("chosen") or {}),
            _fmt_conv_row("q90 no |pred|", va.get("q90") or {}),
            f"  VAL IR delta {_fmt(promo.get('val_ir_delta'), '+.3f')} "
            f"(need ≥+{LO_IR_LIFT:.2f})  DD delta {_fmt(promo.get('val_dd_delta'), '+.3f')} "
            f"(need ≥-{LO_DD_TOL:.2f})  cover {_fmt(100.0 * _as_float(promo.get('coverage')), '.1f')}% "
            f"(need ≥{100.0 * BOOK_ALIGN_COVER:.0f}%)",
            "  TEST (report-only):",
            _fmt_conv_row("q20 equal", te.get("q20") or {}),
            _fmt_conv_row("E chosen", te.get("chosen") or {}),
            _fmt_conv_row("q90 no |pred|", te.get("q90") or {}),
            f"  {promo.get('reason') or 'no decision'}",
        ]
    )


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
