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
    OVERNIGHT_FORMULA,
    PAPER_BUNDLE,
    formula_log_line,
)

# Sleeve excess is printed in percentage points. DIR_LIFT is a fraction.
SHORT_EXCESS_LIFT_PP = 100.0 * float(DIR_LIFT)  # 0.2 pp
# Unlevered net IR: LS must beat long-only by this much on VAL.
IR_LIFT = 0.05

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
# long-only live (no locate, borrow=0)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \\
  --holding overnight --live-costs --long-only
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
            stats = book_pnl(
                pred,
                y,
                quantile=0.2,
                holding="overnight",
                hold_halflife=0.0,
                vol_target=float(vol_target),
                causal_vol=True,
                lever_cap=3.0,
                min_names=int(min_names),
                long_only=bool(long_only),
                overnight_r=r_on,
                turnover_z=tz,
                vol_level=vol,
                **_bundle_costs(bundle),
            )
            books[name] = scalar_book(stats)
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
        "promotion": promo,
        "desktop_commands": DESKTOP_COMMANDS,
    }


def format_shorting_report(payload: dict[str, Any]) -> str:
    promo = payload.get("promotion") or {}
    lines = [
        "OVERNIGHT LONG-SHORT vs LONG-ONLY (VAL gate, TEST report-only)",
        f"  recipe: {payload.get('recipe')}  {payload.get('levers')}",
        f"  calendar: {payload.get('calendar_cuts')}",
        f"  train CS IC={_fmt(payload.get('train_cs_ic'), '+.4f')}",
        "",
        _split_block("LOCKED VAL (gate)", payload.get("val") or {}),
        "",
        f"PROMOTE? {'YES' if promo.get('promote_ls') else 'NO'}",
        f"  default live book = {promo.get('default_book')}",
        f"  {promo.get('reason')}",
        "",
        _split_block("LOCKED TEST (report-only; not a gate)", payload.get("test") or {}),
        "",
        "DESKTOP (Yahoo liquid tape — cloud VM has none):",
        DESKTOP_COMMANDS.rstrip(),
    ]
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
