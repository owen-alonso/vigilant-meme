"""VAL-selected overnight-up rank sleeve (IDEA N).

Liquid TRAIN-greedy (logit2 q=0.95) hit 62% on TRAIN and 57.5% on VAL --
a cliff. This module enumerates rank sleeves on TRAIN (cover >= 5%,
excess >= 0), then **selects on locked VAL**. TEST never enters the pick.
Live q20 is unchanged. ASCII / cp1252-safe prints only.

Kinds rank the overnight-up object directly (not a bigger Mamba / not an
alpha-blend of a residual encoder): pred / pred_r / pred_pos_gap /
pred_r_pos / regularized P(up) / CS z-blend / vote-z ensemble /
P(up|pred,pred_r,|pred|,turnover) / gap*turnover. Optional high-dispersion
date filter and k-of-M sleeve vote. VAL selects; TEST never picks.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

MIN_COVER = 0.05
TARGET_UP_PCT = 60.0
VS_E_LIFT_PP = 0.50
# Drop q=0.95 / abs_q=0.85 -- those were the liquid TRAIN 62% -> VAL 57% cliff.
# Keep a middle q=0.85 / abs_q=0.60 so E-style residual can tighten if VAL holds.
UP_QS = (0.70, 0.80, 0.85, 0.90, 0.93)
UP_ABS_QS = (0.0, 0.50, 0.60, 0.70, 0.80)
BLEND_ALPHAS = (0.0, 0.50, 1.0)
LOGIT2_RIDGE = 1.0
LOGIT4_RIDGE = 1.0
GAP_TURN_W = 0.25
DISP_Q = 0.50
VOTE_KS = (2, 3)
VOTE_MEMBER_KINDS = ("pred", "pred_r", "pred_pos_gap", "logit2")
DISP_KINDS = ("pred_pos_gap", "pred_r", "vote_z", "logit4")
RANK_KINDS = (
    "pred",
    "pred_r",
    "pred_pos_gap",
    "pred_r_pos",
    "logit2",
    "cs_blend",
    "vote_z",
    "logit4",
    "gap_turn",
    "vote_k",
)


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _cs_z(x: np.ndarray, dates: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape[0], dtype=np.float64)
    for key in np.unique(dates):
        sel = dates == key
        row = x[sel]
        ok = np.isfinite(row)
        if int(ok.sum()) < 3:
            continue
        mu = float(row[ok].mean())
        sd = float(row[ok].std())
        z = np.zeros(int(sel.sum()), dtype=np.float64)
        if sd > 1e-12:
            z[ok] = (row[ok] - mu) / sd
        out[sel] = z
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zc))


def fit_logit2_up(
    pred: np.ndarray,
    pred_r: np.ndarray,
    r_on: np.ndarray,
    *,
    n_iter: int = 20,
    ridge: float = LOGIT2_RIDGE,
) -> dict[str, float]:
    """TRAIN-only P(up | pred, pred_r). VAL/TEST never enter.

    Light ridge on the two slopes (not the bias) so a TRAIN 62% logit2
    spike is less likely. VAL still has to select the spec.
    """
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[ok], b[ok], y[ok]
    fallback = {"a_pred": 0.0, "a_pr": 0.0, "bias": 0.0, "ridge": float(ridge)}
    if a.size < 64:
        return fallback
    yb = (y > 0.0).astype(np.float64)
    ap, ar, bias = 0.0, 0.0, float(
        np.log(max(yb.mean(), 1e-3) / max(1.0 - yb.mean(), 1e-3))
    )
    ones = np.ones_like(a)
    pen = np.diag([float(ridge), float(ridge), 0.0])
    for _ in range(int(n_iter)):
        lin = ap * a + ar * b + bias
        pr = _sigmoid(lin)
        w = np.clip(pr * (1.0 - pr), 1e-6, None)
        z = lin + (yb - pr) / w
        sw = np.sqrt(w)
        design = np.column_stack([a * sw, b * sw, ones * sw])
        xtx = design.T @ design
        xty = design.T @ (z * sw)
        try:
            coef = np.linalg.solve(xtx + pen, xty)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(design, z * sw, rcond=None)
        ap, ar, bias = float(coef[0]), float(coef[1]), float(coef[2])
    return {"a_pred": ap, "a_pr": ar, "bias": bias, "ridge": float(ridge)}


def apply_logit2_up(
    pred: np.ndarray,
    pred_r: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    return _sigmoid(
        float(spec.get("a_pred") or 0.0) * np.asarray(pred, dtype=np.float64)
        + float(spec.get("a_pr") or 0.0) * np.asarray(pred_r, dtype=np.float64)
        + float(spec.get("bias") or 0.0)
    )


def fit_logit4_up(
    pred: np.ndarray,
    pred_r: np.ndarray,
    r_on: np.ndarray,
    turnover: np.ndarray | None = None,
    *,
    n_iter: int = 20,
    ridge: float = LOGIT4_RIDGE,
) -> dict[str, float]:
    """TRAIN-only P(up | pred, pred_r, |pred|, turnover_z). VAL/TEST never enter."""
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(pred_r, dtype=np.float64)
    y = np.asarray(r_on, dtype=np.float64)
    t = (
        np.asarray(turnover, dtype=np.float64)
        if turnover is not None
        else np.zeros_like(a)
    )
    if t.shape[0] != a.shape[0]:
        t = np.zeros_like(a)
    t = np.where(np.isfinite(t), t, 0.0)
    abs_a = np.abs(a)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y, t, abs_a = a[ok], b[ok], y[ok], t[ok], abs_a[ok]
    fallback = {
        "a_pred": 0.0,
        "a_pr": 0.0,
        "a_abs": 0.0,
        "a_turn": 0.0,
        "bias": 0.0,
        "ridge": float(ridge),
    }
    if a.size < 64:
        return fallback
    yb = (y > 0.0).astype(np.float64)
    ap = ar = aa = at = 0.0
    bias = float(np.log(max(yb.mean(), 1e-3) / max(1.0 - yb.mean(), 1e-3)))
    ones = np.ones_like(a)
    pen = np.diag([float(ridge)] * 4 + [0.0])
    for _ in range(int(n_iter)):
        lin = ap * a + ar * b + aa * abs_a + at * t + bias
        pr = _sigmoid(lin)
        w = np.clip(pr * (1.0 - pr), 1e-6, None)
        z = lin + (yb - pr) / w
        sw = np.sqrt(w)
        design = np.column_stack([a * sw, b * sw, abs_a * sw, t * sw, ones * sw])
        xtx = design.T @ design
        xty = design.T @ (z * sw)
        try:
            coef = np.linalg.solve(xtx + pen, xty)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(design, z * sw, rcond=None)
        ap, ar, aa, at, bias = (float(coef[i]) for i in range(5))
    return {
        "a_pred": ap,
        "a_pr": ar,
        "a_abs": aa,
        "a_turn": at,
        "bias": bias,
        "ridge": float(ridge),
    }


def apply_logit4_up(
    pred: np.ndarray,
    pred_r: np.ndarray,
    turnover: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(pred_r, dtype=np.float64)
    t = np.asarray(turnover, dtype=np.float64)
    if t.shape[0] != a.shape[0]:
        t = np.zeros_like(a)
    t = np.where(np.isfinite(t), t, 0.0)
    return _sigmoid(
        float(spec.get("a_pred") or 0.0) * a
        + float(spec.get("a_pr") or 0.0) * b
        + float(spec.get("a_abs") or 0.0) * np.abs(a)
        + float(spec.get("a_turn") or 0.0) * t
        + float(spec.get("bias") or 0.0)
    )


def date_disp_pred_r(df: pd.DataFrame) -> dict[int, float]:
    """Per-date CS std of pred_r (causal; next open unused)."""
    out: dict[int, float] = {}
    if df.empty or "pred_r" not in df.columns or "date" not in df.columns:
        return out
    dates = df["date"].to_numpy(dtype=np.int64)
    pr = df["pred_r"].to_numpy(dtype=np.float64)
    for key in np.unique(dates):
        row = pr[dates == key]
        ok = np.isfinite(row)
        if int(ok.sum()) < 3:
            continue
        out[int(key)] = float(row[ok].std())
    return out


def high_disp_threshold(df: pd.DataFrame, q: float = DISP_Q) -> float:
    """TRAIN-only quantile of daily CS pred_r std."""
    vals = np.array(list(date_disp_pred_r(df).values()), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        return 0.0
    return float(np.quantile(vals, float(q)))


def _apply_disp_filter(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    if df.empty or float(tau) <= 0.0 or "date" not in df.columns:
        return df
    disp = date_disp_pred_r(df)
    keep = {d for d, v in disp.items() if v >= float(tau)}
    if not keep:
        return df.iloc[0:0].copy()
    return df.loc[df["date"].isin(keep)].copy()


def _turnover(df: pd.DataFrame) -> np.ndarray:
    if "turnover_z" in df.columns:
        t = df["turnover_z"].to_numpy(dtype=np.float64)
        return np.where(np.isfinite(t), t, 0.0)
    return np.zeros(len(df), dtype=np.float64)


def rank_score(
    df: pd.DataFrame,
    kind: str,
    *,
    blend_alpha: float = 0.5,
    logit2: Mapping[str, Any] | None = None,
    logit4: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Causal rank score. Next open is never used."""
    pred = df["pred"].to_numpy(dtype=np.float64) if "pred" in df.columns else np.array([])
    pred_r = (
        df["pred_r"].to_numpy(dtype=np.float64) if "pred_r" in df.columns else pred
    )
    dates = df["date"].to_numpy(dtype=np.int64) if "date" in df.columns else np.zeros(len(df), dtype=np.int64)
    if kind == "pred":
        return pred
    if kind == "pred_r":
        return pred_r
    if kind == "pred_pos_gap":
        out = pred.astype(np.float64, copy=True)
        out[~(np.isfinite(pred_r) & (pred_r > 0.0))] = np.nan
        return out
    if kind == "pred_r_pos":
        out = pred_r.astype(np.float64, copy=True)
        out[~(np.isfinite(pred_r) & (pred_r > 0.0))] = np.nan
        return out
    if kind == "logit2":
        return apply_logit2_up(pred, pred_r, logit2 or {})
    if kind == "logit4":
        return apply_logit4_up(pred, pred_r, _turnover(df), logit4 or {})
    if kind == "cs_blend":
        a = float(blend_alpha)
        return a * _cs_z(pred, dates) + (1.0 - a) * _cs_z(pred_r, dates)
    if kind == "vote_z":
        z1 = _cs_z(pred, dates)
        z2 = _cs_z(pred_r, dates)
        z3 = _cs_z(apply_logit2_up(pred, pred_r, logit2 or {}), dates)
        return (z1 + z2 + z3) / 3.0
    if kind == "gap_turn":
        gap = pred.astype(np.float64, copy=True)
        gap[~(np.isfinite(pred_r) & (pred_r > 0.0))] = np.nan
        return _cs_z(gap, dates) + float(GAP_TURN_W) * _cs_z(_turnover(df), dates)
    raise ValueError(f"unknown rank kind {kind!r}")


def high_drift_weekdays(df: pd.DataFrame) -> list[int]:
    """TRAIN weekdays whose overnight-up is at or above the TRAIN pool."""
    if df.empty or "r_on" not in df.columns or "weekday" not in df.columns:
        return list(range(5))
    r = df["r_on"].to_numpy(dtype=np.float64)
    ok = np.isfinite(r) & (r != 0.0)
    if int(ok.sum()) < 20:
        return list(range(5))
    pool = float((r[ok] > 0.0).mean())
    keep: list[int] = []
    wd = df["weekday"].to_numpy(dtype=np.int64)
    for d in range(5):
        sel = ok & (wd == d)
        if int(sel.sum()) < 8:
            continue
        if float((r[sel] > 0.0).mean()) + 1e-15 >= pool:
            keep.append(int(d))
    return keep or list(range(5))


def _apply_dow_filter(df: pd.DataFrame, keep: list[int] | None) -> pd.DataFrame:
    if not keep or df.empty or "weekday" not in df.columns:
        return df
    return df.loc[df["weekday"].isin(keep)].copy()


def score_boolean_sleeve(
    df: pd.DataFrame,
    mask: np.ndarray,
    *,
    n_full: int,
) -> dict[str, Any]:
    """Overnight-up of a precomputed causal mask vs the full frame."""
    empty = {
        "up_pct": float("nan"),
        "uncond_up_pct": float("nan"),
        "excess_pp": float("nan"),
        "n": 0.0,
        "n_dates": 0.0,
        "coverage": float("nan"),
        "cover_full": float("nan"),
        "q": float("nan"),
        "abs_tau": 0.0,
    }
    if df.empty or "r_on" not in df.columns:
        return empty
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] != len(df):
        return empty
    r = df["r_on"].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy(dtype=np.int64) if "date" in df.columns else np.zeros(len(df), dtype=np.int64)
    moved = m & np.isfinite(r) & (r != 0.0)
    uncond = r[np.isfinite(r) & (r != 0.0)]
    uncond_up = float((uncond > 0).mean()) if uncond.size else float("nan")
    full = float(n_full if n_full else len(df))
    cover = float(m.sum() / full) if full > 0 else float("nan")
    empty["n"] = float(int(moved.sum()))
    empty["n_dates"] = float(pd.Series(dates[m]).nunique()) if int(m.sum()) else 0.0
    empty["coverage"] = cover
    empty["cover_full"] = cover
    empty["uncond_up_pct"] = (
        float(100.0 * uncond_up) if np.isfinite(uncond_up) else float("nan")
    )
    if int(moved.sum()) == 0:
        return empty
    up = float((r[moved] > 0).mean())
    return {
        **empty,
        "up_pct": float(100.0 * up),
        "excess_pp": (
            float(100.0 * (up - uncond_up)) if np.isfinite(uncond_up) else float("nan")
        ),
    }


def spec_mask(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    min_names: int,
    logit2: Mapping[str, Any] | None = None,
    logit4: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Boolean sleeve of a non-vote spec on ``df`` (no row drop)."""
    from forecast.accuracy import cs_top_abs_mask

    ranked = df.copy()
    ranked["pred"] = rank_score(
        ranked,
        str(spec.get("kind") or "pred"),
        blend_alpha=_as_float(spec.get("blend_alpha"), 0.5),
        logit2=logit2,
        logit4=logit4,
    )
    return cs_top_abs_mask(
        ranked,
        q=_as_float(spec.get("q"), 0.80),
        abs_tau=_as_float(spec.get("abs_tau"), 0.0),
        min_names=min_names,
    )


def vote_mask(
    df: pd.DataFrame,
    members: list[Mapping[str, Any]],
    k: int,
    *,
    min_names: int,
    logit2: Mapping[str, Any] | None = None,
    logit4: Mapping[str, Any] | None = None,
) -> np.ndarray:
    votes = np.zeros(len(df), dtype=np.int64)
    for spec in members:
        votes = votes + spec_mask(
            df, spec, min_names=min_names, logit2=logit2, logit4=logit4
        ).astype(np.int64)
    return votes >= int(k)


def train_kind_winners(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One TRAIN-best spec per complementary kind (cover >= 5%, excess >= 0)."""
    best: dict[str, dict[str, Any]] = {}
    best_key: dict[str, tuple[float, ...]] = {}
    for row in rows:
        kind = str(row.get("kind") or "")
        if kind not in VOTE_MEMBER_KINDS:
            continue
        if str(row.get("dow_mode") or "all") != "all":
            continue
        if str(row.get("disp_mode") or "all") != "all":
            continue
        cover = _as_float(row.get("cover_full", row.get("coverage")))
        xs = _as_float(row.get("excess_pp"))
        up = _as_float(row.get("up_pct"))
        if not (math.isfinite(cover) and cover >= MIN_COVER):
            continue
        if not (math.isfinite(xs) and xs >= 0.0 and math.isfinite(up)):
            continue
        key = (xs, up, cover)
        if kind not in best_key or key > best_key[kind]:
            best_key[kind] = key
            best[kind] = dict(row)
    return [best[k] for k in VOTE_MEMBER_KINDS if k in best]


def fit_overnight_up_rank_on_train(
    df: pd.DataFrame,
    *,
    min_names: int,
) -> dict[str, Any]:
    """TRAIN-only rank kind + (q, |score| floor) + optional high-drift DOW filter."""
    from forecast.accuracy import score_book_aligned_sleeve

    empty = {
        "rows": [],
        "chosen": {},
        "fit_split": "train",
        "logit2": {},
        "logit4": {},
        "disp_tau": 0.0,
        "high_drift_dows": list(range(5)),
        "qs": list(UP_QS),
        "abs_qs": list(UP_ABS_QS),
        "note": (
            "TRAIN-enumerate overnight-up rank sleeves (cover >= 5%). "
            "Kinds: residual, pred_r, gap filters, ridged P(up), CS z-blend, "
            "vote-z, logit4, gap*turnover. Optional high-drift weekday / "
            "high-dispersion date filter and k-of-M sleeve vote. "
            "VAL selects; TEST never picks."
        ),
    }
    if df.empty or "pred" not in df.columns or "r_on" not in df.columns:
        return empty
    logit2 = fit_logit2_up(
        df["pred"].to_numpy(dtype=np.float64),
        df["pred_r"].to_numpy(dtype=np.float64),
        df["r_on"].to_numpy(dtype=np.float64),
    )
    logit4 = fit_logit4_up(
        df["pred"].to_numpy(dtype=np.float64),
        df["pred_r"].to_numpy(dtype=np.float64),
        df["r_on"].to_numpy(dtype=np.float64),
        _turnover(df),
    )
    dows = high_drift_weekdays(df)
    disp_tau = high_disp_threshold(df)
    kinds: list[tuple[str, float]] = [
        ("pred", 0.5),
        ("pred_r", 0.5),
        ("pred_pos_gap", 0.5),
        ("pred_r_pos", 0.5),
        ("logit2", 0.5),
        ("logit4", 0.5),
        ("vote_z", 0.5),
        ("gap_turn", 0.5),
    ]
    kinds.extend(("cs_blend", float(a)) for a in BLEND_ALPHAS)
    rows: list[dict[str, Any]] = []
    n_full = float(len(df))

    def _emit(
        sub: pd.DataFrame,
        kind: str,
        alpha: float,
        *,
        dow_mode: str,
        disp_mode: str,
        keep_dows: list[int] | None,
    ) -> None:
        score = rank_score(
            sub, kind, blend_alpha=alpha, logit2=logit2, logit4=logit4
        )
        ranked = sub.copy()
        ranked["pred"] = score
        mag = np.abs(score)
        mag = mag[np.isfinite(mag)]
        frac = float(len(sub) / n_full) if n_full > 0 else 1.0
        scale_cover = dow_mode != "all" or disp_mode != "all"
        for q in UP_QS:
            for aq in UP_ABS_QS:
                tau = (
                    0.0
                    if float(aq) <= 0.0
                    else (float(np.quantile(mag, float(aq))) if mag.size else 0.0)
                )
                sleeve = score_book_aligned_sleeve(
                    ranked, q=float(q), abs_tau=tau, min_names=min_names
                )
                cover_sub = _as_float(sleeve.get("coverage"))
                cover_full = (
                    cover_sub
                    if not scale_cover or not np.isfinite(cover_sub)
                    else float(cover_sub * frac)
                )
                if np.isfinite(cover_full) and cover_full < MIN_COVER:
                    continue
                if not np.isfinite(_as_float(sleeve.get("up_pct"))):
                    continue
                rows.append(
                    {
                        **sleeve,
                        "kind": kind,
                        "blend_alpha": float(alpha),
                        "dow_mode": dow_mode,
                        "disp_mode": disp_mode,
                        "disp_tau": float(disp_tau),
                        "dows": list(keep_dows) if keep_dows is not None else list(range(5)),
                        "abs_q": float(aq),
                        "cover_full": cover_full,
                        "coverage": cover_full,
                    }
                )

    for dow_mode, keep in (("all", None), ("high_drift", dows)):
        sub = _apply_dow_filter(df, keep)
        if len(sub) < max(20, int(min_names) * 3):
            continue
        for kind, alpha in kinds:
            _emit(
                sub,
                kind,
                alpha,
                dow_mode=dow_mode,
                disp_mode="all",
                keep_dows=keep,
            )
    # High-dispersion dates only, weekday=all, liquid-relevant kinds.
    disp_sub = _apply_disp_filter(df, disp_tau)
    if len(disp_sub) >= max(20, int(min_names) * 3):
        for kind in DISP_KINDS:
            _emit(
                disp_sub,
                kind,
                0.5,
                dow_mode="all",
                disp_mode="high",
                keep_dows=None,
            )
    members = train_kind_winners(rows)
    if len(members) >= 2:
        for k in VOTE_KS:
            mask = vote_mask(
                df, members, int(k), min_names=min_names, logit2=logit2, logit4=logit4
            )
            scored = score_boolean_sleeve(df, mask, n_full=int(n_full))
            cover = _as_float(scored.get("cover_full", scored.get("coverage")))
            if not (math.isfinite(cover) and cover >= MIN_COVER):
                continue
            if not np.isfinite(_as_float(scored.get("up_pct"))):
                continue
            rows.append(
                {
                    **scored,
                    "kind": "vote_k",
                    "blend_alpha": 0.5,
                    "dow_mode": "all",
                    "disp_mode": "all",
                    "disp_tau": float(disp_tau),
                    "dows": list(range(5)),
                    "abs_q": 0.0,
                    "abs_tau": 0.0,
                    "q": float(k),
                    "members": [dict(m) for m in members],
                    "n_members": float(len(members)),
                    "cover_full": cover,
                    "coverage": cover,
                }
            )
    chosen: dict[str, Any] = {}
    best_key = (-1e18, -1e18, -1e18)
    for row in rows:
        xs = _as_float(row.get("excess_pp"))
        up = _as_float(row.get("up_pct"))
        cover = _as_float(row.get("cover_full", row.get("coverage")))
        if not np.isfinite(xs) or not np.isfinite(up):
            continue
        key = (xs, up, cover)
        if key > best_key:
            best_key = key
            chosen = dict(row)
    if not chosen and rows:
        chosen = dict(rows[0])
    return {
        **empty,
        "rows": rows,
        "chosen": chosen,
        "logit2": logit2,
        "logit4": logit4,
        "disp_tau": float(disp_tau),
        "vote_members": [dict(m) for m in members],
        "high_drift_dows": dows,
        "n_candidates": float(len(rows)),
    }


def apply_up_rank_spec(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    logit2: Mapping[str, Any] | None = None,
    logit4: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply a TRAIN-frozen rank spec. Does not refit."""
    kind = str(spec.get("kind") or "pred")
    alpha = _as_float(spec.get("blend_alpha"), 0.5)
    ranked = df.copy()
    if kind == "vote_k":
        members = list(spec.get("members") or [])
        k = int(_as_float(spec.get("q"), 2.0))
        mask = vote_mask(
            ranked, members, k, min_names=3, logit2=logit2, logit4=logit4
        )
        ranked["pred"] = mask.astype(np.float64)
    else:
        ranked["pred"] = rank_score(
            ranked, kind, blend_alpha=alpha, logit2=logit2, logit4=logit4
        )
    mode = str(spec.get("dow_mode") or "all")
    if mode == "high_drift":
        keep = [int(x) for x in (spec.get("dows") or [])]
        ranked = _apply_dow_filter(ranked, keep)
    if str(spec.get("disp_mode") or "all") == "high":
        ranked = _apply_disp_filter(ranked, _as_float(spec.get("disp_tau"), 0.0))
    return ranked


def score_up_rank_sleeve(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    min_names: int,
    logit2: Mapping[str, Any] | None = None,
    logit4: Mapping[str, Any] | None = None,
    n_full: int | None = None,
) -> dict[str, Any]:
    from forecast.accuracy import score_book_aligned_sleeve

    full = float(n_full if n_full is not None else len(df))
    if str(spec.get("kind") or "") == "vote_k":
        members = list(spec.get("members") or [])
        k = int(_as_float(spec.get("q"), 2.0))
        mask = vote_mask(
            df, members, k, min_names=min_names, logit2=logit2, logit4=logit4
        )
        sleeve = score_boolean_sleeve(df, mask, n_full=int(full))
        return {
            **sleeve,
            "kind": "vote_k",
            "blend_alpha": 0.5,
            "dow_mode": "all",
            "disp_mode": "all",
            "abs_q": 0.0,
            "q": float(k),
            "cover_full": _as_float(sleeve.get("cover_full")),
            "coverage": _as_float(sleeve.get("cover_full")),
        }
    ranked = apply_up_rank_spec(df, spec, logit2=logit2, logit4=logit4)
    sleeve = score_book_aligned_sleeve(
        ranked,
        q=_as_float(spec.get("q"), 0.80),
        abs_tau=_as_float(spec.get("abs_tau"), 0.0),
        min_names=min_names,
    )
    cover_sub = _as_float(sleeve.get("coverage"))
    frac = float(len(ranked) / full) if full > 0 else 1.0
    scale_cover = (
        str(spec.get("dow_mode") or "all") != "all"
        or str(spec.get("disp_mode") or "all") != "all"
    )
    cover_full = (
        cover_sub
        if not scale_cover or not np.isfinite(cover_sub)
        else float(cover_sub * frac)
    )
    return {
        **sleeve,
        "kind": spec.get("kind"),
        "blend_alpha": _as_float(spec.get("blend_alpha"), 0.5),
        "dow_mode": spec.get("dow_mode"),
        "disp_mode": spec.get("disp_mode") or "all",
        "abs_q": _as_float(spec.get("abs_q"), 0.0),
        "cover_full": cover_full,
        "coverage": cover_full,
    }


def spec_id(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identity of a frozen TRAIN spec. abs_tau is part of the sleeve, not the id."""
    return (
        str(row.get("kind") or "pred"),
        round(_as_float(row.get("blend_alpha"), 0.5), 4),
        round(_as_float(row.get("q"), 0.80), 4),
        round(_as_float(row.get("abs_q"), 0.0), 4),
        str(row.get("dow_mode") or "all"),
        str(row.get("disp_mode") or "all"),
    )


def pick_up_rank_on_val(
    train_rows: list[Mapping[str, Any]],
    val_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """VAL selects among TRAIN-valid specs. TEST never enters.

    TRAIN filter: cover >= 5% and excess >= 0.
    VAL filter: cover >= 5%.
    Among VAL >= 60%: prefer higher overnight-up, then more cover.
    Among misses: prefer higher VAL up, then more cover.
    Cliff |TRAIN-VAL| is a weak tiebreak only. Cover floor stays 5%.
    """
    train_by = {spec_id(r): dict(r) for r in train_rows}
    best: dict[str, Any] = {}
    best_key: tuple[float, ...] | None = None
    n_val_ok = 0
    n_hit_60 = 0
    best_val_up = float("-inf")
    for vr in val_rows:
        tr = train_by.get(spec_id(vr))
        if tr is None:
            continue
        train_cover = _as_float(tr.get("cover_full", tr.get("coverage")))
        train_xs = _as_float(tr.get("excess_pp"))
        if not (math.isfinite(train_cover) and train_cover >= MIN_COVER):
            continue
        if not (math.isfinite(train_xs) and train_xs >= 0.0):
            continue
        up = _as_float(vr.get("up_pct"))
        cover = _as_float(vr.get("cover_full", vr.get("coverage")))
        if not math.isfinite(up) or not math.isfinite(cover) or cover < MIN_COVER:
            continue
        n_val_ok += 1
        hit = bool(up >= TARGET_UP_PCT)
        if hit:
            n_hit_60 += 1
        if up > best_val_up:
            best_val_up = up
        train_up = _as_float(tr.get("up_pct"))
        cliff = (
            float(train_up - up)
            if math.isfinite(train_up) and math.isfinite(up)
            else 0.0
        )
        if hit:
            key = (1.0, up, cover, -abs(cliff))
        else:
            key = (0.0, up, cover, -abs(cliff))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                **tr,
                "val_up_pct": up,
                "val_cover_full": cover,
                "val_excess_pp": _as_float(vr.get("excess_pp")),
                "train_up_pct": train_up,
                "train_cover_full": train_cover,
                "cliff_pp": cliff,
            }
    return {
        "chosen": best,
        "n_val_ok": float(n_val_ok),
        "n_hit_60": float(n_hit_60),
        "best_val_up": (
            float(best_val_up) if math.isfinite(best_val_up) else float("nan")
        ),
        "select_split": "val",
        "gated_on": "val",
    }


def score_train_specs_on_split(
    df: pd.DataFrame,
    specs: list[Mapping[str, Any]],
    *,
    min_names: int,
    logit2: Mapping[str, Any] | None,
    n_full: int,
    logit4: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Score TRAIN-frozen specs on a split. Does not refit tau / logit / DOW."""
    from forecast.accuracy import score_book_aligned_sleeve

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    full = float(n_full if n_full else len(df))
    for spec in specs:
        if str(spec.get("kind") or "") == "vote_k":
            scored = score_up_rank_sleeve(
                df,
                spec,
                min_names=min_names,
                logit2=logit2,
                logit4=logit4,
                n_full=int(full),
            )
            out.append({**dict(spec), **scored, "kind": "vote_k"})
            continue
        key = (
            str(spec.get("kind") or "pred"),
            round(_as_float(spec.get("blend_alpha"), 0.5), 4),
            str(spec.get("dow_mode") or "all"),
            str(spec.get("disp_mode") or "all"),
        )
        groups.setdefault(key, []).append(spec)
    for (kind, alpha, dow, disp), bunch in groups.items():
        template = {
            "kind": kind,
            "blend_alpha": alpha,
            "dow_mode": dow,
            "disp_mode": disp,
            "disp_tau": (bunch[0].get("disp_tau") if bunch else 0.0),
            "dows": (bunch[0].get("dows") if bunch else None),
        }
        ranked = apply_up_rank_spec(
            df, template, logit2=logit2, logit4=logit4
        )
        frac = float(len(ranked) / full) if full > 0 else 1.0
        scale_cover = dow != "all" or disp != "all"
        for spec in bunch:
            sleeve = score_book_aligned_sleeve(
                ranked,
                q=_as_float(spec.get("q"), 0.80),
                abs_tau=_as_float(spec.get("abs_tau"), 0.0),
                min_names=min_names,
            )
            cover_sub = _as_float(sleeve.get("coverage"))
            cover_full = (
                cover_sub
                if not scale_cover or not np.isfinite(cover_sub)
                else float(cover_sub * frac)
            )
            out.append(
                {
                    **dict(spec),
                    **sleeve,
                    "kind": kind,
                    "blend_alpha": alpha,
                    "dow_mode": dow,
                    "disp_mode": disp,
                    "abs_q": _as_float(spec.get("abs_q"), 0.0),
                    "abs_tau": _as_float(spec.get("abs_tau"), 0.0),
                    "cover_full": cover_full,
                    "coverage": cover_full,
                }
            )
    return out


def decide_overnight_up_rank_promote(
    *,
    val_chosen: Mapping[str, Any],
    val_e: Mapping[str, Any],
    chosen: Mapping[str, Any],
) -> dict[str, Any]:
    """VAL-only. TEST never enters. Live q20 unchanged."""
    up = _as_float(val_chosen.get("up_pct"))
    floor = _as_float(val_chosen.get("uncond_up_pct"))
    cover = _as_float(val_chosen.get("cover_full", val_chosen.get("coverage")))
    e_up = _as_float(val_e.get("up_pct"))
    xs = _as_float(val_chosen.get("excess_pp"))
    if not np.isfinite(xs) and np.isfinite(up) and np.isfinite(floor):
        xs = float(up - floor)
    cover_ok = bool(math.isfinite(cover) and cover >= MIN_COVER)
    hit_60 = bool(cover_ok and math.isfinite(up) and up >= TARGET_UP_PCT)
    vs_e = bool(
        cover_ok
        and math.isfinite(up)
        and math.isfinite(e_up)
        and up >= e_up + VS_E_LIFT_PP
    )
    promote = bool(hit_60 or vs_e)
    kind = str(chosen.get("kind") or "pred")
    q = _as_float(chosen.get("q"), 0.80)
    if not cover_ok:
        reason = (
            f"NO PROMOTE overnight-up rank: VAL cover {100.0 * cover:.1f}% "
            f"< {100.0 * MIN_COVER:.0f}% (do not shrink to a handful of names)."
        )
    elif hit_60:
        reason = (
            f"PROMOTE overnight-up rank {kind} q={q:.2f}: VAL up {up:.2f}% "
            f">= {TARGET_UP_PCT:.1f}% with cover {100.0 * cover:.1f}% >= "
            f"{100.0 * MIN_COVER:.0f}%. vs E {e_up:.2f}%. TEST report-only. "
            "Live q20 unchanged."
        )
    elif vs_e:
        reason = (
            f"PROMOTE overnight-up rank {kind} q={q:.2f}: VAL up {up:.2f}% vs E "
            f"{e_up:.2f}% (lift {up - e_up:+.2f} pp) cover {100.0 * cover:.1f}%. "
            f"Missed {TARGET_UP_PCT:.1f}% target. TEST report-only. "
            "Live q20 unchanged."
        )
    else:
        reason = (
            f"NO PROMOTE overnight-up rank {kind} q={q:.2f}: VAL up {up:.2f}% "
            f"(need {TARGET_UP_PCT:.1f}% or E {e_up:.2f}% +{VS_E_LIFT_PP:.1f} pp) "
            f"cover {100.0 * cover:.1f}%. TEST does not gate."
        )
    return {
        "promote_up_rank": promote,
        "reached_60": hit_60,
        "gated_on": "val",
        "reason": reason,
        "val_up": up,
        "val_e_up": e_up,
        "val_floor": floor,
        "val_excess_pp": xs,
        "val_vs_e_pp": (
            float(up - e_up) if math.isfinite(up) and math.isfinite(e_up) else float("nan")
        ),
        "coverage": cover,
        "min_cover": MIN_COVER,
        "target_up_pct": TARGET_UP_PCT,
        "kind": kind,
        "q": q,
        "abs_q": _as_float(chosen.get("abs_q"), 0.0),
        "dow_mode": chosen.get("dow_mode"),
        "live_book_unchanged": True,
    }


def format_overnight_up_rank_block(payload: Mapping[str, Any]) -> str:
    blob = dict(payload.get("overnight_up_rank") or payload or {})
    if not blob:
        return ""
    promo = dict(blob.get("promotion") or {})
    fit = dict(blob.get("fit") or {})
    cmp = dict(blob.get("compare") or {})
    autopsy = dict(blob.get("autopsy") or fit.get("autopsy") or {})
    chosen = dict(fit.get("chosen") or {})
    greedy = dict(fit.get("train_greedy") or {})
    greedy_val = dict(cmp.get("train_greedy_val") or {})
    va = dict(cmp.get("val") or {})
    te = dict(cmp.get("test") or {})
    e_va = dict(cmp.get("val_e") or {})
    e_te = dict(cmp.get("test_e") or {})
    yes = bool(promo.get("promote_up_rank"))
    lines = [
        f"PROMOTE OVERNIGHT-UP RANK? {'YES' if yes else 'NO'}  "
        f"REACHED 60%? {'YES' if bool(promo.get('reached_60')) else 'NO'}",
        "  TRAIN-filter / VAL-select overnight-up rank (not a bigger Mamba, "
        "not a Dynamic A alpha-blend). Kinds: residual / pred_r / gap filters / "
        "ridged P(up) / CS z-blend / vote-z / logit4 / gap*turnover. "
        "Optional high-drift weekday, high-dispersion dates, k-of-M vote. "
        "Cover floor 5% vs the full split. TEST report-only. Live q20 unchanged.",
        f"  VAL pick kind={chosen.get('kind')!r}  "
        f"blend_a={_as_float(chosen.get('blend_alpha')):.2f}  "
        f"q={_as_float(chosen.get('q')):.2f}  "
        f"abs_q={_as_float(chosen.get('abs_q')):.2f}  "
        f"dow={chosen.get('dow_mode')!r}  "
        f"disp={chosen.get('disp_mode') or 'all'!r}  "
        f"select_split={str(fit.get('select_split') or 'val')!r}  "
        f"n_train={int(_as_float(fit.get('n_candidates'), 0.0))}  "
        f"n_val_ok={int(_as_float(autopsy.get('n_val_ok'), 0.0))}  "
        f"n_hit_60={int(_as_float(autopsy.get('n_hit_60'), 0.0))}",
        f"  TRAIN greedy kind={greedy.get('kind')!r}  "
        f"q={_as_float(greedy.get('q')):.2f}  "
        f"abs_q={_as_float(greedy.get('abs_q')):.2f}  "
        f"up {_as_float(greedy.get('up_pct')):.2f}%  "
        f"-> VAL {_as_float(greedy_val.get('up_pct')):.2f}%  "
        f"cliff {_as_float(autopsy.get('train_greedy_cliff_pp')):+.2f}pp  "
        "(not used)",
        f"  VAL   rank  up {_as_float(va.get('up_pct')):.2f}%  "
        f"xs {_as_float(va.get('excess_pp')):+.2f}pp  "
        f"cover {100.0 * _as_float(va.get('cover_full', va.get('coverage'))):.1f}%  "
        f"n={int(_as_float(va.get('n'), 0.0))}  "
        f"best_val_up {_as_float(autopsy.get('best_val_up')):.2f}%",
        f"  VAL   E     up {_as_float(e_va.get('up_pct')):.2f}%  "
        f"cover {100.0 * _as_float(e_va.get('coverage')):.1f}%",
        f"  VAL   vs E {_as_float(promo.get('val_vs_e_pp')):+.2f} pp  "
        f"vs 60% {_as_float(va.get('up_pct')) - TARGET_UP_PCT:+.2f} pp  "
        f"cover_ok={bool(_as_float(promo.get('coverage')) >= MIN_COVER)}",
        f"  TEST  rank  up {_as_float(te.get('up_pct')):.2f}%  "
        f"cover {100.0 * _as_float(te.get('cover_full', te.get('coverage'))):.1f}%  "
        f"(report-only)",
        f"  TEST  E     up {_as_float(e_te.get('up_pct')):.2f}%  (report-only)",
        f"  {promo.get('reason') or 'no decision'}",
    ]
    if int(_as_float(autopsy.get("n_hit_60"), 0.0)) <= 0:
        lines.append(
            f"  STOP 60%: no TRAIN-valid VAL sleeve with cover >= 5% reached "
            f"{TARGET_UP_PCT:.1f}%. best_val_up="
            f"{_as_float(autopsy.get('best_val_up')):.2f}%. "
            "Blocked by skip-rank vs overnight-up (not the picker). "
            "TEST does not retarget. Live q20 unchanged."
        )
    return "\n".join(lines)


def evaluate_overnight_up_rank(
    frames: Mapping[str, pd.DataFrame],
    *,
    min_names: int,
    e_val: Mapping[str, Any] | None = None,
    e_test: Mapping[str, Any] | None = None,
    log_fn: Any | None = None,
) -> dict[str, Any]:
    """Enumerate on TRAIN, select on VAL, report TEST. TEST never picks."""
    tr = frames["train"]
    va = frames["val"]
    te = frames["test"]
    fit = fit_overnight_up_rank_on_train(tr, min_names=min_names)
    train_greedy = dict(fit.get("chosen") or {})
    logit2 = dict(fit.get("logit2") or {})
    logit4 = dict(fit.get("logit4") or {})
    train_rows = [dict(r) for r in (fit.get("rows") or [])]
    val_rows = score_train_specs_on_split(
        va,
        train_rows,
        min_names=min_names,
        logit2=logit2,
        logit4=logit4,
        n_full=len(va),
    )
    picked = pick_up_rank_on_val(train_rows, val_rows)
    spec = dict(picked.get("chosen") or {})
    if not spec:
        spec = dict(train_greedy)
    greedy_val = (
        score_up_rank_sleeve(
            va,
            train_greedy,
            min_names=min_names,
            logit2=logit2,
            logit4=logit4,
            n_full=len(va),
        )
        if train_greedy
        else {}
    )
    val_s = score_up_rank_sleeve(
        va, spec, min_names=min_names, logit2=logit2, logit4=logit4, n_full=len(va)
    )
    test_s = score_up_rank_sleeve(
        te, spec, min_names=min_names, logit2=logit2, logit4=logit4, n_full=len(te)
    )
    train_s = score_up_rank_sleeve(
        tr, spec, min_names=min_names, logit2=logit2, logit4=logit4, n_full=len(tr)
    )
    greedy_train_up = _as_float(train_greedy.get("up_pct"))
    greedy_val_up = _as_float(greedy_val.get("up_pct"))
    autopsy = {
        "n_train": float(len(train_rows)),
        "n_val_ok": picked.get("n_val_ok"),
        "n_hit_60": picked.get("n_hit_60"),
        "best_val_up": picked.get("best_val_up"),
        "train_greedy_kind": train_greedy.get("kind"),
        "train_greedy_q": _as_float(train_greedy.get("q")),
        "train_greedy_abs_q": _as_float(train_greedy.get("abs_q")),
        "train_greedy_train_up": greedy_train_up,
        "train_greedy_val_up": greedy_val_up,
        "train_greedy_cliff_pp": (
            float(greedy_train_up - greedy_val_up)
            if math.isfinite(greedy_train_up) and math.isfinite(greedy_val_up)
            else float("nan")
        ),
        "select_split": "val",
        "test_used": False,
        "stop_60": bool(int(_as_float(picked.get("n_hit_60"), 0.0)) <= 0),
        "chosen_kind": spec.get("kind"),
        "chosen_disp": spec.get("disp_mode") or "all",
    }
    fit = {
        **fit,
        "chosen": spec,
        "train_greedy": train_greedy,
        "select_split": "val",
        "autopsy": autopsy,
        "n_candidates": float(len(train_rows)),
    }
    promo = decide_overnight_up_rank_promote(
        val_chosen=val_s,
        val_e=e_val or {},
        chosen=spec,
    )
    payload = {
        "fit": fit,
        "compare": {
            "train": train_s,
            "val": val_s,
            "test": test_s,
            "val_e": dict(e_val or {}),
            "test_e": dict(e_test or {}),
            "train_greedy": train_greedy,
            "train_greedy_val": greedy_val,
        },
        "autopsy": autopsy,
        "promotion": promo,
        "note": fit.get("note"),
    }
    if log_fn:
        log_fn(format_overnight_up_rank_block({"overnight_up_rank": payload}))
    return payload
