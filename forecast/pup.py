"""(2a) Direct overnight-up P(up) head — classification, not skip-rank.

Fits one logistic P(overnight_r > 0) on last-bar features (PIT-masked).
Ranks names by P(up). The sleeve is a **single** cover-floor rule
(top 5% within date, cover >= 5% of the full frame) — not an overnight-up
q-grid / skip-rank grid (that style is closed).

Locked VAL overnight-up >= 60% promotes. TEST is report-only. The book
gate (live_locate IR vs +5.58) stays a separate report-only number for
this cut. Cost-aware ranking / IR loss is (2b) and does not block (2a).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from forecast.pit import overnight_up_labels
from forecast.ridge import labelled_rows


MIN_COVER = 0.05
# Thin sleeve. Do not reuse CS-ridge min_names (30 on liquid) — that
# forced ~35%+ cover and is not the overnight-up object.
MIN_NAMES = 3
TARGET_UP_PCT = 60.0
# Settled skip-rank VAL (closed grid). Honest autopsy if (2a) misses 60%.
HONEST_SKIP_RANK_VAL_UP = 59.07
PUP_RIDGE = 1.0
# (2b) follows; do not implement a cost-IR loss here.
COST_RANK_LOSS = "deferred_2b"


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zc))


def labelled_pup_rows(
    symbols: Sequence[Any],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized last-bar rows with PIT-masked overnight-up labels.

    ``y`` is 0/1 (``overnight_r > 0``). ``next_split_days==1`` is already
    dropped from ``valid`` / labels. Rows with NaN y are omitted.
    """
    x, _resid, dates = labelled_rows(symbols, feature_mean, feature_std)
    rows_y: list[np.ndarray] = []
    keep_chunks: list[np.ndarray] = []
    for sym in symbols:
        if not bool(getattr(sym, "valid", np.zeros(0, dtype=bool)).any()):
            continue
        y = overnight_up_labels(
            getattr(sym, "overnight_r", None),
            valid=sym.valid,
            next_split_days=getattr(sym, "next_split_days", None),
        )
        take = y[np.asarray(sym.valid, dtype=bool)]
        rows_y.append(np.asarray(take, dtype=np.float64))
        keep_chunks.append(np.isfinite(take))
    if not rows_y or x.size == 0:
        f = int(np.asarray(feature_mean).shape[0])
        return (
            np.zeros((0, f), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    y = np.concatenate(rows_y, axis=0)
    keep = np.concatenate(keep_chunks, axis=0)
    if keep.shape[0] != x.shape[0]:
        n = min(int(keep.shape[0]), int(x.shape[0]))
        x, y, dates, keep = x[:n], y[:n], dates[:n], keep[:n]
    return x[keep], y[keep], dates[keep]


def fit_pup_logistic(
    x: np.ndarray,
    y_up: np.ndarray,
    *,
    ridge: float = PUP_RIDGE,
    n_iter: int = 25,
) -> dict[str, Any]:
    """TRAIN-only logistic P(up | last-bar features). VAL/TEST never enter."""
    a = np.asarray(x, dtype=np.float64)
    y = np.asarray(y_up, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(a).all(axis=1)
    a, y = a[ok], y[ok]
    n_feat = int(a.shape[1]) if a.ndim == 2 and a.size else 0
    fallback = {
        "weights": np.zeros(n_feat, dtype=np.float64),
        "bias": 0.0,
        "ridge": float(ridge),
        "n": int(y.size),
        "train_up_rate": float("nan"),
    }
    if n_feat < 1 or y.size < 32:
        return fallback
    yb = (y > 0.5).astype(np.float64)
    prior = float(np.clip(yb.mean(), 1e-3, 1.0 - 1e-3))
    w = np.zeros(n_feat, dtype=np.float64)
    bias = float(np.log(prior / (1.0 - prior)))
    ones = np.ones(y.size, dtype=np.float64)
    pen = np.zeros((n_feat + 1, n_feat + 1), dtype=np.float64)
    pen[:n_feat, :n_feat] = float(ridge) * np.eye(n_feat)
    for _ in range(int(n_iter)):
        lin = a @ w + bias
        pr = _sigmoid(lin)
        ww = np.clip(pr * (1.0 - pr), 1e-6, None)
        z = lin + (yb - pr) / ww
        sw = np.sqrt(ww)
        design = np.column_stack([a * sw[:, None], ones * sw])
        xtx = design.T @ design
        xty = design.T @ (z * sw)
        try:
            coef = np.linalg.solve(xtx + pen, xty)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(design, z * sw, rcond=None)
        w = coef[:n_feat].astype(np.float64, copy=False)
        bias = float(coef[-1])
    return {
        "weights": w,
        "bias": bias,
        "ridge": float(ridge),
        "n": int(y.size),
        "train_up_rate": float(yb.mean()),
    }


def apply_pup_logistic(
    x: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    w = np.asarray(spec.get("weights") if spec is not None else 0.0, dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if w.size == 0 or a.size == 0:
        return np.full(a.shape[0], 0.5, dtype=np.float64)
    if w.size != a.shape[1]:
        w = np.resize(w, a.shape[1])
    return _sigmoid(a @ w + float(spec.get("bias") or 0.0))


def pup_sleeve_mask(
    p_up: np.ndarray,
    dates: np.ndarray,
    *,
    min_cover: float = MIN_COVER,
    min_names: int = MIN_NAMES,
) -> np.ndarray:
    """Keep the within-date top slice so global cover is at least ``min_cover``.

    Single rule: per date keep the top ``ceil(min_cover * n_date)`` names by
    P(up) (at least ``min_names`` when the date is wide enough). Not a q-grid.
    """
    p = np.asarray(p_up, dtype=np.float64)
    d = np.asarray(dates, dtype=np.int64)
    keep = np.zeros(p.shape[0], dtype=bool)
    floor = float(min_cover) if min_cover > 0 else MIN_COVER
    for key in np.unique(d):
        sel = d == key
        idx = np.flatnonzero(sel)
        if idx.size < int(min_names):
            continue
        n_keep = max(int(min_names), int(math.ceil(floor * idx.size)))
        n_keep = min(n_keep, int(idx.size))
        scores = p[idx]
        order = np.argsort(-np.where(np.isfinite(scores), scores, -np.inf), kind="mergesort")
        chosen = idx[order[:n_keep]]
        keep[chosen] = np.isfinite(p[chosen])
    return keep


def score_pup_sleeve(
    r_on: np.ndarray,
    mask: np.ndarray,
    *,
    n_full: int,
) -> dict[str, float]:
    r = np.asarray(r_on, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(r)
    n_full = max(int(n_full), int(r.size), 1)
    n_sleeve = int(m.sum())
    cover = float(n_sleeve) / float(n_full)
    if n_sleeve < 1:
        return {
            "up_pct": float("nan"),
            "cover": cover,
            "n_sleeve": 0.0,
            "n_full": float(n_full),
            "n_up": 0.0,
        }
    n_up = int((r[m] > 0.0).sum())
    return {
        "up_pct": 100.0 * float(n_up) / float(n_sleeve),
        "cover": cover,
        "n_sleeve": float(n_sleeve),
        "n_full": float(n_full),
        "n_up": float(n_up),
    }


def score_split_pup(
    x: np.ndarray,
    r_on: np.ndarray,
    dates: np.ndarray,
    spec: Mapping[str, Any],
    *,
    min_cover: float = MIN_COVER,
    min_names: int = MIN_NAMES,
) -> dict[str, Any]:
    p = apply_pup_logistic(x, spec)
    mask = pup_sleeve_mask(p, dates, min_cover=min_cover, min_names=min_names)
    row = score_pup_sleeve(r_on, mask, n_full=int(np.asarray(r_on).shape[0]))
    row["mean_p_up"] = float(np.nanmean(p)) if p.size else float("nan")
    row["min_cover"] = float(min_cover)
    row["kind"] = "p_up"
    return row


def labelled_overnight_r(
    symbols: Sequence[Any],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for sym in symbols:
        if not bool(getattr(sym, "valid", np.zeros(0, dtype=bool)).any()):
            continue
        r = getattr(sym, "overnight_r", None)
        if r is None:
            chunks.append(np.full(int(sym.valid.sum()), np.nan, dtype=np.float64))
            continue
        chunks.append(np.asarray(r, dtype=np.float64)[np.asarray(sym.valid, dtype=bool)])
    if not chunks:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def decide_pup_promote(
    val_row: Mapping[str, Any] | None,
    *,
    target_up: float = TARGET_UP_PCT,
    min_cover: float = MIN_COVER,
    honest_baseline: float = HONEST_SKIP_RANK_VAL_UP,
) -> dict[str, Any]:
    """VAL-only hit-rate gate. TEST never enters. Not a sleeve grid."""
    row = dict(val_row or {})
    up = _as_float(row.get("up_pct"))
    cover = _as_float(row.get("cover"))
    vs_honest = (
        float(up - honest_baseline) if math.isfinite(up) else float("nan")
    )
    if not math.isfinite(cover) or cover + 1e-12 < float(min_cover):
        return {
            "promote": False,
            "reached_60": False,
            "val_up_pct": up,
            "val_cover": cover,
            "vs_honest_pp": vs_honest,
            "reason": (
                f"NO PROMOTE P(up): VAL cover {100.0 * cover:.1f}% "
                f"below floor {100.0 * min_cover:.1f}%."
                if math.isfinite(cover)
                else "NO PROMOTE P(up): empty VAL sleeve."
            ),
        }
    if math.isfinite(up) and up + 1e-12 >= float(target_up):
        return {
            "promote": True,
            "reached_60": True,
            "val_up_pct": up,
            "val_cover": cover,
            "vs_honest_pp": vs_honest,
            "reason": (
                f"PROMOTE P(up): VAL overnight-up {up:.2f}% "
                f"(cover {100.0 * cover:.1f}%) >= {target_up:.0f}%."
            ),
        }
    lift = (
        f"best honest lift vs skip-rank {honest_baseline:.2f}% is "
        f"{vs_honest:+.2f} pp"
        if math.isfinite(vs_honest)
        else "no finite VAL overnight-up"
    )
    return {
        "promote": False,
        "reached_60": False,
        "val_up_pct": up,
        "val_cover": cover,
        "vs_honest_pp": vs_honest,
        "reason": (
            f"NO PROMOTE P(up): VAL overnight-up {up:.2f}% "
            f"(cover {100.0 * cover:.1f}%) missed {target_up:.0f}%. "
            f"{lift}. Autopsy: direct P(up) head, one cover-floor sleeve "
            "(not a skip-rank grid). TEST report-only. Book IR is a "
            "separate gate (live_locate vs +5.58)."
        ),
    }


def evaluate_pup(
    bundle: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    min_cover: float = MIN_COVER,
    min_names: int = MIN_NAMES,
) -> dict[str, Any]:
    """Score TRAIN / VAL / TEST. Selection uses VAL only."""
    mean = bundle["feature_mean"]
    std = bundle["feature_std"]
    splits: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        symbols = bundle.get(f"{name}_symbols") or []
        x, y_up, dates = labelled_pup_rows(symbols, mean, std)
        r_on = labelled_overnight_r(symbols)
        if r_on.shape[0] != x.shape[0] and x.size:
            # labelled_pup_rows drops NaN y; align r_on the same way.
            raw_x, _y_resid, raw_d = labelled_rows(symbols, mean, std)
            del raw_x, raw_d
            raw_r = labelled_overnight_r(symbols)
            y_all = []
            for sym in symbols:
                if not bool(getattr(sym, "valid", np.zeros(0, dtype=bool)).any()):
                    continue
                y_all.append(
                    overnight_up_labels(
                        getattr(sym, "overnight_r", None),
                        valid=sym.valid,
                        next_split_days=getattr(sym, "next_split_days", None),
                    )[np.asarray(sym.valid, dtype=bool)]
                )
            y_cat = np.concatenate(y_all) if y_all else np.zeros((0,))
            ok = np.isfinite(y_cat)
            if ok.shape[0] == raw_r.shape[0]:
                r_on = raw_r[ok]
        row = score_split_pup(
            x, r_on if r_on.shape[0] == x.shape[0] else y_up, dates, spec,
            min_cover=min_cover, min_names=min_names,
        )
        row["n_labelled"] = float(x.shape[0])
        row["fit_n"] = float(spec.get("n") or 0)
        splits[name] = row
    gate = decide_pup_promote(splits.get("val") or {})
    return {
        "kind": "p_up_overnight",
        "fit": {
            "ridge": spec.get("ridge"),
            "n": spec.get("n"),
            "train_up_rate": spec.get("train_up_rate"),
            "bias": spec.get("bias"),
        },
        "min_cover": float(min_cover),
        "target_up_pct": TARGET_UP_PCT,
        "honest_skip_rank_val_up": HONEST_SKIP_RANK_VAL_UP,
        "splits": splits,
        "gate": gate,
        "book_gate": {
            "metric": "live_locate_unlevered_net_ir",
            "baseline_ir": 5.58,
            "role": "report-only for this cut (hit-rate gate is separate)",
        },
        "cost_rank_loss": COST_RANK_LOSS,
        "test_report_only": True,
    }


def format_pup_block(payload: Mapping[str, Any]) -> str:
    splits = payload.get("splits") or {}
    gate = payload.get("gate") or {}
    lines = [
        "P(up) overnight head (2a) — direct classification, not skip-rank",
        f"  sleeve = rank P(up); cover floor {100.0 * float(payload.get('min_cover') or MIN_COVER):.0f}% "
        "(one rule, no q-grid)",
        f"  VAL gate overnight-up >= {float(payload.get('target_up_pct') or TARGET_UP_PCT):.0f}%  "
        "TEST report-only",
    ]
    for name in ("train", "val", "test"):
        row = splits.get(name) or {}
        tag = "  (report-only)" if name == "test" else ""
        lines.append(
            f"  {name:5} overnight-up { _fmt_pct(row.get('up_pct')) }  "
            f"cover { _fmt_cover(row.get('cover')) }  "
            f"n={int(_as_float(row.get('n_sleeve'), 0))}{tag}"
        )
    lines.append(f"  PROMOTE P(up)? {'YES' if gate.get('promote') else 'NO'}  "
                 f"REACHED 60%? {'YES' if gate.get('reached_60') else 'NO'}")
    if gate.get("reason"):
        lines.append(f"  {gate['reason']}")
    lines.append(
        "  Book live_locate IR vs +5.58 is report-only on this cut "
        "(separate from the hit-rate gate)."
    )
    return "\n".join(lines)


def _fmt_pct(value: Any) -> str:
    v = _as_float(value)
    return f"{v:.2f}%" if math.isfinite(v) else "  n/a"

def _fmt_cover(value: Any) -> str:
    v = _as_float(value)
    return f"{100.0 * v:.1f}%" if math.isfinite(v) else " n/a"
