"""Overnight gap residual: a first-class estimand, not mixed into close-to-close.

Causal contract (horizon ``h``, default 1 daily bar)
----------------------------------------------------
Features ``X_t`` use bars ``<= t`` only. Same-bar ``open_t`` is a *candle*
feature known at the close. The overnight **label** is the next open:

    r^{on}_t = log(open_{t+h}) - log(close_t)

    y_t = (r^{on}_t - beta_t * r^{on, hedge}_t) / sigma_t

``beta_t`` and ``sigma_t`` use returns through close ``t`` only. Hedge
*forward* overnight return is a label term, never a feature.
``open_{t+h}`` and ``close_{t+h}`` never enter ``X_t``.

Close-to-close is ``log(close_{t+h}) - log(close_t)``. Next-session
open→close is ``log(close_{t+h}) - log(open_{t+h})``. Open+N fill is

    r^{fill}_t = r^{on}_t + (N / 390) * r^{session}_t

Those are different books. Overnight-trained ``w`` is not assumed to
transfer onto close-to-close. Do **not** silently replace overnight ``y``
with the fill blend unless locked val lifts.

Trade that matches the overnight label
--------------------------------------
Enter at the close of ``t`` (MOC). Exit at the open of ``t+h`` (MOO / auction).
Flat during the next regular session. Do **not** EWMA-hold through the
session: that would mix in open→close, which is a worse estimand.

Open+N fill (sensitivity, not the default)
------------------------------------------
Enter MOC ``t``, exit in the continuous session ~N minutes after the open.
That avoids MOO but mixes in ``N/390`` of next-session return. Report
separately; promote only on the locked-val gate.

Live frictions the paper overnight IR omits: open-auction slippage vs the
official print (worse than a flat overlay, worse for thin/high-vol names),
locate/borrow on shorts held overnight, and a residual book also auctions
the sector/SPY hedge.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from forecast.config import BARS_PER_SESSION

LABEL_CLOSE = "close"
LABEL_OVERNIGHT = "overnight"
LABEL_SESSION = "session"
LABEL_FILL = "open_fill"

_CLOSE_ALIASES = frozenset({"", "close", "close_close", "cc"})
_ON_ALIASES = frozenset({"overnight", "on", "gap", "close_open"})
_OC_ALIASES = frozenset({"session", "oc", "open_close", "intraday"})
_FILL_ALIASES = frozenset({"open_fill", "fill", "open_n", "open+n", "vwap_open"})
_FILL_MINUTES_RE = re.compile(r"^(?:open|fill)[_+]?(\d+)$")

OVERNIGHT_FORMULA = (
    "r_on[t] = log(open[t+h]) - log(close[t]); "
    "y[t] = (r_on[t] - beta[t] * r_on_hedge[t]) / sigma[t]; "
    "X[t] uses bars <= t (same-bar open[t] is a candle feature; "
    "open[t+h] / close[t+h] are labels, never features). "
    "Trade: MOC t -> MOO t+h; flat in the next session."
)
FILL_FORMULA = (
    "r_fill[t] = r_on[t] + (N/390)*r_session[t]; "
    "Trade: MOC t -> open+N continuous fill; not the default overnight label."
)

# Paper 10 bp flatten (old headline that understated overnight costs).
PAPER_BUNDLE: dict[str, Any] = {
    "name": "paper_10bp",
    "round_trip_bps": 10.0,
    "open_auction_bps": 0.0,
    "moc_bps": 0.0,
    "moo_bps": 0.0,
    "session_exit_bps": 0.0,
    "borrow_bps": 0.0,
    "hedge_cost_bps": 0.0,
    "impact_vol_k": 0.0,
    "thin_mult": 1.0,
    "thin_pctile": 0.0,
    "locate_pctile": 0.0,
    "ex_post_gap_k": 0.0,
    "adv_borrow_k": 0.0,
    "adv_auction_k": 0.0,
}

# Flat overlay used in the first overnight live-ish print (IR ~1.6–2.0).
# Auction is a constant on the *exit* half-notional, not name-level MOO.
LIVE_FLAT_BUNDLE: dict[str, Any] = {
    "name": "live_flat",
    "round_trip_bps": 20.0,
    "open_auction_bps": 10.0,
    "moc_bps": 0.0,
    "moo_bps": 0.0,
    "session_exit_bps": 0.0,
    "borrow_bps": 5.0,
    "hedge_cost_bps": 10.0,
    "impact_vol_k": 0.0,
    "thin_mult": 1.0,
    "thin_pctile": 0.0,
    "locate_pctile": 0.0,
    "ex_post_gap_k": 0.0,
    "adv_borrow_k": 0.0,
    "adv_auction_k": 0.0,
}

# Name-level auction: extra MOC/MOO vs official prints, thin-name multiplier,
# vol impact from trailing vol_level (known at t). Default Owen live book.
LIVE_BUNDLE: dict[str, Any] = {
    "name": "live",
    "round_trip_bps": 20.0,
    "open_auction_bps": 0.0,
    "moc_bps": 5.0,
    "moo_bps": 10.0,
    "session_exit_bps": 0.0,
    "borrow_bps": 5.0,
    "hedge_cost_bps": 10.0,
    "impact_vol_k": 8.0,
    "thin_mult": 2.0,
    "thin_pctile": 0.30,
    "locate_pctile": 0.0,
    "ex_post_gap_k": 0.0,
    "adv_borrow_k": 0.0,
    "adv_auction_k": 0.0,
}

LIVE_LOCATE_BUNDLE: dict[str, Any] = {
    **LIVE_BUNDLE,
    "name": "live_locate",
    "locate_pctile": 0.30,
}

LIVE_LONG_ONLY_BUNDLE: dict[str, Any] = {
    **LIVE_BUNDLE,
    "name": "live_long_only",
    "borrow_bps": 0.0,
    "locate_pctile": 0.0,
}

# Stress: ugly MOO, more HTB, higher impact. If this kills the book, say so.
HARSH_BUNDLE: dict[str, Any] = {
    "name": "harsh_auction",
    "round_trip_bps": 20.0,
    "open_auction_bps": 0.0,
    "moc_bps": 10.0,
    "moo_bps": 30.0,
    "session_exit_bps": 0.0,
    "borrow_bps": 10.0,
    "hedge_cost_bps": 15.0,
    "impact_vol_k": 16.0,
    "thin_mult": 3.0,
    "thin_pctile": 0.40,
    "locate_pctile": 0.30,
    "ex_post_gap_k": 0.0,
    "adv_borrow_k": 0.0,
    "adv_auction_k": 0.0,
}

# Ex-post |gap| impact is a *sensitivity* (uses the realized overnight move).
EX_POST_GAP_BUNDLE: dict[str, Any] = {
    **LIVE_BUNDLE,
    "name": "live_ex_post_gap",
    "ex_post_gap_k": 0.25,
}

# Open+N continuous exit: still overnight borrow, MOC entry, no MOO.
FILL_LIVE_BUNDLE: dict[str, Any] = {
    "name": "fill_live",
    "round_trip_bps": 20.0,
    "open_auction_bps": 0.0,
    "moc_bps": 5.0,
    "moo_bps": 0.0,
    "session_exit_bps": 5.0,
    "borrow_bps": 5.0,
    "hedge_cost_bps": 10.0,
    "impact_vol_k": 8.0,
    "thin_mult": 2.0,
    "thin_pctile": 0.30,
    "locate_pctile": 0.0,
    "ex_post_gap_k": 0.0,
    "adv_borrow_k": 0.0,
    "adv_auction_k": 0.0,
}

# Live + ADV-scaled borrow/auction (thin names cost more than the flat overlay).
LIVE_ADV_BUNDLE: dict[str, Any] = {
    **LIVE_BUNDLE,
    "name": "live_adv",
    "adv_borrow_k": 0.5,
    "adv_auction_k": 0.5,
}

# Borrow stress: HTB names (low ADV) pay a steeper locate. Long-only zeros this.
BORROW_STRESS_BUNDLE: dict[str, Any] = {
    **LIVE_BUNDLE,
    "name": "borrow_stress",
    "borrow_bps": 15.0,
    "adv_borrow_k": 1.0,
    "adv_auction_k": 0.0,
    "locate_pctile": 0.30,
}

# Auction stress: MOC/MOO scale with 1/ADV on top of the harsh print.
AUCTION_STRESS_BUNDLE: dict[str, Any] = {
    **HARSH_BUNDLE,
    "name": "auction_stress",
    "adv_borrow_k": 0.0,
    "adv_auction_k": 1.0,
    "locate_pctile": 0.0,
}

LIVE_ADV_LONG_ONLY_BUNDLE: dict[str, Any] = {
    **LIVE_ADV_BUNDLE,
    "name": "live_adv_long_only",
    "borrow_bps": 0.0,
    "locate_pctile": 0.0,
    "adv_borrow_k": 0.0,
}

COST_BUNDLES: dict[str, dict[str, Any]] = {
    "paper": PAPER_BUNDLE,
    "paper_10bp": PAPER_BUNDLE,
    "live_flat": LIVE_FLAT_BUNDLE,
    "live": LIVE_BUNDLE,
    "live_locate": LIVE_LOCATE_BUNDLE,
    "live_long_only": LIVE_LONG_ONLY_BUNDLE,
    "live_adv": LIVE_ADV_BUNDLE,
    "live_adv_long_only": LIVE_ADV_LONG_ONLY_BUNDLE,
    "harsh": HARSH_BUNDLE,
    "harsh_auction": HARSH_BUNDLE,
    "borrow_stress": BORROW_STRESS_BUNDLE,
    "auction_stress": AUCTION_STRESS_BUNDLE,
    "ex_post_gap": EX_POST_GAP_BUNDLE,
    "fill_live": FILL_LIVE_BUNDLE,
}

DEFAULT_FILL_MINUTES = 15
VAL_LIFT = 0.003
# Overnight skip val-2017 was +0.0354; do not kill it for a 2023 patch.
VAL_2017_KEEP = 0.015


def parse_label_spec(kind: str | None) -> tuple[str, int]:
    """Return ``(label_kind, fill_minutes)``. ``fill_minutes`` is 0 unless open+N."""
    raw = str(kind or LABEL_CLOSE).strip().lower()
    match = _FILL_MINUTES_RE.match(raw)
    if match:
        return LABEL_FILL, int(match.group(1))
    if raw in _FILL_ALIASES:
        return LABEL_FILL, 0
    if raw in _CLOSE_ALIASES:
        return LABEL_CLOSE, 0
    if raw in _ON_ALIASES:
        return LABEL_OVERNIGHT, 0
    if raw in _OC_ALIASES:
        return LABEL_SESSION, 0
    raise ValueError(
        f"unknown label_return {kind!r}; expected close, overnight, session, "
        "open_fill, or open15/fill30"
    )


def normalize_label_return(kind: str | None) -> str:
    """Map CLI / checkpoint aliases onto close | overnight | session | open_fill."""
    return parse_label_spec(kind)[0]


def fill_minutes_for(kind: str | None, fill_minutes: int | None = None) -> int:
    parsed, parsed_mins = parse_label_spec(kind)
    if parsed != LABEL_FILL:
        return 0
    if fill_minutes is not None and int(fill_minutes) > 0:
        return int(fill_minutes)
    if parsed_mins > 0:
        return int(parsed_mins)
    return DEFAULT_FILL_MINUTES


def uses_next_open(kind: str | None) -> bool:
    return normalize_label_return(kind) in (LABEL_OVERNIGHT, LABEL_SESSION, LABEL_FILL)


def uses_next_close(kind: str | None) -> bool:
    return normalize_label_return(kind) in (LABEL_CLOSE, LABEL_SESSION, LABEL_FILL)


def fill_frac_minutes(n_minutes: int, *, bars_per_session: int = BARS_PER_SESSION) -> float:
    bars = max(1, int(bars_per_session))
    return float(min(1.0, max(0.0, int(n_minutes) / float(bars))))


def log_positive_price(px: pd.Series) -> pd.Series:
    """Natural log; NaN unless the price is finite and strictly positive.

    Do **not** clip to 1e-12. That would turn a missing/bad open into a fake
    −∞ gap and look like a tradeable overnight move.
    """
    s = pd.to_numeric(px, errors="coerce").astype(np.float64)
    ok = np.isfinite(s.to_numpy()) & (s.to_numpy() > 0.0)
    out = pd.Series(np.nan, index=s.index, dtype=np.float64)
    out.loc[ok] = np.log(s.loc[ok])
    return out


def same_bar_open_for_features(open_px: pd.Series, close: pd.Series) -> pd.Series:
    """Candle features may substitute close when same-bar open is unusable.

    That substitution uses prices known at ``t``. It is **not** used for the
    overnight / session / fill label (those need a real next open).
    """
    o = pd.to_numeric(open_px, errors="coerce").astype(np.float64)
    c = pd.to_numeric(close, errors="coerce").astype(np.float64)
    ok = np.isfinite(o.to_numpy()) & (o.to_numpy() > 0.0)
    return o.where(ok, c)


def forward_log_return(
    *,
    close: pd.Series,
    open_px: pd.Series,
    kind: str,
    horizon: int,
    fill_minutes: int = 0,
) -> pd.Series:
    """Causal forward log-return for the chosen estimand. Future prices only."""
    h = int(horizon)
    log_close = log_positive_price(close)
    label = normalize_label_return(kind)
    if label == LABEL_CLOSE:
        return log_close.shift(-h) - log_close
    log_open = log_positive_price(open_px)
    r_on = log_open.shift(-h) - log_close
    r_sess = log_close.shift(-h) - log_open.shift(-h)
    if label == LABEL_OVERNIGHT:
        return r_on
    if label == LABEL_SESSION:
        return r_sess
    alpha = fill_frac_minutes(fill_minutes_for(kind, fill_minutes))
    return r_on + alpha * r_sess


def next_open_valid(open_px: pd.Series, horizon: int) -> pd.Series:
    """True when the horizon open exists and is a positive finite print."""
    nxt = pd.to_numeric(open_px, errors="coerce").astype(np.float64).shift(-int(horizon))
    return nxt.notna() & np.isfinite(nxt) & (nxt > 0.0)


def holding_for_label(kind: str | None) -> str:
    """Backtest holding period that matches the residual label."""
    label = normalize_label_return(kind)
    if label == LABEL_OVERNIGHT:
        return "overnight"
    if label == LABEL_FILL:
        return "open_fill"
    return "close"


def overnight_one_way_turnover(weights: np.ndarray) -> np.ndarray:
    """Enter-from-flat and exit-to-flat each date: one-way = sum(|w|).

    Matches ``0.5 * L1(w - 0) + 0.5 * L1(0 - w)`` used by the close-to-close
    book for a single rebalance. A 50/50 long-short row has one-way 1.0, i.e.
    one full round-trip of the overnight book per night.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        return np.asarray([float(np.abs(w).sum())], dtype=np.float64)
    return np.abs(w).sum(axis=1).astype(np.float64)


def _as_2d(arr: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        if out.size == shape[1]:
            out = np.broadcast_to(out.reshape(1, -1), shape).copy()
        elif out.size == shape[0]:
            out = np.broadcast_to(out.reshape(-1, 1), shape).copy()
        else:
            out = out.reshape(shape)
    if out.shape != shape:
        raise ValueError(f"aux panel shape {out.shape} != weights {shape}")
    return out


def row_cs_thin_mask(turnover_z: np.ndarray, pctile: float) -> np.ndarray:
    """True for names at or below the within-row percentile of ``turnover_z``.

    ``turnover_z`` is known at close ``t`` (feature, not a label). Bottom
    ADV/turnover names are the HTB / high-auction-cost proxy.
    """
    z = np.asarray(turnover_z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    out = np.zeros(z.shape, dtype=bool)
    p = float(pctile)
    if p <= 0 or p >= 1:
        if p >= 1:
            return np.isfinite(z)
        return out
    for i in range(z.shape[0]):
        row = z[i]
        finite = np.isfinite(row)
        if int(finite.sum()) < 3:
            continue
        if float(np.nanmax(row[finite]) - np.nanmin(row[finite])) < 1e-12:
            continue
        cut = float(np.nanpercentile(row[finite], 100.0 * p))
        out[i] = finite & (row < cut)
    return out


def apply_locate_gate(
    weights: np.ndarray,
    turnover_z: np.ndarray | None,
    *,
    pctile: float,
    long_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero shorts on thin names; rescale remaining shorts toward 0.5 NAV.

    Longs are unchanged (still ~0.5). If every short is blocked the date is
    long-biased at 0.5 NAV — that is the locate constraint, not a silent
    long-only renormalization. Returns ``(weights, n_blocked_per_date)``.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    squeeze = w.ndim == 1
    if squeeze:
        w = w.reshape(1, -1)
    blocked = np.zeros(w.shape[0], dtype=np.float64)
    if long_only or float(pctile) <= 0 or turnover_z is None:
        return (w[0] if squeeze else w), blocked
    thin = row_cs_thin_mask(_as_2d(turnover_z, w.shape), float(pctile))
    for i in range(w.shape[0]):
        short = w[i] < 0
        hit = short & thin[i]
        blocked[i] = float(hit.sum())
        if not bool(hit.any()):
            continue
        w[i, hit] = 0.0
        ss = float((-np.clip(w[i], None, 0.0)).sum())
        if ss > 1e-12:
            w[i] = np.where(w[i] < 0.0, w[i] * (0.5 / ss), w[i])
    return (w[0] if squeeze else w), blocked


def overnight_stress_costs(
    *,
    round_trip_bps: float,
    open_auction_bps: float = 0.0,
    borrow_bps: float = 0.0,
    hedge_cost_bps: float = 0.0,
    weights: np.ndarray,
    moc_bps: float = 0.0,
    moo_bps: float = 0.0,
    session_exit_bps: float = 0.0,
    turnover_z: np.ndarray | None = None,
    vol_level: np.ndarray | None = None,
    realized_abs: np.ndarray | None = None,
    impact_vol_k: float = 0.0,
    thin_mult: float = 1.0,
    thin_pctile: float = 0.0,
    ex_post_gap_k: float = 0.0,
    dollar_adv: np.ndarray | None = None,
    adv_borrow_k: float = 0.0,
    adv_auction_k: float = 0.0,
) -> np.ndarray:
    """Per-date cost (fraction of NAV) for an overnight flatten book.

    ``round_trip_bps`` is charged on overnight one-way turnover (enter+exit).
    ``open_auction_bps`` is the *legacy* extra one-way cost on the open exit
    leg (``0.5 * sum(|w|)``). ``moc_bps`` / ``moo_bps`` are extra one-way
    auction costs on full ``sum(|w|)`` (official print vs fill). Thin names
    (low CS ``turnover_z``) multiply MOC+MOO. ``impact_vol_k`` adds
    ``k * max(vol_level, 0)`` bps on ``|w|`` from a feature known at ``t``.
    ``ex_post_gap_k`` is a sensitivity that uses ``|realized|`` — not default.
    ``borrow_bps`` applies to short notional. ``hedge_cost_bps`` is a flat
    overlay for auctioning the sector/SPY residual hedge.
    """
    return overnight_cost_breakdown(
        weights=weights,
        round_trip_bps=round_trip_bps,
        open_auction_bps=open_auction_bps,
        borrow_bps=borrow_bps,
        hedge_cost_bps=hedge_cost_bps,
        moc_bps=moc_bps,
        moo_bps=moo_bps,
        session_exit_bps=session_exit_bps,
        turnover_z=turnover_z,
        vol_level=vol_level,
        realized_abs=realized_abs,
        impact_vol_k=impact_vol_k,
        thin_mult=thin_mult,
        thin_pctile=thin_pctile,
        ex_post_gap_k=ex_post_gap_k,
        dollar_adv=dollar_adv,
        adv_borrow_k=adv_borrow_k,
        adv_auction_k=adv_auction_k,
    )["total"]


def overnight_cost_breakdown(
    *,
    weights: np.ndarray,
    round_trip_bps: float,
    open_auction_bps: float = 0.0,
    borrow_bps: float = 0.0,
    hedge_cost_bps: float = 0.0,
    moc_bps: float = 0.0,
    moo_bps: float = 0.0,
    session_exit_bps: float = 0.0,
    turnover_z: np.ndarray | None = None,
    vol_level: np.ndarray | None = None,
    realized_abs: np.ndarray | None = None,
    impact_vol_k: float = 0.0,
    thin_mult: float = 1.0,
    thin_pctile: float = 0.0,
    ex_post_gap_k: float = 0.0,
    dollar_adv: np.ndarray | None = None,
    adv_borrow_k: float = 0.0,
    adv_auction_k: float = 0.0,
) -> dict[str, np.ndarray]:
    """Named per-date cost components (fraction of NAV) plus ``total``."""
    from forecast.levers import adv_cost_scale

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    abs_w = np.abs(w)
    one_way = overnight_one_way_turnover(w)
    exit_leg = 0.5 * abs_w.sum(axis=1)
    short_nav = np.clip(-w, 0.0, None)
    rt = (float(round_trip_bps) * 1e-4) * one_way
    legacy_moo = (float(open_auction_bps) * 1e-4) * exit_leg
    hedge = (float(hedge_cost_bps) * 1e-4) * np.ones_like(one_way)

    borrow_scale = np.ones_like(abs_w)
    auction_adv_scale = np.ones_like(abs_w)
    if dollar_adv is not None:
        if float(adv_borrow_k) > 0:
            borrow_scale = adv_cost_scale(_as_2d(dollar_adv, w.shape), k=float(adv_borrow_k))
        if float(adv_auction_k) > 0:
            auction_adv_scale = adv_cost_scale(
                _as_2d(dollar_adv, w.shape), k=float(adv_auction_k)
            )
    borrow = (float(borrow_bps) * 1e-4) * (short_nav * borrow_scale).sum(axis=1)

    thin_scale = np.ones_like(abs_w)
    if float(thin_pctile) > 0 and turnover_z is not None and float(thin_mult) > 1.0:
        thin = row_cs_thin_mask(_as_2d(turnover_z, w.shape), float(thin_pctile))
        thin_scale = np.where(thin, float(thin_mult), 1.0)
    auction_notional = abs_w * thin_scale * auction_adv_scale
    moc = (float(moc_bps) * 1e-4) * auction_notional.sum(axis=1)
    moo = (float(moo_bps) * 1e-4) * auction_notional.sum(axis=1)
    sess = (float(session_exit_bps) * 1e-4) * auction_notional.sum(axis=1)

    impact = np.zeros_like(one_way)
    if float(impact_vol_k) and vol_level is not None:
        vol = np.clip(_as_2d(vol_level, w.shape), 0.0, None)
        # Missing vol is "no extra impact", not a NaN date that pandas IR then drops.
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        impact = (float(impact_vol_k) * 1e-4) * (vol * abs_w).sum(axis=1)

    ex_post = np.zeros_like(one_way)
    if float(ex_post_gap_k) and realized_abs is not None:
        gap = np.clip(_as_2d(realized_abs, w.shape), 0.0, None)
        # k * |r| * |w|: 0.25 * 1% gap * 1 NAV = 25 bp on a 1% move.
        ex_post = float(ex_post_gap_k) * (gap * abs_w).sum(axis=1)

    total = rt + legacy_moo + moc + moo + sess + borrow + hedge + impact + ex_post
    return {
        "round_trip": rt.astype(np.float64),
        "legacy_open_auction": legacy_moo.astype(np.float64),
        "moc": moc.astype(np.float64),
        "moo": moo.astype(np.float64),
        "session_exit": sess.astype(np.float64),
        "borrow": borrow.astype(np.float64),
        "hedge": hedge.astype(np.float64),
        "impact": impact.astype(np.float64),
        "ex_post_gap": ex_post.astype(np.float64),
        "total": total.astype(np.float64),
    }


def resolve_cost_bundle(name: str | None) -> dict[str, Any]:
    raw = str(name or "").strip().lower()
    if not raw:
        return dict(PAPER_BUNDLE)
    if raw not in COST_BUNDLES:
        raise ValueError(
            f"unknown cost bundle {name!r}; expected one of {sorted(COST_BUNDLES)}"
        )
    return dict(COST_BUNDLES[raw])


def merge_cost_kwargs(
    bundle: Mapping[str, Any] | None = None, **overrides: Any
) -> dict[str, Any]:
    """Start from a named/dict bundle, then apply explicit CLI overrides."""
    out = dict(PAPER_BUNDLE)
    if bundle:
        out.update({k: v for k, v in dict(bundle).items() if v is not None})
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def book_side_stats(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    long_nav = np.clip(w, 0.0, None).sum(axis=1)
    short_nav = np.clip(-w, 0.0, None).sum(axis=1)
    n_long = (w > 1e-12).sum(axis=1)
    n_short = (w < -1e-12).sum(axis=1)
    return {
        "mean_long_nav": float(long_nav.mean()) if long_nav.size else float("nan"),
        "mean_short_nav": float(short_nav.mean()) if short_nav.size else float("nan"),
        "mean_gross": float((long_nav + short_nav).mean()) if long_nav.size else float("nan"),
        "mean_n_long": float(n_long.mean()) if n_long.size else float("nan"),
        "mean_n_short": float(n_short.mean()) if n_short.size else float("nan"),
    }


def capacity_note(
    *,
    long_only: bool,
    mean_long_nav: float,
    mean_short_nav: float,
    mean_turnover: float,
    locate_pctile: float = 0.0,
) -> str:
    """ADV/locate comments. Overnight flatten turns ~100% of the book nightly."""
    if long_only:
        return (
            f"long-only overnight: long NAV {mean_long_nav:.2f}, short 0, "
            f"one-way turn {mean_turnover:.2f}/night. No locate. Long-sleeve "
            "ADV participation is ~2x the 50/50 long-short long sleeve at the "
            "same NAV. Residual still assumes a liquid ETF hedge overlay."
        )
    locate = (
        f" Locate gate: no shorts in the bottom {100 * locate_pctile:.0f}% "
        "CS turnover_z (HTB proxy)."
        if locate_pctile > 0
        else " Shorts are unconstrained (paper locate)."
    )
    return (
        f"long-short overnight: long NAV {mean_long_nav:.2f} / short "
        f"{mean_short_nav:.2f}, one-way turn {mean_turnover:.2f}/night."
        f"{locate} Flattening 85 liquid names at MOC/MOO is a small-NAV book; "
        "do not scale into a large fraction of auction volume."
    )


def formula_log_line(kind: str | None, *, horizon: int = 1, fill_minutes: int = 0) -> str:
    label = normalize_label_return(kind)
    h = int(horizon)
    if label == LABEL_OVERNIGHT:
        return f"label_return=overnight h={h}  {OVERNIGHT_FORMULA}"
    if label == LABEL_SESSION:
        return (
            f"label_return=session h={h}  "
            "r_oc[t] = log(close[t+h]) - log(open[t+h]); features still at close t"
        )
    if label == LABEL_FILL:
        n = fill_minutes_for(kind, fill_minutes)
        return (
            f"label_return=open_fill N={n}m h={h}  {FILL_FORMULA} "
            f"alpha={fill_frac_minutes(n):.4f}"
        )
    return (
        f"label_return=close h={h}  "
        "r_cc[t] = log(close[t+h]) - log(close[t]); locked close-to-close book"
    )


def slim_cs_stats(stats: dict[str, Any]) -> dict[str, float]:
    keys = (
        "cs_ic",
        "cs_ic_spearman",
        "cs_ic_tstat",
        "cs_n_dates",
        "cs_mean_n",
        "cs_n_flat",
    )
    out: dict[str, float] = {}
    for key in keys:
        if key in stats:
            out[key] = float(stats[key])
    return out
