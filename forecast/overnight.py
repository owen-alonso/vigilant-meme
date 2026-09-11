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
open→close is ``log(close_{t+h}) - log(open_{t+h})``. Those are different
books. Overnight-trained ``w`` is not assumed to transfer onto close-to-close.

Trade that matches the label
----------------------------
Enter at the close of ``t`` (MOC). Exit at the open of ``t+h`` (MOO / auction).
Flat during the next regular session. Do **not** EWMA-hold through the
session: that would mix in open→close, which is a worse estimand.

Live frictions the paper overnight IR omits: open-auction slippage vs the
official print, locate/borrow on shorts held overnight, and a residual book
also auctions the sector/SPY hedge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LABEL_CLOSE = "close"
LABEL_OVERNIGHT = "overnight"
LABEL_SESSION = "session"

_CLOSE_ALIASES = frozenset({"", "close", "close_close", "cc"})
_ON_ALIASES = frozenset({"overnight", "on", "gap", "close_open"})
_OC_ALIASES = frozenset({"session", "oc", "open_close", "intraday"})

OVERNIGHT_FORMULA = (
    "r_on[t] = log(open[t+h]) - log(close[t]); "
    "y[t] = (r_on[t] - beta[t] * r_on_hedge[t]) / sigma[t]; "
    "X[t] uses bars <= t (same-bar open[t] is a candle feature; "
    "open[t+h] / close[t+h] are labels, never features). "
    "Trade: MOC t -> MOO t+h; flat in the next session."
)


def normalize_label_return(kind: str | None) -> str:
    """Map CLI / checkpoint aliases onto close | overnight | session."""
    raw = str(kind or LABEL_CLOSE).strip().lower()
    if raw in _CLOSE_ALIASES:
        return LABEL_CLOSE
    if raw in _ON_ALIASES:
        return LABEL_OVERNIGHT
    if raw in _OC_ALIASES:
        return LABEL_SESSION
    raise ValueError(
        f"unknown label_return {kind!r}; expected close, overnight, or session"
    )


def uses_next_open(kind: str | None) -> bool:
    return normalize_label_return(kind) in (LABEL_OVERNIGHT, LABEL_SESSION)


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
    overnight / session label (those need a real next open).
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
) -> pd.Series:
    """Causal forward log-return for the chosen estimand. Future prices only."""
    h = int(horizon)
    log_close = log_positive_price(close)
    label = normalize_label_return(kind)
    if label == LABEL_CLOSE:
        return log_close.shift(-h) - log_close
    log_open = log_positive_price(open_px)
    if label == LABEL_OVERNIGHT:
        return log_open.shift(-h) - log_close
    return log_close.shift(-h) - log_open.shift(-h)


def next_open_valid(open_px: pd.Series, horizon: int) -> pd.Series:
    """True when the horizon open exists and is a positive finite print."""
    nxt = pd.to_numeric(open_px, errors="coerce").astype(np.float64).shift(-int(horizon))
    return nxt.notna() & np.isfinite(nxt) & (nxt > 0.0)


def holding_for_label(kind: str | None) -> str:
    """Backtest holding period that matches the residual label."""
    return "overnight" if normalize_label_return(kind) == LABEL_OVERNIGHT else "close"


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


def overnight_stress_costs(
    *,
    round_trip_bps: float,
    open_auction_bps: float = 0.0,
    borrow_bps: float = 0.0,
    hedge_cost_bps: float = 0.0,
    weights: np.ndarray,
) -> np.ndarray:
    """Per-date cost (fraction of NAV) for an overnight flatten book.

    ``round_trip_bps`` is charged on overnight one-way turnover (enter+exit).
    ``open_auction_bps`` is extra one-way cost on the open *exit* leg
    (``0.5 * sum(|w|)``). ``borrow_bps`` applies to short notional held
    overnight. ``hedge_cost_bps`` is a flat overlay for auctioning the
    sector/SPY residual hedge (~1 NAV notional, one round-trip / night).
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    one_way = overnight_one_way_turnover(w)
    exit_leg = 0.5 * np.abs(w).sum(axis=1)
    short_nav = np.clip(-w, 0.0, None).sum(axis=1)
    cost = (float(round_trip_bps) * 1e-4) * one_way
    cost = cost + (float(open_auction_bps) * 1e-4) * exit_leg
    cost = cost + (float(borrow_bps) * 1e-4) * short_nav
    cost = cost + (float(hedge_cost_bps) * 1e-4) * np.ones_like(one_way)
    return cost.astype(np.float64)


def formula_log_line(kind: str | None, *, horizon: int = 1) -> str:
    label = normalize_label_return(kind)
    h = int(horizon)
    if label == LABEL_OVERNIGHT:
        return f"label_return=overnight h={h}  {OVERNIGHT_FORMULA}"
    if label == LABEL_SESSION:
        return (
            f"label_return=session h={h}  "
            "r_oc[t] = log(close[t+h]) - log(open[t+h]); features still at close t"
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
