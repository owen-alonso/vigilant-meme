"""Closed-form CS ridge and causal walk-forward refits.

Frozen ridge (one ``w`` fit on train) is the generate.py skip. Walk-forward
refits ``w`` on labels whose horizon is strictly before the score date, so
ancient pre-GFC structure can age out without peeking at today's residual.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from forecast.data import CROSS_SECTION_FEATURES, FEATURE_NAMES, SymbolArrays


def feature_mask(mode: str) -> np.ndarray:
    """Boolean mask over ``FEATURE_NAMES``. ``all`` keeps every column."""
    names = list(FEATURE_NAMES)
    n = len(names)
    raw = (mode or "all").strip().lower()
    mask = np.ones(n, dtype=bool)
    if raw in ("", "all"):
        return mask
    if raw in ("cs", "cs_only"):
        keep = set(CROSS_SECTION_FEATURES)
        return np.array([nm in keep for nm in names], dtype=bool)
    if raw in ("no_calendar", "no_tod"):
        drop = {"traded", "staleness", "new_session", "tod_sin", "tod_cos", "tod_frac", "dow_frac"}
        return np.array([nm not in drop for nm in names], dtype=bool)
    raise ValueError(f"unknown feature mask {mode!r}")


def labelled_rows(
    symbols: Sequence[SymbolArrays],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized last-bar rows: ``x [N, F]``, ``y [N]``, ``dates [N]`` (days since epoch)."""
    mean = np.asarray(feature_mean, dtype=np.float64)
    std = np.asarray(feature_std, dtype=np.float64)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    rows_d: list[np.ndarray] = []
    for sym in symbols:
        if not bool(sym.valid.any()):
            continue
        raw = sym.features[sym.valid].astype(np.float64, copy=False)
        rows_x.append((raw - mean) / std)
        rows_y.append(sym.target[sym.valid].astype(np.float64, copy=False))
        if sym.dates is None:
            rows_d.append(np.full(int(sym.valid.sum()), -1, dtype=np.int64))
        else:
            rows_d.append(sym.dates[sym.valid].astype(np.int64, copy=False))
    if not rows_x:
        f = int(mean.shape[0])
        return (
            np.zeros((0, f), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(rows_x, axis=0),
        np.concatenate(rows_y, axis=0),
        np.concatenate(rows_d, axis=0),
    )


def _prepare_cs_design(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int,
    cs_demean: bool,
    cs_zscore: bool,
    rank_target: bool,
    feature_mask_bool: np.ndarray | None,
    date_halflife: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Within-date design for ridge. Returns ``x, y, sample_weight`` or None."""
    if not cs_demean and not cs_zscore and not rank_target:
        x = x.copy()
        if feature_mask_bool is not None:
            mask = np.asarray(feature_mask_bool, dtype=bool)
            x[:, ~mask] = 0.0
        w = np.ones(x.shape[0], dtype=np.float64)
        if date_halflife > 0 and dates.size:
            d_max = int(dates.max())
            w = 0.5 ** (np.maximum(0, d_max - dates.astype(np.int64)) / float(date_halflife))
        return x, y.copy(), w
    mask = (
        np.ones(x.shape[1], dtype=bool)
        if feature_mask_bool is None
        else np.asarray(feature_mask_bool, dtype=bool)
    )
    x = np.asarray(x, dtype=np.float64).copy()
    x[:, ~mask] = 0.0
    y = np.asarray(y, dtype=np.float64).copy()
    weights: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    d_max = int(dates.max()) if dates.size else 0
    hl = float(date_halflife)
    have = False
    for key in np.unique(dates):
        sel = dates == key
        n = int(sel.sum())
        if n < int(min_names):
            continue
        xd = x[sel]
        yd = y[sel]
        if rank_target:
            order = np.argsort(yd, kind="mergesort")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            if n > 1:
                yd = (ranks - ranks.mean()) / max(ranks.std(), 1e-8)
            else:
                yd = ranks - ranks.mean()
        if cs_demean or cs_zscore or rank_target:
            xd = xd - xd.mean(axis=0, keepdims=True)
            if not rank_target:
                yd = yd - yd.mean()
        if cs_zscore:
            std = xd.std(axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            xd = xd / std
        w = np.ones(n, dtype=np.float64)
        if hl > 0:
            w *= 0.5 ** (max(0, d_max - int(key)) / hl)
        xs.append(xd)
        ys.append(yd)
        weights.append(w)
        have = True
    if not have:
        return None
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)


def fit_ridge_xy(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    ridge: float = 1.0,
    min_names: int = 8,
    cs_demean: bool = True,
    cs_zscore: bool = False,
    rank_target: bool = False,
    feature_mask_bool: np.ndarray | None = None,
    date_halflife: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """CS ridge of ``y`` on ``x``. Returns weights, bias (0 if CS), in-sample mean CS IC."""
    n_features = int(x.shape[1]) if x.ndim == 2 else 0
    prepared = _prepare_cs_design(
        x,
        y,
        dates,
        min_names=min_names,
        cs_demean=cs_demean,
        cs_zscore=cs_zscore,
        rank_target=rank_target,
        feature_mask_bool=feature_mask_bool,
        date_halflife=date_halflife,
    )
    if prepared is None:
        return np.zeros(n_features, dtype=np.float32), 0.0, float("nan")
    xd, yd, w = prepared
    sw = np.sqrt(np.clip(w, 0.0, None))
    design = np.concatenate([xd, np.ones((xd.shape[0], 1), dtype=np.float64)], axis=1)
    dw = design * sw[:, None]
    yw = yd * sw
    lam = max(0.0, float(ridge))
    xtx = dw.T @ dw
    xtx.flat[:: xtx.shape[0] + 1] += lam
    try:
        coef = np.linalg.solve(xtx, dw.T @ yw)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx, dw.T @ yw, rcond=None)[0]
    weights = coef[:-1].astype(np.float32)
    bias = 0.0 if (cs_demean or rank_target or cs_zscore) else float(coef[-1])
    pred = x @ weights.astype(np.float64) + bias
    ic = mean_cs_ic(pred, y, dates, min_names=min_names)
    # Scale |w| so CS pred std matches CS y std on the fit sample (same as the old skip).
    if np.isfinite(ic) and pred.size >= 2:
        p_std = float(pred.std())
        y_std = float(y.std())
        if p_std > 1e-8 and y_std > 1e-8:
            amp = abs(ic) * y_std / p_std
            weights = (weights * amp).astype(np.float32)
            bias = float(bias * amp)
    return weights, bias, ic


def mean_cs_ic(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> float:
    from forecast.training import mean_cs_stats

    return float(mean_cs_stats(pred, target, dates, min_names=min_names)["cs_ic"])


def cs_stats(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> dict[str, float]:
    from forecast.training import mean_cs_stats

    return mean_cs_stats(pred, target, dates, min_names=min_names)


def date_ics(
    pred: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    *,
    min_names: int = 3,
) -> np.ndarray:
    """Per-date CS Pearson; rows with too few names are omitted."""
    from forecast.training import _pearson

    rows: list[tuple[int, float]] = []
    for key in np.unique(dates):
        sel = dates == key
        if int(sel.sum()) < int(min_names):
            continue
        rows.append((int(key), _pearson(pred[sel], target[sel])))
    if not rows:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _one_date_design(
    xd: np.ndarray,
    yd: np.ndarray,
    *,
    min_names: int,
    cs_demean: bool,
    cs_zscore: bool,
    rank_target: bool,
    feature_mask_bool: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(xd.shape[0])
    if n < int(min_names):
        return None
    x = np.asarray(xd, dtype=np.float64).copy()
    y = np.asarray(yd, dtype=np.float64).copy()
    if feature_mask_bool is not None:
        mask = np.asarray(feature_mask_bool, dtype=bool)
        x[:, ~mask] = 0.0
    if rank_target:
        order = np.argsort(y, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        y = ranks - ranks.mean()
        if n > 1:
            y = y / max(float(ranks.std()), 1e-8)
    if cs_demean or cs_zscore:
        x = x - x.mean(axis=0, keepdims=True)
        if not rank_target:
            y = y - y.mean()
    if cs_zscore:
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        x = x / std
    return x, y


def walk_forward_predict(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    score_dates: np.ndarray,
    lookback_days: int | None,
    ridge: float = 1.0,
    min_names: int = 8,
    cs_demean: bool = True,
    cs_zscore: bool = False,
    rank_target: bool = False,
    feature_mask_bool: np.ndarray | None = None,
    date_halflife: float = 0.0,
    min_train_dates: int = 60,
) -> np.ndarray:
    """Causal scores for ``score_dates``: fit on ``dates < d`` (and ``>= d - lookback``).

    Same-day ``y`` is never in the design. ``lookback_days=None`` is expanding.
    Recency ``date_halflife`` is ignored here (window membership is the clock).
    """
    del date_halflife  # walk-forward uses a hard lookback, not exp weights
    pred = np.full(x.shape[0], np.nan, dtype=np.float64)
    f = int(x.shape[1])
    uniq_score = set(int(k) for k in np.unique(np.asarray(score_dates, dtype=np.int64)))
    uniq_all = [int(k) for k in np.unique(dates)]
    by_date_x: dict[int, np.ndarray] = {}
    by_date_y: dict[int, np.ndarray] = {}
    by_date_idx: dict[int, np.ndarray] = {}
    contrib: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key in uniq_all:
        sel = np.flatnonzero(dates == key)
        by_date_x[key] = x[sel]
        by_date_y[key] = y[sel]
        by_date_idx[key] = sel
        des = _one_date_design(
            x[sel],
            y[sel],
            min_names=min_names,
            cs_demean=cs_demean,
            cs_zscore=cs_zscore,
            rank_target=rank_target,
            feature_mask_bool=feature_mask_bool,
        )
        if des is None:
            continue
        xd, yd = des
        contrib[key] = (xd.T @ xd, xd.T @ yd)

    acc_xtx = np.zeros((f, f), dtype=np.float64)
    acc_xty = np.zeros((f,), dtype=np.float64)
    lam = max(0.0, float(ridge))
    eye = np.eye(f, dtype=np.float64)
    hist_keys: list[int] = []
    cursor = 0
    n_fit_dates = 0

    def _add(key: int) -> None:
        nonlocal n_fit_dates, acc_xtx, acc_xty
        block = contrib.get(key)
        if block is None:
            hist_keys.append(key)
            return
        acc_xtx = acc_xtx + block[0]
        acc_xty = acc_xty + block[1]
        hist_keys.append(key)
        n_fit_dates += 1

    def _pop() -> None:
        nonlocal n_fit_dates, acc_xtx, acc_xty
        key = hist_keys.pop(0)
        block = contrib.get(key)
        if block is None:
            return
        acc_xtx = acc_xtx - block[0]
        acc_xty = acc_xty - block[1]
        n_fit_dates -= 1

    for d in uniq_all:
        while cursor < len(uniq_all) and uniq_all[cursor] < d:
            _add(uniq_all[cursor])
            cursor += 1
        if lookback_days is not None and lookback_days > 0:
            lo = d - int(lookback_days)
            while hist_keys and hist_keys[0] < lo:
                _pop()
        if d not in uniq_score:
            continue
        if n_fit_dates < int(min_train_dates):
            continue
        try:
            weights = np.linalg.solve(acc_xtx + lam * eye, acc_xty)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(acc_xtx + lam * eye, acc_xty, rcond=None)[0]
        idx = by_date_idx[d]
        xd = by_date_x[d].copy()
        if feature_mask_bool is not None:
            mask = np.asarray(feature_mask_bool, dtype=bool)
            xd[:, ~mask] = 0.0
        pred[idx] = xd @ weights
    return pred


def ridge_kwargs_from_train_cfg(train_cfg: Any, bundle: dict[str, Any]) -> dict[str, Any]:
    mask_mode = str(getattr(train_cfg, "ridge_features", "all") or "all")
    return {
        "ridge": float(getattr(train_cfg, "ridge_skip", 1.0)),
        "min_names": int(bundle.get("cs_min_names", 8)),
        "cs_demean": bool(getattr(train_cfg, "ridge_cs_demean", True)),
        "cs_zscore": bool(getattr(train_cfg, "ridge_cs_zscore", False)),
        "rank_target": bool(getattr(train_cfg, "ridge_rank_target", False)),
        "feature_mask_bool": feature_mask(mask_mode),
        "date_halflife": float(getattr(train_cfg, "ridge_date_halflife", 0.0) or 0.0),
    }
