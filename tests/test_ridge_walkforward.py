"""Causal walk-forward ridge and ranking-target skip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from forecast.ridge import (
    feature_mask,
    fit_ridge_xy,
    walk_forward_predict,
)
from forecast.training import masked_listnet_loss


def test_walk_forward_does_not_use_same_day_label():
    rng = np.random.default_rng(0)
    n_dates, n_names, f = 40, 10, 4
    dates = np.repeat(np.arange(n_dates, dtype=np.int64) * 2, n_names)
    x = rng.normal(size=(n_dates * n_names, f))
    y = x[:, 0] + 0.1 * rng.normal(size=n_dates * n_names)
    pred = walk_forward_predict(
        x,
        y,
        dates,
        score_dates=np.unique(dates),
        lookback_days=None,
        ridge=1e-3,
        min_names=8,
        min_train_dates=5,
    )
    y2 = y.copy()
    last = dates == dates.max()
    y2[last] = rng.normal(size=int(last.sum()))
    pred2 = walk_forward_predict(
        x,
        y2,
        dates,
        score_dates=np.unique(dates),
        lookback_days=None,
        ridge=1e-3,
        min_names=8,
        min_train_dates=5,
    )
    earlier = dates < dates.max()
    assert np.allclose(pred[earlier], pred2[earlier], equal_nan=True)
    # Last date's scores may change (they use earlier y, not last y).
    assert np.allclose(pred[last], pred2[last], equal_nan=True, atol=1e-10)


def test_rank_target_ridge_tracks_within_date_order():
    n_dates, n_names, f = 20, 12, 3
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    rng = np.random.default_rng(1)
    signal = rng.normal(size=n_dates * n_names)
    x = rng.normal(size=(n_dates * n_names, f))
    x[:, 1] = signal
    y = np.exp(signal)  # monotone but not linear
    w_rank, _, ic_rank = fit_ridge_xy(
        x, y, dates, ridge=1e-3, min_names=8, rank_target=True, cs_demean=True
    )
    w_val, _, ic_val = fit_ridge_xy(
        x, y, dates, ridge=1e-3, min_names=8, rank_target=False, cs_demean=True
    )
    assert abs(w_rank[1]) > abs(w_rank[0])
    assert ic_rank > 0.5
    # Rank target is allowed to beat raw-value ridge on a nonlinear monotone y.
    assert ic_rank + 1e-6 >= ic_val - 0.2


def test_cs_feature_mask_zeros_calendar_columns():
    from forecast.data import CROSS_SECTION_FEATURES, FEATURE_NAMES

    mask = feature_mask("cs")
    assert int(mask.sum()) == len(CROSS_SECTION_FEATURES)
    assert int(mask.sum()) < len(FEATURE_NAMES)
    cal = feature_mask("no_calendar")
    assert int(cal.sum()) < cal.size


def test_listnet_lower_when_ranks_agree():
    pred = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    y = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    mask = torch.ones_like(pred)
    dates = torch.tensor([[10, 10, 10], [11, 11, 11]])
    loss = float(masked_listnet_loss(pred, y, mask, date_ids=dates))
    aligned = pred.clone()
    aligned[1] = torch.tensor([1.0, 2.0, 3.0])
    loss_ok = float(masked_listnet_loss(aligned, y, mask, date_ids=dates))
    assert loss_ok < loss


def test_drop_crashes_ignores_labels_inside_window():
    from forecast.ridge import CRASH_WINDOWS, _ymd_to_days, fit_ridge_xy

    crash_day = _ymd_to_days(CRASH_WINDOWS[0][0]) + 5
    safe_day = _ymd_to_days(CRASH_WINDOWS[0][0]) - 40
    n_names, f = 12, 3
    dates = np.array([safe_day] * n_names + [crash_day] * n_names, dtype=np.int64)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2 * n_names, f))
    y = x[:, 0].copy()
    y_flip = y.copy()
    y_flip[dates == crash_day] = -x[dates == crash_day, 0]
    w0, _, _ = fit_ridge_xy(
        x, y, dates, ridge=1e-3, min_names=8, rank_target=True, drop_crashes=True
    )
    w1, _, _ = fit_ridge_xy(
        x, y_flip, dates, ridge=1e-3, min_names=8, rank_target=True, drop_crashes=True
    )
    assert np.allclose(w0, w1, atol=1e-6)


def test_sign_constrain_matches_univariate_sign():
    from forecast.ridge import univariate_cs_ics

    n_dates, n_names, f = 30, 12, 3
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    rng = np.random.default_rng(3)
    good = rng.normal(size=n_dates * n_names)
    x = rng.normal(size=(n_dates * n_names, f))
    x[:, 0] = good
    x[:, 1] = -good
    y = good + 0.05 * rng.normal(size=n_dates * n_names)
    w, _, ic = fit_ridge_xy(
        x, y, dates, ridge=1e-2, min_names=8, rank_target=True, sign_constrain=True
    )
    uni = univariate_cs_ics(x, y, dates, min_names=8)
    for i in range(f):
        if abs(float(uni[i])) >= 0.003:
            assert float(w[i]) * float(uni[i]) >= -1e-8
    assert ic > 0.4


def test_listnet_linear_tracks_rank_feature():
    from forecast.ridge import fit_listnet_xy

    n_dates, n_names, f = 16, 12, 3
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    rng = np.random.default_rng(4)
    signal = rng.normal(size=n_dates * n_names)
    x = rng.normal(size=(n_dates * n_names, f))
    x[:, 2] = signal
    y = np.exp(signal)
    w, _, ic = fit_listnet_xy(
        x, y, dates, ridge=1.0, min_names=8, rank_target=True, steps=80, lr=0.1
    )
    assert abs(float(w[2])) > abs(float(w[0]))
    assert ic > 0.4


def test_new_feature_masks_are_stricter():
    from forecast.ridge import feature_mask

    all_m = feature_mask("all")
    core = feature_mask("core")
    no_long = feature_mask("no_long_ts")
    assert int(core.sum()) < int(no_long.sum()) < int(all_m.sum())


def test_year_cs_ics_splits_calendar_years():
    from forecast.ridge import year_cs_ics

    d0 = int((np.datetime64("2018-06-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    d1 = int((np.datetime64("2019-06-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    dates = np.array([d0] * 8 + [d1] * 8, dtype=np.int64)
    pred = np.linspace(-1, 1, 16)
    y = pred.copy()
    rows = year_cs_ics(pred, y, dates, min_names=3)
    years = {int(r["year"]) for r in rows}
    assert years == {2018, 2019}
    assert all(r["cs_ic"] > 0.9 for r in rows)
