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
