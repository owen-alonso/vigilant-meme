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
    from forecast.data import CS_PRODUCT_FEATURES, FEATURE_NAMES, VOL_FEATURES
    from forecast.ridge import feature_mask

    all_m = feature_mask("all")
    core = feature_mask("core")
    no_long = feature_mask("no_long_ts")
    no_vol = feature_mask("no_vol_products")
    names = list(FEATURE_NAMES)
    assert int(core.sum()) < int(no_long.sum()) < int(all_m.sum())
    assert int(no_vol.sum()) < int(no_long.sum())
    for col in VOL_FEATURES:
        assert not bool(no_vol[names.index(col)])
    for col in CS_PRODUCT_FEATURES:
        assert not bool(no_vol[names.index(col)])


def test_year_balance_equalizes_short_years():
    from forecast.ridge import fit_ridge_xy

    # Many dates in year 0, few in year 1. Signal flips in the short year.
    n_names, f = 12, 2
    d0 = int((np.datetime64("2000-06-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    d1 = int((np.datetime64("2001-06-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    dates0 = np.repeat(np.arange(20, dtype=np.int64) + d0, n_names)
    dates1 = np.repeat(np.arange(3, dtype=np.int64) + d1, n_names)
    dates = np.concatenate([dates0, dates1])
    rng = np.random.default_rng(7)
    x = rng.normal(size=(dates.size, f))
    sig = rng.normal(size=dates.size)
    x[:, 0] = sig
    y = sig.copy()
    y[dates >= d1] = -sig[dates >= d1]
    w_u, _, _ = fit_ridge_xy(x, y, dates, ridge=1e-2, min_names=8, rank_target=True)
    w_b, _, _ = fit_ridge_xy(
        x, y, dates, ridge=1e-2, min_names=8, rank_target=True, year_balance=True
    )
    # Balancing the short opposite year should shrink the year-0 weight.
    assert abs(float(w_b[0])) < abs(float(w_u[0]))


def test_year_stable_mask_drops_flipping_column():
    from forecast.ridge import year_stable_mask

    n_names, f = 12, 2
    dates = []
    x_rows = []
    y_rows = []
    rng = np.random.default_rng(8)
    for year, x1_sign in ((2000, 1.0), (2001, 1.0), (2002, -1.0), (2003, -1.0)):
        d0 = int((np.datetime64(f"{year}-03-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
        for k in range(8):
            dates.extend([d0 + k] * n_names)
            good = rng.normal(size=n_names)
            x_rows.append(np.stack([good, x1_sign * good], axis=1))
            y_rows.append(good + 0.05 * rng.normal(size=n_names))
    x = np.concatenate(x_rows)
    y = np.concatenate(y_rows)
    d = np.asarray(dates, dtype=np.int64)
    keep = year_stable_mask(x, y, d, min_names=8, min_frac=0.7, min_abs=0.01)
    assert bool(keep[0])
    assert not bool(keep[1])


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


def test_trailing_vol_is_causal():
    from forecast.ridge import trailing_realized_vol

    keys = np.arange(40, dtype=np.int64)
    rets = np.linspace(-0.02, 0.03, 40)
    vol = trailing_realized_vol(keys, rets, window=10, min_obs=5)
    rets2 = rets.copy()
    rets2[-1] = 0.5
    vol2 = trailing_realized_vol(keys, rets2, window=10, min_obs=5)
    for k in keys[:-1]:
        if k in vol and k in vol2:
            assert vol[int(k)] == pytest.approx(vol2[int(k)])
    assert vol[int(keys[-1])] != pytest.approx(vol2[int(keys[-1])])


def test_regime_bucket_edges_ignore_held_out_scores():
    from forecast.ridge import assign_score_buckets, bucket_edges_from_train

    train = {i: float(i) for i in range(10)}
    edges = bucket_edges_from_train(train, 2)
    held = {100: 1e9}
    b_train = assign_score_buckets(train, edges)
    b_all = assign_score_buckets({**train, **held}, edges)
    for k, v in b_train.items():
        assert b_all[k] == v
    assert b_all[100] == 1


def test_regime_heads_recover_dispersion_sign_flip():
    from forecast.ridge import (
        assign_score_buckets,
        bucket_edges_from_train,
        fit_regime_heads,
        fit_ridge_xy,
        mean_cs_ic,
        predict_regime_heads,
        regime_score_map,
    )

    n_dates, n_names = 80, 12
    names = ["ret_1", "sig", "mkt_ret_1", "noise"]
    rng = np.random.default_rng(11)
    rows_x = []
    rows_y = []
    dates = []
    for d in range(n_dates):
        high = d >= 40
        ret1 = rng.normal(scale=3.0 if high else 0.25, size=n_names)
        sig = rng.normal(size=n_names)
        mkt = np.full(n_names, 0.01 if high else 0.0)
        noise = rng.normal(size=n_names)
        rows_x.append(np.stack([ret1, sig, mkt, noise], axis=1))
        y = (-sig if high else sig) + 0.05 * rng.normal(size=n_names)
        rows_y.append(y)
        dates.extend([d] * n_names)
    x = np.concatenate(rows_x)
    y = np.concatenate(rows_y)
    d = np.asarray(dates, dtype=np.int64)
    scores = regime_score_map(x, d, kind="cs_disp", names=names)
    edges = bucket_edges_from_train(scores, 2)
    buckets = assign_score_buckets(scores, edges)
    w_heads, _, counts = fit_regime_heads(
        x,
        y,
        d,
        buckets,
        n_buckets=2,
        ridge=1e-3,
        min_names=8,
        rank_target=True,
        feat_winsor=0.0,
        min_dates_per_bucket=10,
    )
    pred = predict_regime_heads(x, d, buckets, w_heads)
    w_one, _, _ = fit_ridge_xy(x, y, d, ridge=1e-3, min_names=8, rank_target=True)
    ic_reg = mean_cs_ic(pred, y, d, min_names=8)
    ic_one = mean_cs_ic(x @ w_one.astype(np.float64), y, d, min_names=8)
    assert min(counts) >= 10
    assert ic_reg > 0.4
    assert ic_reg > ic_one + 0.15


def test_trailing_window_keep_is_left_closed():
    from forecast.ridge import trailing_window_keep

    end = int((np.datetime64("2014-01-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    dates = np.array([end - 200, end - 10, end, end + 10], dtype=np.int64)
    keep = trailing_window_keep(dates, end_days=end, years=1.0)
    assert bool(keep[0]) and bool(keep[1])
    assert not bool(keep[2]) and not bool(keep[3])


def test_late_train_holdout_is_disjoint_and_last():
    from forecast.ridge import late_train_holdout_mask

    d0 = int((np.datetime64("2008-01-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    dates = np.repeat(np.arange(2000, dtype=np.int64) + d0, 4)
    fit, sel = late_train_holdout_mask(dates, holdout_years=2, min_holdout_dates=60)
    assert not bool((fit & sel).any())
    assert bool(fit.any()) and bool(sel.any())
    assert int(dates[sel].min()) > int(dates[fit].max())


def test_trailing_skip_ic_does_not_use_same_day_label():
    from forecast.ridge import apply_skip_ic_shrink, trailing_skip_ic_stats

    n_dates, n_names = 80, 12
    rng = np.random.default_rng(12)
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    pred = rng.normal(size=n_dates * n_names)
    y = pred + 0.3 * rng.normal(size=n_dates * n_names)
    keys, mu, tt, _n = trailing_skip_ic_stats(
        pred, y, dates, lookback_days=20, min_names=8, min_obs=8
    )
    y2 = y.copy()
    last = dates == dates.max()
    y2[last] = rng.normal(size=int(last.sum()))
    keys2, mu2, tt2, _n2 = trailing_skip_ic_stats(
        pred, y2, dates, lookback_days=20, min_names=8, min_obs=8
    )
    assert np.array_equal(keys, keys2)
    assert np.allclose(mu, mu2, equal_nan=True)
    assert np.allclose(tt, tt2, equal_nan=True)
    # Yesterday's IC is in today's window; mutating the last date must not
    # change shrink factors on any date, including the last.
    shrunk = apply_skip_ic_shrink(
        pred, dates, date_keys=keys, trailing_ic=mu, trailing_t=tt, train_ic=0.2, mode="scale"
    )
    shrunk2 = apply_skip_ic_shrink(
        pred, dates, date_keys=keys2, trailing_ic=mu2, trailing_t=tt2, train_ic=0.2, mode="scale"
    )
    assert np.allclose(shrunk, shrunk2, equal_nan=True)


def test_trailing_skip_ic_window_is_left_closed():
    from forecast.ridge import trailing_skip_ic_stats
    from forecast.training import _pearson

    n_names = 12
    # Three dates. Date 0 IC = +1, date 1 IC = -1, date 2 unused.
    pred = np.concatenate(
        [
            np.linspace(-1, 1, n_names),
            np.linspace(-1, 1, n_names),
            np.linspace(-1, 1, n_names),
        ]
    )
    y = np.concatenate(
        [
            np.linspace(-1, 1, n_names),
            np.linspace(1, -1, n_names),
            np.zeros(n_names),
        ]
    )
    dates = np.repeat(np.array([10, 20, 30], dtype=np.int64), n_names)
    keys, mu, _tt, n_obs = trailing_skip_ic_stats(
        pred, y, dates, lookback_days=15, min_names=8, min_obs=1
    )
    # Date 20 window is [5, 20) → only date 10, IC=+1.
    i20 = int(np.where(keys == 20)[0][0])
    i30 = int(np.where(keys == 30)[0][0])
    assert n_obs[i20] == 1
    assert mu[i20] == pytest.approx(1.0, abs=1e-6)
    # Date 30 window is [15, 30) → only date 20 (lookback 15), IC=-1.
    assert n_obs[i30] == 1
    assert mu[i30] == pytest.approx(-1.0, abs=1e-6)
    # Date 10 has no prior dates in window.
    i10 = int(np.where(keys == 10)[0][0])
    assert n_obs[i10] == 0
    assert not np.isfinite(mu[i10])
    assert _pearson(pred[:n_names], y[:n_names]) == pytest.approx(1.0, abs=1e-6)


def test_flatten_tstat_zeros_when_trailing_ic_is_dead():
    from forecast.ridge import apply_skip_ic_shrink, trailing_skip_ic_stats
    from forecast.training import mean_cs_stats

    n_dates, n_names = 60, 12
    rng = np.random.default_rng(13)
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    signal = rng.normal(size=n_dates * n_names)
    pred = signal.copy()
    y = signal.copy()
    # Second half: skip is anti-aligned.
    later = dates >= 30
    y[later] = -signal[later]
    keys, mu, tt, _n = trailing_skip_ic_stats(
        pred, y, dates, lookback_days=25, min_names=8, min_obs=8
    )
    flat = apply_skip_ic_shrink(
        pred,
        dates,
        date_keys=keys,
        trailing_ic=mu,
        trailing_t=tt,
        train_ic=0.5,
        mode="flatten_tstat",
    )
    # Late dates should see a negative trailing IC and flatten.
    late_pred = flat[dates == n_dates - 1]
    assert float(np.std(late_pred)) < 1e-12
    stats = mean_cs_stats(flat, y, dates, min_names=8, flat_as_zero=True)
    raw = mean_cs_stats(pred, y, dates, min_names=8, flat_as_zero=True)
    # Flattening the dead half should beat (or match) always-on skip.
    assert stats["cs_ic"] + 1e-9 >= raw["cs_ic"]
    assert stats["cs_n_flat"] > 0
