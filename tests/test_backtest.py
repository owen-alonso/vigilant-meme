"""Quantile long-short backtest math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import book_pnl, quantile_weights, rank_weights, resize_long_only


def test_quantile_weights_two_name_spread():
    s = pd.Series({"AAA": 0.2, "BBB": -0.1})
    w = quantile_weights(s, quantile=0.2)
    assert w["AAA"] == pytest.approx(0.5)
    assert w["BBB"] == pytest.approx(-0.5)


def test_quantile_weights_long_short_equal_notional():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=20), index=[f"S{i}" for i in range(20)])
    w = quantile_weights(s, quantile=0.2)
    assert w[w > 0].sum() == pytest.approx(0.5)
    assert w[w < 0].sum() == pytest.approx(-0.5)


def test_resize_long_only_conf_drops_low_abs_pred():
    scores = pd.Series({"A": 0.01, "B": 1.0, "C": 0.8, "D": -0.5})
    w = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0})
    out = resize_long_only(w, scores, long_size="equal", conf_pctile=0.5)
    assert out["A"] == pytest.approx(0.0)
    assert out["B"] == pytest.approx(1.0)
    assert out.sum() == pytest.approx(1.0)


def test_resize_long_only_inv_vol_prefers_quiet_names():
    scores = pd.Series({"A": 1.0, "B": 1.0})
    w = pd.Series({"A": 0.5, "B": 0.5})
    vol = pd.Series({"A": 0.4, "B": 0.1})
    out = resize_long_only(w, scores, vol=vol, long_size="inv_vol")
    assert out["B"] > out["A"]
    assert out.sum() == pytest.approx(1.0)


def test_quantile_weights_long_only_sums_to_one():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=20), index=[f"S{i}" for i in range(20)])
    w = quantile_weights(s, quantile=0.2, long_only=True)
    assert w[w > 0].sum() == pytest.approx(1.0)
    assert float((w < 0).sum()) == 0


def test_book_pnl_skips_thin_dates():
    dates = pd.bdate_range("2022-01-03", periods=20)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (20, 1)), index=dates, columns=names
    )
    pred.iloc[:5, 3:] = np.nan
    realized = pred.copy()
    stats = book_pnl(pred, realized, quantile=0.2, min_names=8, round_trip_bps=0.0)
    assert stats["n_dates"] == 15


def test_book_pnl_perfect_ranks_has_positive_net_ir():
    dates = pd.bdate_range("2022-01-03", periods=80)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (80, 1)), index=dates, columns=names
    )
    noise = np.random.default_rng(1).normal(scale=0.05, size=pred.shape)
    realized = pred + noise
    stats = book_pnl(
        pred,
        realized,
        quantile=0.2,
        round_trip_bps=10.0,
        vol_target=1.0,
        weighting="quantile",
        hold_halflife=0.0,
        causal_vol=False,
    )
    assert stats["n_dates"] == 80
    assert stats["mean_cs_ic"] > 0.8
    assert stats["net_ir"] > 2.0
    assert 0.0 <= stats["hit_rate"] <= 1.0
    assert stats["max_dd"] <= 0.0


def test_rank_weights_are_dollar_neutral():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=20), index=[f"S{i}" for i in range(20)])
    w = rank_weights(s)
    assert w.sum() == pytest.approx(0.0, abs=1e-9)
    assert w.abs().sum() == pytest.approx(1.0)


def test_hold_smoothing_cuts_turnover():
    dates = pd.bdate_range("2022-01-03", periods=60)
    names = [f"S{i}" for i in range(12)]
    rng = np.random.default_rng(2)
    pred = pd.DataFrame(rng.normal(size=(60, 12)), index=dates, columns=names)
    realized = pd.DataFrame(rng.normal(scale=0.02, size=(60, 12)), index=dates, columns=names)
    raw = book_pnl(
        pred,
        realized,
        weighting="quantile",
        hold_halflife=0.0,
        causal_vol=False,
        vol_target=0.0,
        round_trip_bps=10.0,
    )
    held = book_pnl(
        pred,
        realized,
        weighting="quantile",
        hold_halflife=5.0,
        causal_vol=False,
        vol_target=0.0,
        round_trip_bps=10.0,
    )
    assert held["mean_turnover"] < raw["mean_turnover"]


def test_causal_vol_lever_ignores_future_returns():
    dates = pd.bdate_range("2022-01-03", periods=80)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (80, 1)), index=dates, columns=names
    )
    realized = pred * 0.2
    flipped = realized.copy()
    flipped.iloc[-20:] *= -8.0
    kwargs = dict(
        weighting="quantile",
        hold_halflife=0.0,
        causal_vol=True,
        vol_target=0.15,
        lever_cap=3.0,
        round_trip_bps=0.0,
        min_vol_days=21,
    )
    a = book_pnl(pred, realized, **kwargs)
    b = book_pnl(pred, flipped, **kwargs)
    mid = 40
    assert float(a["leverage"].iloc[mid]) == pytest.approx(float(b["leverage"].iloc[mid]), rel=1e-8)
    full_a = book_pnl(pred, realized, **{**kwargs, "causal_vol": False})
    full_b = book_pnl(pred, flipped, **{**kwargs, "causal_vol": False})
    assert abs(full_a["lever"] - full_b["lever"]) > 1e-6


def test_softer_vol_target_shrinks_max_dd_not_unlevered_ir():
    dates = pd.bdate_range("2022-01-03", periods=80)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (80, 1)), index=dates, columns=names
    )
    realized = pred * 0.15
    realized.iloc[:12] = -pred.iloc[:12] * 0.4
    hot = book_pnl(
        pred,
        realized,
        weighting="quantile",
        hold_halflife=0.0,
        causal_vol=False,
        vol_target=1.0,
        lever_cap=0.0,
        round_trip_bps=0.0,
    )
    cool = book_pnl(
        pred,
        realized,
        weighting="quantile",
        hold_halflife=0.0,
        causal_vol=False,
        vol_target=0.15,
        lever_cap=0.0,
        round_trip_bps=0.0,
    )
    assert cool["unlevered_gross_ir"] == pytest.approx(hot["unlevered_gross_ir"], rel=1e-8)
    assert hot["max_dd"] < 0
    assert abs(cool["max_dd"]) == pytest.approx(0.15 * abs(hot["max_dd"]), rel=1e-6)

