"""Quantile long-short backtest math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import book_pnl, quantile_weights


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


def test_book_pnl_perfect_ranks_has_positive_net_ir():
    dates = pd.bdate_range("2022-01-03", periods=80)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (80, 1)), index=dates, columns=names
    )
    noise = np.random.default_rng(1).normal(scale=0.05, size=pred.shape)
    realized = pred + noise
    stats = book_pnl(pred, realized, quantile=0.2, round_trip_bps=10.0, vol_target=1.0)
    assert stats["n_dates"] == 80
    assert stats["mean_cs_ic"] > 0.8
    assert stats["net_ir"] > 2.0
    assert 0.0 <= stats["hit_rate"] <= 1.0
    assert stats["max_dd"] <= 0.0
