"""Quantile long-short backtest math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import (
    book_pnl,
    causal_cc_dispersion,
    causal_disp_series,
    name_membership_churn,
    quantile_weights,
    rank_weights,
    resize_long_only,
    sticky_long_step,
    trailing_mean_cs_ic,
    trailing_on_resid_dispersion,
    weekday_mask_is_flat,
)


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


def test_trailing_mean_cs_ic_ignores_same_day_label():
    dates = pd.bdate_range("2022-01-03", periods=40)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (40, 1)), index=dates, columns=names
    )
    realized = pred * 0.02
    trail = trailing_mean_cs_ic(pred, realized, window=10, min_names=5, min_obs=5)
    y2 = realized.copy()
    y2.loc[dates[-1]] = -pred.loc[dates[-1]] * 0.5
    trail2 = trailing_mean_cs_ic(pred, y2, window=10, min_names=5, min_obs=5)
    assert trail.equals(trail2) or np.allclose(trail, trail2, equal_nan=True)
    # First dates are warmup (NaN); a later date is finite and uses only prior ICs.
    later = trail.dropna()
    assert len(later) > 0
    assert not np.isfinite(trail.iloc[0])


def test_ic_gate_flattens_when_trailing_ic_is_dead():
    dates = pd.bdate_range("2022-01-03", periods=50)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (50, 1)), index=dates, columns=names
    )
    realized = pred.copy()
    realized.iloc[:25] = pred.iloc[:25] * 0.02
    realized.iloc[25:] = -pred.iloc[25:] * 0.02
    always = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        long_only=True,
        min_names=8,
        round_trip_bps=10.0,
    )
    gated = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        long_only=True,
        min_names=8,
        round_trip_bps=10.0,
        ic_gate_window=12,
        ic_gate_tau=0.0,
    )
    assert gated["ic_gate_window"] == 12
    assert gated["ic_gate_n_flat"] > 0
    assert gated["ic_gate_coverage"] < 1.0
    assert gated["ic_gate_coverage"] > 0.2
    # Dead second half should cut some nights vs always-on path length still kept.
    assert gated["n_dates"] == always["n_dates"]


def _lo_panel(n_days: int = 40):
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (n_days, 1)), index=dates, columns=names
    )
    realized = pred * 0.02
    return dates, pred, realized


def _overnight_lo(**kwargs):
    pred = kwargs.pop("pred")
    realized = kwargs.pop("realized")
    return book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        long_only=True,
        min_names=8,
        round_trip_bps=10.0,
        **kwargs,
    )


def test_weekday_mask_friday_and_weekend_only():
    dates, pred, realized = _lo_panel(40)
    always = _overnight_lo(pred=pred, realized=realized, weekday_mask="always")
    skip_fr = _overnight_lo(pred=pred, realized=realized, weekday_mask="flat_friday")
    only_we = _overnight_lo(pred=pred, realized=realized, weekday_mask="weekend_only")
    skip_mo = _overnight_lo(pred=pred, realized=realized, weekday_mask="flat_monday")
    n_fri = int(sum(int(pd.Timestamp(t).dayofweek) == 4 for t in dates))
    n_mon = int(sum(int(pd.Timestamp(t).dayofweek) == 0 for t in dates))
    assert skip_fr["n_dates"] == always["n_dates"]
    assert skip_fr["weekday_mask"] == "flat_friday"
    assert skip_fr["weekday_n_flat"] == n_fri
    assert skip_fr["weekday_coverage"] == pytest.approx(1.0 - n_fri / 40)
    assert only_we["weekday_n_flat"] == 40 - n_fri
    assert only_we["weekday_coverage"] == pytest.approx(n_fri / 40)
    assert only_we["weekday_coverage"] < 0.30
    assert skip_mo["weekday_n_flat"] == n_mon
    w_fr = skip_fr["weights"]
    for ts in dates:
        if int(pd.Timestamp(ts).dayofweek) == 4:
            assert float(w_fr.loc[ts].abs().sum()) == pytest.approx(0.0)
        else:
            assert float(w_fr.loc[ts].abs().sum()) > 0.0


def test_weekday_mask_is_causal_calendar_only():
    dates, pred, realized = _lo_panel(30)
    y2 = realized.copy()
    y2.loc[dates[-1]] = -pred.loc[dates[-1]]
    a = _overnight_lo(pred=pred, realized=realized, weekday_mask="flat_friday")
    b = _overnight_lo(pred=pred, realized=y2, weekday_mask="flat_friday")
    assert a["weekday_n_flat"] == b["weekday_n_flat"]
    assert weekday_mask_is_flat(pd.Timestamp("2022-01-07"), "flat_friday")  # Friday
    assert not weekday_mask_is_flat(pd.Timestamp("2022-01-06"), "flat_friday")
    assert weekday_mask_is_flat(pd.Timestamp("2022-01-06"), "weekend_only")
    assert not weekday_mask_is_flat(pd.Timestamp("2022-01-07"), "weekend_only")
    assert weekday_mask_is_flat(pd.Timestamp("2022-01-03"), "flat_monday")


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


def test_causal_cc_dispersion_ignores_next_open_and_same_day_overnight():
    dates = pd.bdate_range("2022-01-03", periods=30)
    names = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(
        100.0 + np.linspace(0, 2, 30)[:, None] + np.linspace(0, 1, 8),
        index=dates,
        columns=names,
    )
    y = pd.DataFrame(0.01, index=dates, columns=names)
    cc = causal_cc_dispersion(close, min_names=5, window=1)
    y2 = y.copy()
    y2.loc[dates[-1]] = 9.0
    cc2 = causal_disp_series("cc", close=close, resid=y2, window=1, min_names=5)
    assert np.allclose(cc, cc2, equal_nan=True)
    on = trailing_on_resid_dispersion(y, window=5, min_names=5)
    on2 = trailing_on_resid_dispersion(y2, window=5, min_names=5)
    # Date t's own overnight residual must not enter the t decision.
    assert np.allclose(on, on2, equal_nan=True)
    assert not np.isfinite(on.iloc[0])


def test_disp_gate_flattens_high_cs_chaos_nights():
    dates = pd.bdate_range("2022-01-03", periods=40)
    names = [f"S{i}" for i in range(10)]
    close = pd.DataFrame(100.0, index=dates, columns=names)
    rng = np.random.default_rng(0)
    for i, ts in enumerate(dates):
        shock = 0.08 if i >= 25 else 0.004
        r = rng.normal(scale=shock, size=10)
        close.loc[ts] = close.iloc[max(0, i - 1)] * np.exp(r) if i else 100.0 * np.exp(r)
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (40, 1)), index=dates, columns=names
    )
    realized = pred * 0.02
    trail = causal_cc_dispersion(close, min_names=5, window=1)
    tau = float(np.nanquantile(trail.to_numpy(dtype=np.float64), 0.7))
    always = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        long_only=True,
        min_names=8,
        round_trip_bps=10.0,
    )
    gated = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        long_only=True,
        min_names=8,
        round_trip_bps=10.0,
        disp_gate_trail=trail,
        disp_gate_tau=tau,
        disp_gate_kind="cc",
        disp_gate_window=1,
        close_px=close,
    )
    assert gated["disp_gate_kind"] == "cc"
    assert gated["disp_gate_n_flat"] > 0
    assert gated["disp_gate_coverage"] < 1.0
    assert gated["disp_gate_coverage"] >= 0.30
    assert gated["n_dates"] == always["n_dates"]


def test_sticky_long_step_keeps_mid_rank_and_drops_below_exit():
    names = [f"S{i}" for i in range(10)]
    scores = pd.Series(np.linspace(-1.0, 1.0, 10), index=names)
    # top 20% = S8,S9; top 40% = S6..S9
    w0, held0 = sticky_long_step(
        scores, set(), q_enter=0.20, q_exit=0.40, min_names=8
    )
    assert set(held0) == {"S8", "S9"}
    assert w0[w0 > 0].sum() == pytest.approx(1.0)

    # S7 is 4th from top (in exit band, not enter). Keep if already held.
    w1, held1 = sticky_long_step(
        scores, {"S7"}, q_enter=0.20, q_exit=0.40, min_names=8
    )
    assert "S7" in held1 and "S8" in held1 and "S9" in held1
    assert w1["S7"] == pytest.approx(1.0 / 3.0)

    # S5 is 6th from top — below exit band. Drop.
    w2, held2 = sticky_long_step(
        scores, {"S5"}, q_enter=0.20, q_exit=0.40, min_names=8
    )
    assert "S5" not in held2
    assert set(held2) == {"S8", "S9"}
    assert float(w2.get("S5", 0.0)) == pytest.approx(0.0)


def test_sticky_book_churns_less_than_q20_rebuild_when_ranks_persist():
    dates = pd.bdate_range("2022-01-03", periods=40)
    names = [f"S{i}" for i in range(10)]
    # Slow rank rotation: each name stays near its rank most nights.
    base = np.linspace(-1.0, 1.0, 10)
    pred = pd.DataFrame(
        [base + 0.05 * np.sin(i / 4.0 + np.linspace(0, 0.4, 10)) for i in range(40)],
        index=dates,
        columns=names,
    )
    realized = pred * 0.02
    q20 = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        long_only=True,
        min_names=8,
        round_trip_bps=10.0,
    )
    sticky = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        long_only=True,
        min_names=8,
        round_trip_bps=10.0,
        sticky_q_enter=0.15,
        sticky_q_exit=0.40,
    )
    assert sticky["sticky_q_enter"] == pytest.approx(0.15)
    assert sticky["mean_name_churn"] < q20["mean_name_churn"]
    assert sticky["sticky_coverage"] >= 0.30
    assert sticky["n_dates"] == q20["n_dates"]
    # Overnight flatten still charges ~1.0 one-way; name churn is the diagnostic.
    assert q20["mean_turnover"] == pytest.approx(1.0, abs=0.05)


def test_name_membership_churn_is_zero_when_holdings_never_change():
    dates = pd.bdate_range("2022-01-03", periods=5)
    w = pd.DataFrame(
        [[0.5, 0.5, 0.0, 0.0]] * 5,
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    out = name_membership_churn(w)
    # First night: enter A,B from empty → 2/4. Later nights: 0.
    assert out["mean_name_churn"] == pytest.approx((0.5 + 0 + 0 + 0 + 0) / 5)
    assert out["mean_n_held"] == pytest.approx(2.0)
    assert out["sticky_coverage"] == pytest.approx(1.0)

