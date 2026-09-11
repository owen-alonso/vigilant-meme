"""Causal tests for overnight accuracy levers (no look-ahead)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import book_pnl, build_arg_parser, cost_kwargs_from_args
from forecast.config import DataConfig
from forecast.data import FEATURE_NAMES, attach_cross_section_features, attach_residual_target, compute_features
from forecast.levers import (
    blend_readouts,
    cap_name_risk,
    long_only_cs_target,
    sign_consistency_weights,
    sleeve_row_mask,
    trailing_leverage_scale,
    train_era_liquid_names,
    within_date_zscore,
)
from forecast.overnight import overnight_cost_breakdown, resolve_cost_bundle
from forecast.ridge import fit_ridge_xy, year_stable_mask
from forecast.training import build_arg_parser as train_parser
from forecast.training import configs_from_cli


def _ymd(year: int, month: int = 3, day: int = 1) -> int:
    return int(
        (np.datetime64(f"{year:04d}-{month:02d}-{day:02d}") - np.datetime64("1970-01-01"))
        / np.timedelta64(1, "D")
    )


def _daily_grid(n: int = 50, *, seed: int = 0, vol: float = 1_000_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    open_px = close * (1.0 + rng.normal(scale=0.002, size=n))
    return pd.DataFrame(
        {
            "datetime": dates,
            "session": pd.to_datetime(dates).normalize(),
            "mos": np.zeros(n, dtype=int),
            "open": open_px,
            "high": np.maximum(open_px, close) + 0.1,
            "low": np.minimum(open_px, close) - 0.1,
            "close": close,
            "volume": np.full(n, vol),
            "traded": np.ones(n),
        }
    )


def test_sign_consistency_shrinks_flipping_column_not_hard_drop():
    n_names, f = 12, 2
    dates = []
    x_rows = []
    y_rows = []
    rng = np.random.default_rng(8)
    for year, x1_sign in ((2000, 1.0), (2001, 1.0), (2002, -1.0), (2003, -1.0)):
        d0 = _ymd(year)
        for k in range(8):
            dates.extend([d0 + k] * n_names)
            good = rng.normal(size=n_names)
            x_rows.append(np.stack([good, x1_sign * good], axis=1))
            y_rows.append(good + 0.05 * rng.normal(size=n_names))
    x = np.concatenate(x_rows)
    y = np.concatenate(y_rows)
    d = np.asarray(dates, dtype=np.int64)
    w = sign_consistency_weights(x, y, d, min_names=8, min_abs=0.01)
    keep = year_stable_mask(x, y, d, min_names=8, min_frac=0.7, min_abs=0.01)
    assert w[0] > w[1]
    assert w[1] < 0.6
    assert w[1] > 0.0
    assert not bool(keep[1])


def test_sign_consistency_weights_ignore_held_out_year():
    n_names = 12
    rng = np.random.default_rng(3)
    dates = []
    x_rows = []
    y_rows = []
    for year, sign in ((2014, 1.0), (2015, 1.0), (2016, 1.0)):
        d0 = _ymd(year)
        for k in range(6):
            dates.extend([d0 + k] * n_names)
            good = rng.normal(size=n_names)
            x_rows.append(np.stack([good, sign * good], axis=1))
            y_rows.append(good)
    x = np.concatenate(x_rows)
    y = np.concatenate(y_rows)
    d = np.asarray(dates, dtype=np.int64)
    w = sign_consistency_weights(x, y, d, min_names=8)
    d0 = _ymd(2017)
    extra_d = np.repeat(np.arange(d0, d0 + 6), n_names)
    extra_good = rng.normal(size=extra_d.size)
    extra_x = np.stack([extra_good, -extra_good], axis=1)
    extra_y = extra_good
    w2 = sign_consistency_weights(
        np.concatenate([x, extra_x]),
        np.concatenate([y, extra_y]),
        np.concatenate([d, extra_d]),
        min_names=8,
    )
    w_train_only = sign_consistency_weights(x, y, d, min_names=8)
    assert np.allclose(w, w_train_only)
    assert not np.allclose(w, w2)


def test_long_only_target_zeros_bottom_quantile():
    y = np.linspace(-2, 2, 10)
    lo = long_only_cs_target(y, quantile=0.2)
    assert lo[0] == pytest.approx(0.0)
    assert lo[-1] > lo[-2] > 0
    assert float((lo > 0).sum()) == 2


def test_long_only_ridge_fits_top_sleeve_not_shorts():
    n_dates, n_names, f = 30, 12, 3
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    rng = np.random.default_rng(4)
    signal = rng.normal(size=n_dates * n_names)
    x = rng.normal(size=(n_dates * n_names, f))
    x[:, 0] = signal
    y = signal.copy()
    y[signal < np.median(signal)] = -5.0 * np.abs(signal[signal < np.median(signal)])
    w_ls, _, _ = fit_ridge_xy(
        x, y, dates, ridge=0.1, min_names=8, rank_target=True, cs_demean=True
    )
    w_lo, _, _ = fit_ridge_xy(
        x,
        y,
        dates,
        ridge=0.1,
        min_names=8,
        rank_target=True,
        cs_demean=True,
        long_only_quantile=0.3,
    )
    assert abs(float(w_lo[0])) > abs(float(w_lo[1]))
    assert np.isfinite(w_ls).all() and np.isfinite(w_lo).all()


def test_blend_readouts_is_within_date_zscore():
    dates = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    a = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    b = np.array([3.0, 2.0, 1.0, 30.0, 20.0, 10.0])
    out = blend_readouts([a, b], dates)
    z = within_date_zscore(a, dates)
    assert out[0] == pytest.approx(0.0, abs=1e-8)
    assert abs(out[2]) < abs(z[2])


def test_sleeve_row_mask_uses_same_date_turnover_only():
    dates = np.repeat(np.arange(4, dtype=np.int64), 6)
    tz = np.tile(np.linspace(-2, 2, 6), 4)
    keep = sleeve_row_mask(tz, dates, floor=2.0 / 3.0)
    assert int(keep[:6].sum()) == 2
    tz2 = tz.copy()
    tz2[-6:] = 99.0
    keep2 = sleeve_row_mask(tz2, dates, floor=2.0 / 3.0)
    assert np.array_equal(keep[:18], keep2[:18])


def test_train_era_liquid_names_lock_ignores_val_volume():
    dates = pd.bdate_range("2016-01-04", periods=80)
    train_end = dates[40]
    panels = {}
    for i, name in enumerate(("AAA", "BBB", "CCC")):
        vol = np.full(80, 1e6 * (i + 1))
        vol[40:] = 1e9 if name == "CCC" else vol[40:]
        close = np.full(80, 10.0)
        panels[name] = pd.DataFrame(
            {
                "session": dates,
                "datetime": dates,
                "close": close,
                "volume": vol,
            }
        )
    kept = train_era_liquid_names(panels, ["AAA", "BBB", "CCC"], train_end, pctile=0.5)
    assert "AAA" not in kept
    assert "CCC" in kept or "BBB" in kept
    # CCC's huge *val* volume must not change membership vs a copy with quiet val.
    panels["CCC"] = panels["CCC"].copy()
    panels["CCC"].loc[40:, "volume"] = 1.0
    kept2 = train_era_liquid_names(panels, ["AAA", "BBB", "CCC"], train_end, pctile=0.5)
    assert set(kept) == set(kept2)


def test_trailing_leverage_scale_does_not_use_same_day_label():
    n_dates, n_names = 40, 10
    dates = np.repeat(np.arange(n_dates, dtype=np.int64) * 2, n_names)
    rng = np.random.default_rng(1)
    pred = rng.normal(size=n_dates * n_names)
    y = pred * 0.4 + rng.normal(scale=0.5, size=pred.shape[0])
    _keys, scale = trailing_leverage_scale(
        pred, y, dates, lookback_days=10, train_ic=0.2, min_names=8, min_obs=5
    )
    y2 = y.copy()
    last = dates == dates.max()
    y2[last] = rng.normal(size=int(last.sum()))
    _k2, scale2 = trailing_leverage_scale(
        pred, y2, dates, lookback_days=10, train_ic=0.2, min_names=8, min_obs=5
    )
    earlier = dates < dates.max()
    assert np.allclose(scale[earlier], scale2[earlier], equal_nan=True)


def test_gap_risk_cap_renorm_ready():
    w = np.array([0.5, 0.4, 0.1])
    capped = cap_name_risk(w, max_weight=0.3, long_only=True)
    assert float(capped.max()) == pytest.approx(0.3)
    vol = np.array([3.0, 0.0, 0.0])
    down = cap_name_risk(w, vol, vol_k=2.0, long_only=True)
    assert down[0] < w[0]


def test_size_residual_next_open_is_label_not_feature():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        residual_target=True,
        sector_residual=False,
        size_residual=True,
        beta_halflife=5,
        benchmark_symbol="SPY",
        label_return="overnight",
    )
    a = compute_features(_daily_grid(50, seed=1), cfg)
    spy = compute_features(_daily_grid(50, seed=2), cfg)
    iwm = compute_features(_daily_grid(50, seed=3), cfg)
    a["symbol"], spy["symbol"], iwm["symbol"] = "AAPL", "SPY", "IWM"
    base = attach_cross_section_features(
        {"AAPL": a.copy(), "SPY": spy.copy(), "IWM": iwm.copy()}, cfg
    )
    base = attach_residual_target(base, cfg)
    spiked = _daily_grid(50, seed=3)
    spiked.loc[spiked.index[20], "open"] = float(spiked["open"].iloc[20]) * 1.2
    iwm2 = compute_features(spiked, cfg)
    iwm2["symbol"] = "IWM"
    alt = attach_cross_section_features(
        {"AAPL": a.copy(), "SPY": spy.copy(), "IWM": iwm2}, cfg
    )
    alt = attach_residual_target(alt, cfg)
    t = 19
    for name in FEATURE_NAMES:
        assert base["AAPL"][name].iloc[t] == pytest.approx(
            float(alt["AAPL"][name].iloc[t]), abs=1e-10
        )
    assert base["AAPL"]["target"].iloc[t] != pytest.approx(float(alt["AAPL"]["target"].iloc[t]))


def test_peer_residual_other_name_next_open_is_label():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        residual_target=True,
        sector_residual=False,
        peer_residual=True,
        beta_halflife=5,
        benchmark_symbol="SPY",
        label_return="overnight",
        equities_only=False,
    )
    a = compute_features(_daily_grid(50, seed=1), cfg)
    b = compute_features(_daily_grid(50, seed=2), cfg)
    spy = compute_features(_daily_grid(50, seed=3), cfg)
    a["symbol"], b["symbol"], spy["symbol"] = "AAA", "BBB", "SPY"
    base = attach_cross_section_features(
        {"AAA": a.copy(), "BBB": b.copy(), "SPY": spy.copy()}, cfg
    )
    base = attach_residual_target(base, cfg)
    spiked = _daily_grid(50, seed=2)
    spiked.loc[spiked.index[20], "open"] = float(spiked["open"].iloc[20]) * 1.25
    b2 = compute_features(spiked, cfg)
    b2["symbol"] = "BBB"
    alt = attach_cross_section_features(
        {"AAA": a.copy(), "BBB": b2, "SPY": spy.copy()}, cfg
    )
    alt = attach_residual_target(alt, cfg)
    t = 19
    for name in FEATURE_NAMES:
        assert base["AAA"][name].iloc[t] == pytest.approx(
            float(alt["AAA"][name].iloc[t]), abs=1e-10
        )
    assert base["AAA"]["target"].iloc[t] != pytest.approx(float(alt["AAA"]["target"].iloc[t]))


def test_thin_shorts_pay_more_borrow_under_micro():
    # row_cs_thin_mask needs ≥3 finite names on the date.
    w = np.array([[-0.5, 0.25, 0.25, 0.0]])
    tz = np.array([[-2.0, 1.0, 1.0, 0.5]])
    flat = overnight_cost_breakdown(
        weights=w, round_trip_bps=0.0, borrow_bps=5.0, turnover_z=tz
    )
    micro = overnight_cost_breakdown(
        weights=w,
        round_trip_bps=0.0,
        borrow_bps=5.0,
        borrow_thin_k=1.0,
        thin_pctile=0.5,
        turnover_z=tz,
    )
    assert micro["borrow"][0] > flat["borrow"][0]


def test_adv_impact_charges_sqrt_participation():
    w = np.array([[1.0, 0.0]])
    tz = np.array([[-1.0, 1.0]])
    parts = overnight_cost_breakdown(
        weights=w, round_trip_bps=0.0, impact_adv_k=6.0, turnover_z=tz
    )
    assert parts["adv_impact"][0] > 0
    lo = overnight_cost_breakdown(
        weights=np.array([[0.5, 0.5]]),
        round_trip_bps=0.0,
        impact_adv_k=6.0,
        turnover_z=np.array([[-1.0, -1.0]]),
    )
    assert lo["adv_impact"][0] > 0


def test_ic_shrink_does_not_change_mean_cs_ic():
    dates = pd.bdate_range("2022-01-03", periods=40)
    names = [f"S{i}" for i in range(12)]
    rng = np.random.default_rng(2)
    pred = pd.DataFrame(rng.normal(size=(40, 12)), index=dates, columns=names)
    realized = pred * 0.2 + pd.DataFrame(
        rng.normal(scale=0.5, size=(40, 12)), index=dates, columns=names
    )
    base = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.15,
        causal_vol=True,
        round_trip_bps=10.0,
        min_names=5,
    )
    shrunk = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.15,
        causal_vol=True,
        round_trip_bps=10.0,
        min_names=5,
        ic_shrink_lookback=10,
        gap_risk_cap=0.25,
    )
    assert shrunk["mean_cs_ic"] == pytest.approx(base["mean_cs_ic"], abs=1e-8)
    assert shrunk["ic_shrink_lookback"] == pytest.approx(10.0)


def test_training_and_backtest_cli_flags():
    tp = train_parser()
    args = tp.parse_args(
        [
            "--label-return",
            "overnight",
            "--size-residual",
            "--peer-residual",
            "--ridge-sign-shrink",
            "--ridge-long-only",
            "--adv-floor-pctile",
            "0.67",
            "--train-era-adv-pctile",
            "0.3",
        ]
    )
    data_cfg, _model, train_cfg = configs_from_cli(args)
    assert data_cfg.size_residual is True
    assert data_cfg.peer_residual is True
    assert data_cfg.adv_floor_pctile == pytest.approx(0.67)
    assert data_cfg.train_era_adv_pctile == pytest.approx(0.3)
    assert data_cfg.label_return == "overnight"
    assert train_cfg.ridge_sign_shrink is True
    assert train_cfg.ridge_long_only is True
    bp = build_arg_parser()
    bargs = bp.parse_args(
        [
            "--cost-bundle",
            "live_micro",
            "--long-only",
            "--ic-shrink-lookback",
            "63",
            "--gap-risk-cap",
            "0.2",
        ]
    )
    costs = cost_kwargs_from_args(bargs)
    assert costs["name"] == "live_micro_long_only"
    assert costs["borrow_bps"] == pytest.approx(0.0)
    assert costs["impact_adv_k"] == pytest.approx(6.0)
    assert bargs.ic_shrink_lookback == 63
    help_train = tp.format_help()
    assert "--ridge-sign-shrink" in help_train
    assert "--size-residual" in help_train
    help_bt = bp.format_help()
    assert "--ic-shrink-lookback" in help_bt
    assert "live_micro" in help_bt


def test_live_micro_bundle_keeps_long_only_first_class():
    micro = resolve_cost_bundle("live_micro")
    lo = resolve_cost_bundle("live_micro_long_only")
    assert micro["borrow_thin_k"] == pytest.approx(1.0)
    assert lo["borrow_bps"] == pytest.approx(0.0)
    assert lo["borrow_thin_k"] == pytest.approx(0.0)


def test_synthetic_overnight_skip_with_new_objectives(tmp_path: Path):
    from forecast.synthetic import write_cs_overnight_universe
    from forecast.training import train
    from forecast.config import ForecastModelConfig, ForecastTrainConfig
    import torch

    data_dir = tmp_path / "data"
    write_cs_overnight_universe(data_dir, n_names=12, n_days=180, seed=2, rho=0.7)
    data_cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=24,
        stride=1,
        min_context=6,
        warmup_bars=12,
        vol_halflife=8,
        z_window=16,
        z_min_periods=6,
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=8,
        allow_mixed_prices=True,
        cs_zscore=True,
        label_return="overnight",
    )
    summary = train(
        data_cfg,
        ForecastModelConfig(d_model=16, n_layer=1, d_state=8, linear_skip=True),
        ForecastTrainConfig(
            skip_only=True,
            ridge_skip=10.0,
            freeze_skip=True,
            ridge_cs_demean=True,
            ridge_rank_target=True,
            ridge_sign_shrink=True,
            ridge_long_only=True,
            ridge_long_only_quantile=0.3,
            precision="fp32",
            checkpoint_dir=str(tmp_path / "ckpt"),
            eval_train_split=False,
        ),
        device=torch.device("cpu"),
        log_fn=None,
    )
    assert summary["skip_only"] is True
    assert np.isfinite(summary["test"]["cs_ic"])
    assert summary["test"]["cs_ic"] > 0.02
