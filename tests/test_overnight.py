"""Causal overnight gap residual: labels, adjustments, holding-period costs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.alphavantage import bars_to_canonical
from forecast.backtest import book_pnl
from forecast.config import DataConfig
from forecast.data import FEATURE_NAMES, compute_features, embargo_calendar_horizon, normalize_bars
from forecast.diagnostics import looks_like_unadjusted_open, ohlc_body_diagnostics
from forecast.overnight import (
    OVERNIGHT_FORMULA,
    forward_log_return,
    holding_for_label,
    log_positive_price,
    next_open_valid,
    normalize_label_return,
    overnight_one_way_turnover,
    overnight_stress_costs,
    uses_next_open,
)
from forecast.yahoo import parse_yahoo_chart


def _daily_grid(n: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = np.linspace(100.0, 110.0, n)
    open_px = close * 0.995
    open_px[20] = close[19] * 1.04
    return pd.DataFrame(
        {
            "datetime": dates,
            "session": pd.to_datetime(dates).normalize(),
            "mos": np.zeros(n, dtype=int),
            "open": open_px,
            "high": np.maximum(open_px, close) + 0.1,
            "low": np.minimum(open_px, close) - 0.1,
            "close": close,
            "volume": np.full(n, 1000.0),
            "traded": np.ones(n),
        }
    )


def _cfg(**kwargs) -> DataConfig:
    defaults = dict(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        residual_target=False,
        label_return="overnight",
    )
    defaults.update(kwargs)
    return DataConfig(**defaults)


def test_normalize_label_return_aliases():
    assert normalize_label_return("gap") == "overnight"
    assert normalize_label_return("close_open") == "overnight"
    assert normalize_label_return("oc") == "session"
    assert normalize_label_return(None) == "close"
    assert uses_next_open("overnight")
    assert not uses_next_open("close")
    assert holding_for_label("overnight") == "overnight"
    assert holding_for_label("close") == "close"
    assert "log(open[t+h])" in OVERNIGHT_FORMULA


def test_overnight_label_is_next_open_minus_this_close():
    grid = _daily_grid()
    raw = forward_log_return(
        close=grid["close"], open_px=grid["open"], kind="overnight", horizon=1
    )
    t = 19
    expect = float(np.log(grid["open"].iloc[t + 1]) - np.log(grid["close"].iloc[t]))
    assert raw.iloc[t] == pytest.approx(expect)


def test_same_bar_open_is_not_in_overnight_label():
    grid = _daily_grid()
    on = compute_features(grid.copy(), _cfg())
    spiked = grid.copy()
    spiked.loc[spiked.index[19], "open"] = float(spiked["open"].iloc[19]) * 1.5
    on2 = compute_features(spiked, _cfg())
    t = 19
    assert on["target_raw"].iloc[t] == pytest.approx(float(on2["target_raw"].iloc[t]))
    assert on["body_co"].iloc[t] != pytest.approx(float(on2["body_co"].iloc[t]))


def test_next_open_does_not_enter_features_at_t():
    grid = _daily_grid()
    on = compute_features(grid.copy(), _cfg())
    spiked = grid.copy()
    spiked.loc[spiked.index[20], "open"] = float(spiked["open"].iloc[20]) * 1.25
    on2 = compute_features(spiked, _cfg())
    t = 19
    for name in FEATURE_NAMES:
        assert on[name].iloc[t] == pytest.approx(float(on2[name].iloc[t]), abs=1e-12)
    assert on["target_raw"].iloc[t] != pytest.approx(float(on2["target_raw"].iloc[t]))


def test_next_close_does_not_enter_overnight_label():
    grid = _daily_grid()
    on = compute_features(grid.copy(), _cfg())
    spiked = grid.copy()
    spiked.loc[spiked.index[20], "close"] = float(spiked["close"].iloc[20]) * 1.3
    on2 = compute_features(spiked, _cfg())
    t = 19
    assert on["target_raw"].iloc[t] == pytest.approx(float(on2["target_raw"].iloc[t]))


def test_invalid_next_open_is_nan_not_clipped_epsilon():
    open_px = pd.Series([10.0, 0.0, -1.0, np.nan, 11.0])
    logged = log_positive_price(open_px)
    assert np.isnan(logged.iloc[1])
    assert np.isnan(logged.iloc[2])
    assert np.isnan(logged.iloc[3])
    assert logged.iloc[0] == pytest.approx(np.log(10.0))
    grid = _daily_grid()
    grid.loc[grid.index[20], "open"] = 0.0
    on = compute_features(grid, _cfg())
    assert not np.isfinite(on["target_raw"].iloc[19]) or not bool(on["valid"].iloc[19])
    assert not bool(on["valid"].iloc[19])
    assert not bool(next_open_valid(grid["open"], 1).iloc[19])


def test_normalize_bars_nans_nonpositive_open(tmp_path: Path):
    df = pd.DataFrame(
        {
            "datetime": pd.bdate_range("2020-01-02", periods=3),
            "open": [10.0, 0.0, 11.0],
            "high": [10.2, 10.1, 11.2],
            "low": [9.8, 9.9, 10.8],
            "close": [10.1, 10.0, 11.1],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    out = normalize_bars(df, origin="test")
    assert np.isnan(out["open"].iloc[1])
    assert out["close"].iloc[1] == pytest.approx(10.0)


def test_split_adjusted_ohlc_keeps_overnight_gap():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577923200, 1578009600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [200.0, 101.0],
                                "high": [202.0, 103.0],
                                "low": [198.0, 99.0],
                                "close": [200.0, 102.0],
                                "volume": [1000, 1100],
                            }
                        ],
                        "adjclose": [{"adjclose": [100.0, 102.0]}],
                    },
                }
            ],
            "error": None,
        }
    }
    bars = parse_yahoo_chart(payload, interval="daily")
    # Pre-split row is halved consistently; overnight is log(101/100) not log(101/200).
    gap = float(np.log(bars["open"].iloc[1]) - np.log(bars["close"].iloc[0]))
    assert float(bars["close"].iloc[0]) == pytest.approx(100.0)
    assert float(bars["open"].iloc[0]) == pytest.approx(100.0)
    assert gap == pytest.approx(np.log(101.0) - np.log(100.0))
    assert abs(gap) < 0.05


def test_unadjusted_open_vs_adj_close_is_flagged():
    close = np.full(80, 50.0)
    open_px = np.full(80, 100.0)
    stats = ohlc_body_diagnostics(open_px, close)
    assert looks_like_unadjusted_open(stats)
    aligned = ohlc_body_diagnostics(close * 0.995, close)
    assert not looks_like_unadjusted_open(aligned)


def test_bars_to_canonical_rescales_open_with_adjclose():
    df = pd.DataFrame(
        {
            "datetime": ["2020-01-02"],
            "open": [80.0],
            "high": [90.0],
            "low": [70.0],
            "close": [80.0],
            "adjustedclose": [40.0],
            "volume": [1.0],
        }
    )
    bars = bars_to_canonical(df, interval="daily", source="yahoo")
    assert float(bars["open"].iloc[0]) == pytest.approx(40.0)
    assert float(bars["close"].iloc[0]) == pytest.approx(40.0)


def test_overnight_embargo_drops_last_train_label():
    panel = compute_features(_daily_grid(40), _cfg())
    train = embargo_calendar_horizon(panel.iloc[:20].copy(), _cfg())
    assert not bool(train["valid"].iloc[-1])
    assert bool(panel["valid"].any())


def test_overnight_flatten_turnover_is_full_round_trip():
    w = np.array([-0.5, 0.0, 0.5])
    assert overnight_one_way_turnover(w)[0] == pytest.approx(1.0)
    cost = overnight_stress_costs(round_trip_bps=10.0, weights=w)
    assert cost[0] == pytest.approx(10e-4)
    auction = overnight_stress_costs(
        round_trip_bps=10.0, open_auction_bps=10.0, weights=w
    )
    assert auction[0] == pytest.approx(10e-4 + 10e-4 * 0.5)
    borrow = overnight_stress_costs(round_trip_bps=0.0, borrow_bps=10.0, weights=w)
    assert borrow[0] == pytest.approx(10e-4 * 0.5)


def test_overnight_book_charges_more_than_close_roll():
    dates = pd.bdate_range("2022-01-03", periods=40)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (40, 1)), index=dates, columns=names
    )
    realized = pred * 0.02
    noise = np.random.default_rng(0).normal(scale=0.002, size=pred.shape)
    realized = realized + noise
    close_book = book_pnl(
        pred,
        realized,
        hold_halflife=0.0,
        causal_vol=False,
        vol_target=0.0,
        round_trip_bps=10.0,
        holding="close",
        min_names=8,
    )
    on_book = book_pnl(
        pred,
        realized,
        hold_halflife=1.0,  # must be forced to 0
        causal_vol=False,
        vol_target=0.0,
        round_trip_bps=10.0,
        holding="overnight",
        min_names=8,
    )
    assert on_book["holding"] == "overnight"
    assert on_book["hold_halflife"] == pytest.approx(0.0)
    assert on_book["mean_turnover"] > close_book["mean_turnover"]
    assert on_book["unlevered_net_ir"] < close_book["unlevered_net_ir"]


def test_long_only_overnight_has_no_short_borrow():
    dates = pd.bdate_range("2022-01-03", periods=30)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (30, 1)), index=dates, columns=names
    )
    realized = pred * 0.01
    ls = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        round_trip_bps=10.0,
        borrow_bps=50.0,
        min_names=8,
    )
    lo = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        round_trip_bps=10.0,
        borrow_bps=50.0,
        long_only=True,
        min_names=8,
    )
    assert lo["long_only"] is True
    assert ls["mean_cost"] > lo["mean_cost"]


def test_lastbar_residual_zero_steps_matches_skip():
    import importlib.util

    from forecast.data import FEATURE_NAMES
    from forecast.ridge import cs_stats, feature_mask, fit_ridge_xy

    path = Path(__file__).resolve().parents[1] / "scripts" / "cs_overnight.py"
    spec = importlib.util.spec_from_file_location("cs_overnight_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(3)
    n_dates, n_names, f = 24, 12, len(FEATURE_NAMES)
    assert f == feature_mask(mod.PROMOTED["mask_mode"]).size

    def pack(offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = rng.normal(size=(n_dates * n_names, f)).astype(np.float32)
        d = np.repeat(np.arange(n_dates, dtype=np.int64) + offset, n_names)
        y = (x[:, 0] * 0.4 + rng.normal(scale=0.6, size=x.shape[0])).astype(np.float32)
        return x, y, d

    tr_x, tr_y, tr_d = pack(0)
    va_x, va_y, va_d = pack(100)
    te_x, te_y, te_d = pack(200)
    cache = {
        "train_x": tr_x,
        "train_y": tr_y,
        "train_d": tr_d,
        "val_x": va_x,
        "val_y": va_y,
        "val_d": va_d,
        "test_x": te_x,
        "test_y": te_y,
        "test_d": te_d,
        "cs_min_names": 8,
    }
    mask = feature_mask(mod.PROMOTED["mask_mode"])
    w, b, _ic = fit_ridge_xy(
        tr_x.astype(np.float64),
        tr_y.astype(np.float64),
        tr_d,
        ridge=mod.PROMOTED["ridge"],
        min_names=8,
        cs_demean=True,
        rank_target=True,
        feat_winsor=mod.PROMOTED["feat_winsor"],
        feature_mask_bool=mask,
    )
    skip_val = float(
        cs_stats(va_x.astype(np.float64) @ w + b, va_y.astype(np.float64), va_d, min_names=8)[
            "cs_ic"
        ]
    )
    residual = mod._fit_lastbar_residual(cache, w, b, steps=0, hidden=8)
    assert residual["kind"] == "lastbar_mlp_residual"
    assert residual["best_val_ic"] == pytest.approx(skip_val, abs=1e-5)


def test_overnight_cli_flag():
    from forecast.training import build_arg_parser, configs_from_cli

    args = build_arg_parser().parse_args(["--label-return", "overnight", "--interval", "weekly"])
    data_cfg, _, _ = configs_from_cli(args)
    assert data_cfg.label_return == "overnight"
    assert data_cfg.seq_len == 52


def test_skip_only_overnight_recovers_planted_gap(tmp_path: Path):
    import torch

    from forecast.config import ForecastModelConfig, ForecastTrainConfig
    from forecast.synthetic import write_cs_overnight_universe
    from forecast.training import selection_score, train

    data_dir = tmp_path / "data"
    write_cs_overnight_universe(data_dir, n_names=12, n_days=220, seed=2, rho=0.7)
    ckpt_dir = tmp_path / "ckpt"
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
        equities_only=False,
        train_from="",
    )
    summary = train(
        data_cfg,
        ForecastModelConfig(d_model=16, n_layer=1, d_state=8, linear_skip=True),
        ForecastTrainConfig(
            skip_only=True,
            ridge_skip=1.0,
            freeze_skip=True,
            ridge_cs_demean=True,
            precision="fp32",
            checkpoint_dir=str(ckpt_dir),
        ),
        device=torch.device("cpu"),
        log_fn=None,
    )
    test = summary["test"]
    assert summary["cross_section"] is True
    assert np.isfinite(test["cs_ic"])
    assert test["cs_ic"] > 0.05
    assert summary["best_val_ic"] == pytest.approx(selection_score(summary["skip_only_val"]))
