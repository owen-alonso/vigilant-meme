"""Accuracy levers 1–6 and 8: locked-val protocol, residuals, sizing, costs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import book_pnl, build_arg_parser, cost_kwargs_from_args
from forecast.config import DataConfig
from forecast.data import attach_cross_section_features, attach_residual_target, compute_features
from forecast.levers import (
    VAL_LIFT,
    adv_cost_scale,
    apply_gap_risk_cap,
    blend_skip_mlp,
    calibration_scale,
    causal_regime_scale,
    dollar_adv_series,
    fit_long_only_xy,
    train_era_liquid_names,
    val_gate,
    year_sign_consistency_mask,
)
from forecast.overnight import overnight_cost_breakdown, resolve_cost_bundle
from forecast.ridge import feature_mask, fit_ridge_xy
from forecast.training import (
    configs_from_cli,
    build_arg_parser as train_parser,
    masked_long_only_rank_loss,
)
from forecast.universe import peer_symbols_for, size_symbol_for


def _days(ymd: str) -> int:
    return int((np.datetime64(ymd) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))


def _daily_grid(n: int = 50, *, start: str = "2020-01-02", drift: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    close = np.linspace(100.0, 110.0, n) * (1.0 + drift)
    open_px = close * 0.995
    return pd.DataFrame(
        {
            "datetime": dates,
            "session": pd.to_datetime(dates).normalize(),
            "mos": np.zeros(n, dtype=int),
            "open": open_px,
            "high": np.maximum(open_px, close) + 0.1,
            "low": np.minimum(open_px, close) - 0.1,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
            "traded": np.ones(n),
        }
    )


def test_val_gate_requires_lift_and_keeps_2017():
    assert val_gate(0.080, 0.073, candidate_2017=0.036, baseline_2017=0.035)
    assert not val_gate(0.074, 0.073, candidate_2017=0.036)  # lift < 0.003
    assert not val_gate(0.080, 0.073, candidate_2017=0.010)  # kills 2017
    assert VAL_LIFT == pytest.approx(0.003)


def test_year_sign_consistency_drops_flipping_column():
    n_names = 12
    dates = []
    x_rows = []
    y_rows = []
    rng = np.random.default_rng(4)
    for year, x1_sign in ((2000, 1.0), (2001, 1.0), (2002, 1.0), (2003, -1.0), (2004, -1.0)):
        d0 = _days(f"{year}-03-01")
        for k in range(6):
            dates.extend([d0 + k] * n_names)
            good = rng.normal(size=n_names)
            x_rows.append(np.stack([good, x1_sign * good], axis=1))
            y_rows.append(good + 0.05 * rng.normal(size=n_names))
    x = np.concatenate(x_rows)
    y = np.concatenate(y_rows)
    d = np.asarray(dates, dtype=np.int64)
    keep = year_sign_consistency_mask(
        x, y, d, min_names=8, min_frac=0.7, min_abs=0.01, trailing_years=3
    )
    assert bool(keep[0])
    assert not bool(keep[1])


def test_long_only_skip_emphasizes_top_quantile():
    rng = np.random.default_rng(1)
    n_dates, n_names, f = 40, 10, 3
    dates = np.repeat(np.arange(n_dates, dtype=np.int64) + 18000, n_names)
    x = rng.normal(size=(n_dates * n_names, f))
    y = x[:, 0].copy()
    w_ls, _, _ = fit_ridge_xy(x, y, dates, ridge=1.0, min_names=8, rank_target=True)
    w_lo, _, _ = fit_long_only_xy(x, y, dates, ridge=1.0, min_names=8, quantile=0.2)
    assert abs(float(w_lo[0])) > 0
    assert abs(float(w_lo[0])) >= 0.25 * abs(float(w_ls[0]))


def test_train_era_adv_lock_does_not_peek():
    medians = {"A": 1e9, "B": 1e8, "C": 1e6, "D": 5e5}
    names = ["A", "B", "C", "D"]
    keep = train_era_liquid_names(medians, names, floor_usd=1e8, min_names=0)
    assert keep == ["A", "B"]
    keep2 = train_era_liquid_names(medians, names, floor_usd=1e12, min_names=2)
    assert keep2 == ["A", "B"]


def test_dollar_adv_rejects_nonpositive():
    adv = dollar_adv_series(np.array([10.0, 0.0, 5.0]), np.array([2.0, 3.0, 0.0]))
    assert adv[0] == pytest.approx(20.0)
    assert np.isnan(adv[1])
    assert np.isnan(adv[2])


def test_adv_cost_scale_charges_thin_names_more():
    adv = np.array([1e9, 1e8, 1e7])
    scale = adv_cost_scale(adv, k=1.0, max_mult=8.0)
    assert scale[2] > scale[1]
    assert scale[0] == pytest.approx(1.0)
    assert scale[1] == pytest.approx(1.0)


def test_gap_risk_cap_scales_notional():
    w = np.array([0.5, 0.5])
    vol = np.array([0.04, 0.04])
    capped = apply_gap_risk_cap(w, vol, cap=0.02)
    assert float((np.abs(capped) * vol).sum()) == pytest.approx(0.02)


def test_causal_regime_scale_flattens_dead_high_vol():
    dates = np.array([10, 10, 11, 11], dtype=np.int64)
    keys = np.array([10, 11], dtype=np.int64)
    trailing_ic = np.array([0.05, -0.02])
    trailing_t = np.array([2.0, -1.0])
    spy = {10: 0.01, 11: 0.05}
    scale = causal_regime_scale(
        dates,
        date_keys=keys,
        trailing_ic=trailing_ic,
        trailing_t=trailing_t,
        train_ic=0.07,
        spy_vol=spy,
        train_vol_cut=0.03,
        mode="flatten_ic",
    )
    assert scale[0] == pytest.approx(1.0)
    assert scale[2] == pytest.approx(0.0)


def test_size_residual_uses_iwm_forward_not_features():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        residual_target=True,
        sector_residual=True,
        size_residual=True,
        beta_halflife=5,
        benchmark_symbol="SPY",
        label_return="overnight",
    )
    a = compute_features(_daily_grid(50), cfg)
    spy = compute_features(_daily_grid(50, drift=0.01), cfg)
    xlk = compute_features(_daily_grid(50, drift=0.005), cfg)
    iwm = compute_features(_daily_grid(50, drift=-0.01), cfg)
    a["symbol"], spy["symbol"], xlk["symbol"], iwm["symbol"] = "AAPL", "SPY", "XLK", "IWM"
    base_panels = attach_cross_section_features(
        {"AAPL": a.copy(), "SPY": spy.copy(), "XLK": xlk.copy(), "IWM": iwm.copy()}, cfg
    )
    base = attach_residual_target(base_panels, cfg)
    spiked_grid = _daily_grid(50, drift=-0.01)
    spiked_grid.loc[spiked_grid.index[-1], "open"] = float(spiked_grid["open"].iloc[-1]) * 1.08
    spiked = compute_features(spiked_grid, cfg)
    spiked["symbol"] = "IWM"
    alt_panels = attach_cross_section_features(
        {"AAPL": a.copy(), "SPY": spy.copy(), "XLK": xlk.copy(), "IWM": spiked}, cfg
    )
    alt = attach_residual_target(alt_panels, cfg)
    t = 48
    assert base["AAPL"]["ret_1"].iloc[t] == pytest.approx(float(alt["AAPL"]["ret_1"].iloc[t]))
    assert base["AAPL"]["target"].iloc[t] != pytest.approx(float(alt["AAPL"]["target"].iloc[t]))
    assert size_symbol_for("AAPL") == "IWM"


def test_peer_residual_excludes_self_and_is_causal():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        residual_target=True,
        sector_residual=True,
        peer_residual=True,
        beta_halflife=5,
        benchmark_symbol="SPY",
        label_return="overnight",
    )
    names = ["AAPL", "MSFT", "NVDA", "SPY", "XLK"]
    panels = {}
    for i, nm in enumerate(names):
        g = compute_features(_daily_grid(50, drift=0.002 * i), cfg)
        g["symbol"] = nm
        panels[nm] = g
    base_panels = attach_cross_section_features({k: v.copy() for k, v in panels.items()}, cfg)
    base = attach_residual_target(base_panels, cfg)
    spiked_grid = _daily_grid(50, drift=0.002)
    spiked_grid.loc[spiked_grid.index[-1], "open"] = float(spiked_grid["open"].iloc[-1]) * 1.10
    spiked = compute_features(spiked_grid, cfg)
    spiked["symbol"] = "MSFT"
    alt_in = {k: v.copy() for k, v in panels.items()}
    alt_in["MSFT"] = spiked
    alt_panels = attach_cross_section_features(alt_in, cfg)
    alt = attach_residual_target(alt_panels, cfg)
    t = 48
    assert base["AAPL"]["ret_1"].iloc[t] == pytest.approx(float(alt["AAPL"]["ret_1"].iloc[t]))
    assert base["AAPL"]["target"].iloc[t] != pytest.approx(float(alt["AAPL"]["target"].iloc[t]))
    peers = peer_symbols_for("AAPL", ["AAPL", "MSFT", "NVDA", "JPM"])
    assert "AAPL" not in peers
    assert "MSFT" in peers
    assert "JPM" not in peers


def test_live_adv_bundle_and_thin_borrow():
    bundle = resolve_cost_bundle("live_adv")
    assert bundle["adv_borrow_k"] > 0
    assert bundle["adv_auction_k"] > 0
    lo = resolve_cost_bundle("live_adv_long_only")
    assert lo["borrow_bps"] == pytest.approx(0.0)
    w = np.array([[0.5, -0.5]])
    adv = np.array([[1e9, 1e7]])
    flat = overnight_cost_breakdown(
        weights=w, round_trip_bps=0.0, borrow_bps=10.0, dollar_adv=adv, adv_borrow_k=0.0
    )
    scaled = overnight_cost_breakdown(
        weights=w, round_trip_bps=0.0, borrow_bps=10.0, dollar_adv=adv, adv_borrow_k=1.0
    )
    assert scaled["borrow"][0] > flat["borrow"][0]


def test_gap_cap_and_ic_shrink_cli():
    args = build_arg_parser().parse_args(
        [
            "--live-costs",
            "--long-only",
            "--ic-shrink-lookback",
            "63",
            "--gap-risk-cap",
            "0.02",
            "--cost-bundle",
            "live_adv",
        ]
    )
    costs = cost_kwargs_from_args(args)
    assert costs["name"] == "live_adv_long_only"
    assert costs["borrow_bps"] == pytest.approx(0.0)
    assert args.ic_shrink_lookback == 63
    assert args.gap_risk_cap == pytest.approx(0.02)


def test_book_pnl_gap_cap_records_cap():
    idx = pd.bdate_range("2020-01-02", periods=30)
    rng = np.random.default_rng(0)
    pred = pd.DataFrame(rng.normal(size=(30, 8)), index=idx, columns=[f"s{i}" for i in range(8)])
    realized = pred.copy() * 0.2
    vol = pd.DataFrame(0.05, index=idx, columns=pred.columns)
    capped = book_pnl(
        pred,
        realized,
        vol_target=0.0,
        holding="overnight",
        min_names=4,
        hold_halflife=0,
        vol_level=vol,
        gap_risk_cap=0.01,
        long_only=True,
    )
    assert capped["gap_risk_cap"] == pytest.approx(0.01)
    assert np.isfinite(capped["unlevered_net_ir"])


def test_training_cli_exposes_lever_flags():
    p = train_parser()
    args = p.parse_args(
        [
            "--label-return",
            "overnight",
            "--skip-only",
            "--ridge-objective",
            "long_only",
            "--ridge-year-stable",
            "train_recency",
            "--size-residual",
            "--peer-residual",
            "--train-adv-floor-pctile",
            "0.67",
            "--ensemble-mlp",
            "--long-only-loss-weight",
            "0.5",
        ]
    )
    data_cfg, _model_cfg, train_cfg = configs_from_cli(args)
    assert data_cfg.label_return == "overnight"
    assert data_cfg.size_residual is True
    assert data_cfg.peer_residual is True
    assert data_cfg.train_adv_floor_pctile == pytest.approx(0.67)
    assert train_cfg.ridge_objective == "long_only"
    assert train_cfg.ridge_year_stable == "train_recency"
    assert train_cfg.ensemble_mlp is True
    assert train_cfg.long_only_loss_weight == pytest.approx(0.5)


def test_long_only_rank_loss_is_finite():
    import torch

    mean = torch.linspace(-1, 1, 10)
    target = torch.linspace(-2, 2, 10)
    mask = torch.ones(10, dtype=torch.bool)
    dates = torch.zeros(10, dtype=torch.int64)
    loss = masked_long_only_rank_loss(mean, target, mask, date_ids=dates, quantile=0.2)
    assert torch.isfinite(loss)


def test_ensemble_mix_can_be_skip_only():
    rng = np.random.default_rng(2)
    n_dates, n_names, f = 50, 8, 4
    dates = np.repeat(np.arange(n_dates, dtype=np.int64) + 20000, n_names)
    x = rng.normal(size=(n_dates * n_names, f))
    y = x[:, 0] + 0.1 * rng.normal(size=x.shape[0])
    w, b, _ = fit_ridge_xy(x, y, dates, ridge=5.0, min_names=6, rank_target=True)
    skip = x @ w + b
    ens = blend_skip_mlp(x, y, dates, skip, min_names=6, mixes=(0.0, 0.5, 1.0), steps=20)
    assert ens["mix"] in (0.0, 0.5, 1.0)
    assert np.isfinite(ens["hold_ic"])


def test_accuracy_levers_script_synthetic(tmp_path: Path):
    import importlib.util
    import json

    path = Path(__file__).resolve().parents[1] / "scripts" / "cs_accuracy_levers.py"
    spec = importlib.util.spec_from_file_location("cs_accuracy_levers", path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    out = tmp_path / "levers.json"
    rc = script.main(["--synthetic", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert "overnight_skip" in payload["promoted"]
    assert payload["table"].startswith("| lever")
    assert "Vendor data-quality ingest" in payload["vendor_ingest"]


def test_feature_mask_unchanged_default():
    m = feature_mask("no_long_ts")
    assert int(m.sum()) < len(m)


def test_dollar_adv_column_on_features():
    cfg = DataConfig(interval="daily", horizon=1, warmup_bars=5, vol_halflife=5, z_window=10, z_min_periods=3)
    panel = compute_features(_daily_grid(20), cfg)
    assert "dollar_adv" in panel.columns
    assert float(panel["dollar_adv"].iloc[-1]) == pytest.approx(
        float(panel["volume"].iloc[-1] * panel["close"].iloc[-1])
    )
