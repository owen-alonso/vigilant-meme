"""Forecast data causality, horizon prints, session ffill, timezone, close>0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.config import DataConfig, ForecastModelConfig, ForecastTrainConfig, validate_loss_head
from forecast.data import (
    FEATURE_NAMES,
    SequenceDataset,
    SymbolArrays,
    _session_naive_datetime,
    attach_cross_section_features,
    attach_residual_target,
    build_datasets,
    build_session_grid,
    compute_features,
    embargo_calendar_horizon,
    fit_ridge_readout,
    load_bars,
)
from forecast.model import ReturnForecaster
from forecast.training import (
    compute_metrics,
    configs_from_cli,
    decide_val_plateau,
    masked_correlation_loss,
    masked_loss,
    build_arg_parser,
)


def _tiny_cfg(**kwargs) -> DataConfig:
    defaults = dict(
        interval="1min",
        horizon=2,
        warmup_bars=3,
        vol_halflife=2,
        z_window=5,
        z_min_periods=2,
        min_session_bars=2,
        require_horizon_traded=True,
        max_session_gap_days=4,
    )
    defaults.update(kwargs)
    return DataConfig(**defaults)


def _grid(n: int = 20, traded: np.ndarray | None = None) -> pd.DataFrame:
    close = 10.0 + np.linspace(0, 0.2, n)
    if traded is None:
        traded = np.ones(n)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-02 09:30", periods=n, freq="min"),
            "session": pd.Timestamp("2024-01-02"),
            "mos": np.arange(n),
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": np.full(n, 100.0),
            "traded": traded.astype(np.float64),
        }
    )


def test_valid_requires_horizon_print():
    traded = np.ones(20)
    traded[10] = 0.0
    panel = compute_features(_grid(20, traded), _tiny_cfg(horizon=2))
    # mos=8 looks 2 bars ahead at mos=10, which is untraded.
    row = panel.loc[panel["mos"] == 8].iloc[0]
    assert row["horizon_traded"] == 0
    assert not bool(row["valid"])
    # A pair of traded bars with a traded horizon can be valid after warmup.
    later = panel.loc[panel["mos"] == 12].iloc[0]
    assert later["horizon_traded"] == 1


def test_stale_horizon_allowed_when_flag_off():
    traded = np.ones(20)
    traded[10] = 0.0
    panel = compute_features(_grid(20, traded), _tiny_cfg(horizon=2, require_horizon_traded=False))
    row = panel.loc[panel["mos"] == 8].iloc[0]
    assert row["horizon_traded"] == 0
    # May still fail warmup/features; the extra constraint is not applied.
    assert row["traded"] == 1


def test_ffill_does_not_cross_sessions(tmp_path: Path):
    rows = []
    for i, minute in enumerate([0, 1]):
        rows.append(
            {
                "datetime": pd.Timestamp("2024-01-02 09:30") + pd.Timedelta(minutes=minute),
                "Open": 10.0,
                "High": 10.0,
                "Low": 10.0,
                "Close": 10.0 + i,
                "Volume": 1.0,
            }
        )
    rows.append(
        {
            "datetime": pd.Timestamp("2024-01-09 10:00"),
            "Open": 50.0,
            "High": 50.0,
            "Low": 50.0,
            "Close": 50.0,
            "Volume": 1.0,
        }
    )
    path = tmp_path / "TEST_clean_1min.parquet"
    pd.DataFrame(rows).to_parquet(path)
    df = load_bars(path)
    grid = build_session_grid(df, _tiny_cfg(min_session_bars=1))
    day2 = grid[grid["session"] == pd.Timestamp("2024-01-09")]
    morning = day2[day2["mos"] < 30]
    assert morning.empty
    first = day2.iloc[0]
    assert int(first["mos"]) == 30
    assert float(first["close"]) == 50.0


def test_large_gap_does_not_become_one_minute_return():
    n = 12
    g1 = _grid(n)
    g2 = _grid(n)
    g2["session"] = pd.Timestamp("2024-01-20")
    g2["datetime"] = pd.date_range("2024-01-20 09:30", periods=n, freq="min")
    g2["close"] = g2["close"] + 5.0
    grid = pd.concat([g1, g2], ignore_index=True)
    panel = compute_features(grid, _tiny_cfg(max_session_gap_days=4, warmup_bars=0))
    join = panel[panel["session"] == pd.Timestamp("2024-01-20")].iloc[0]
    # ret_1 is filled to 0 after validity; the join itself must not be labelled.
    assert not bool(join["valid"])


def test_utc_timestamps_convert_to_eastern():
    # 14:30 UTC in January is 09:30 EST.
    utc = pd.Series(pd.to_datetime(["2024-01-02 14:30:00"]).tz_localize("UTC"))
    naive = _session_naive_datetime(utc)
    assert naive.dt.tz is None
    assert int(naive.dt.hour.iloc[0]) == 9
    assert int(naive.dt.minute.iloc[0]) == 30


def test_nonpositive_close_raises(tmp_path: Path):
    path = tmp_path / "BAD_clean_1min.parquet"
    pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-02 09:30")],
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [0.0],
            "Volume": [1.0],
        }
    ).to_parquet(path)
    with pytest.raises(ValueError, match="close <= 0"):
        load_bars(path)


def test_features_ignore_future_price_spike():
    grid = _grid(16)
    cfg = _tiny_cfg(warmup_bars=4, horizon=2)
    base = compute_features(grid, cfg)
    spiked = grid.copy()
    spiked.loc[15, "close"] = 999.0
    alt = compute_features(spiked, cfg)
    # Position 8 must not see the spike at 15 (beyond its causal past).
    cols = ["ret_1", "ret_5", "vol_level"]
    assert np.allclose(
        base.loc[8, cols].to_numpy(dtype=np.float64),
        alt.loc[8, cols].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_validate_loss_head_rejects_mismatches():
    with pytest.raises(ValueError):
        validate_loss_head(
            ForecastModelConfig(heteroscedastic=True),
            ForecastTrainConfig(loss="huber", sigma_aux_weight=0.0),
        )
    with pytest.raises(ValueError):
        validate_loss_head(
            ForecastModelConfig(heteroscedastic=False),
            ForecastTrainConfig(loss="gaussian"),
        )
    validate_loss_head(
        ForecastModelConfig(heteroscedastic=True),
        ForecastTrainConfig(loss="huber", sigma_aux_weight=0.5),
    )


def test_predict_restores_training_flag():
    import torch

    model = ReturnForecaster(
        ForecastModelConfig(n_features=18, d_model=16, n_layer=1, dropout=0.5)
    )
    model.train()
    x = torch.randn(2, 8, 18)
    model.predict(x)
    assert model.training


def test_masked_loss_all_masked_is_nan():
    import torch

    mean = torch.zeros(2, 4)
    log_sigma = torch.zeros_like(mean)
    target = torch.ones_like(mean)
    mask = torch.zeros_like(mean)
    loss = masked_loss(mean, log_sigma, target, mask, ForecastTrainConfig(loss="huber"))
    assert torch.isnan(loss)


def test_val_plateau_cuts_lr_instead_of_stopping():
    kwargs = dict(
        plateau_evals=2,
        plateau_factor=0.5,
        min_scale=0.01,
        early_stop_evals=0,
    )
    n, scale, stop, restore, msg = decide_val_plateau(
        improved=False,
        evals_without_gain=0,
        lr_scale=1.0,
        **kwargs,
    )
    assert (n, scale, stop, restore) == (1, 1.0, False, False)
    assert msg is None

    n, scale, stop, restore, msg = decide_val_plateau(
        improved=False,
        evals_without_gain=1,
        lr_scale=1.0,
        **kwargs,
    )
    assert stop is False
    assert restore is True
    assert n == 2
    assert scale == pytest.approx(0.5)
    assert msg is not None and "continuing" in msg

    n, scale, stop, restore, msg = decide_val_plateau(
        improved=True,
        evals_without_gain=3,
        lr_scale=0.5,
        **kwargs,
    )
    assert (n, scale, stop, restore) == (0, 0.5, False, False)


def test_val_plateau_early_stop_still_optional():
    n, scale, stop, restore, msg = decide_val_plateau(
        improved=False,
        evals_without_gain=1,
        lr_scale=1.0,
        plateau_evals=8,
        plateau_factor=0.5,
        min_scale=0.01,
        early_stop_evals=2,
    )
    assert stop is True
    assert restore is False
    assert scale == 1.0
    assert msg is not None and "early stop" in msg


def _daily_grid(n: int = 60, split_at: int | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = np.linspace(100.0, 110.0, n)
    if split_at is not None:
        close = close.copy()
        close[split_at:] = close[split_at:] * 0.25
    return pd.DataFrame(
        {
            "datetime": dates,
            "session": pd.to_datetime(dates).normalize(),
            "mos": np.zeros(n, dtype=int),
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(n, 1000.0),
            "traded": np.ones(n),
        }
    )


def test_split_sized_forward_return_is_unlabelled():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        max_abs_log_return=0.40,
        max_abs_target=8.0,
    )
    panel = compute_features(_daily_grid(50, split_at=30), cfg)
    pre_split = panel.iloc[29]
    assert abs(float(pre_split["target_raw"])) > 0.40
    assert not bool(pre_split["valid"])


def test_calendar_embargo_drops_last_horizon_labels():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        max_abs_log_return=0.40,
    )
    panel = compute_features(_daily_grid(80), cfg)
    assert bool(panel.iloc[38]["valid"])
    train = embargo_calendar_horizon(panel.iloc[:40], cfg)
    assert not bool(train["valid"].iloc[-1])


def test_linear_skip_is_the_init_readout():
    import torch

    model = ReturnForecaster(
        ForecastModelConfig(n_features=18, d_model=16, n_layer=1, dropout=0.0)
    )
    model.eval()
    x = torch.randn(2, 8, 18)
    mean, _ = model(x)
    skip = model.skip(x).squeeze(-1)
    assert torch.allclose(mean, skip, atol=1e-5)
    assert not torch.allclose(model.skip.weight, torch.zeros_like(model.skip.weight))


def test_cs_center_loss_changes_huber_when_dates_disagree():
    import torch

    mean = torch.tensor([[2.0, 4.0], [10.0, 12.0]])
    target = torch.tensor([[1.0, 3.0], [9.0, 11.0]])
    mask = torch.ones_like(mean)
    dates = torch.tensor([[1, 1], [2, 2]])
    cfg = ForecastTrainConfig(
        loss="huber",
        location_loss_weight=1.0,
        ic_loss_weight=0.0,
        sign_loss_weight=0.0,
        rank_loss_weight=0.0,
        pred_std_weight=0.0,
        cs_center_loss=True,
    )
    centered = float(masked_loss(mean, torch.zeros_like(mean), target, mask, cfg, date_ids=dates))
    cfg.cs_center_loss = False
    raw = float(masked_loss(mean, torch.zeros_like(mean), target, mask, cfg, date_ids=dates))
    assert centered < raw
    import torch

    mean = torch.tensor([[0.2, 0.4, -0.1, 0.3]])
    target = mean.clone()
    mask = torch.ones_like(mean)
    aligned = masked_correlation_loss(mean, target, mask)
    flipped = masked_correlation_loss(-mean, target, mask)
    assert float(aligned) < float(flipped)
    assert float(aligned) == pytest.approx(0.0, abs=1e-5)


def test_correlation_loss_is_pooled_across_sequences():
    import torch

    mean = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, 0.0, 0.0]])
    target = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    pooled = float(masked_correlation_loss(mean, target, mask))
    # Four aligned bars outweigh two flipped ones; per-sequence mean of
    # (0 + 2) / 2 would stay at 1.0.
    assert pooled < 0.5


def test_feature_count_matches_model_default():
    assert len(FEATURE_NAMES) == ForecastModelConfig().n_features


def test_cross_section_peer_is_the_other_name_same_bar():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
    )
    a = compute_features(_daily_grid(40), cfg)
    b_grid = _daily_grid(40)
    b_grid["close"] = b_grid["close"] * 1.5 + np.linspace(0, 2.0, 40)
    b = compute_features(b_grid, cfg)
    a["symbol"] = "AAA"
    b["symbol"] = "BBB"
    out = attach_cross_section_features({"AAA": a, "BBB": b}, cfg)
    mid = 20
    assert out["AAA"]["peer_ret_1"].iloc[mid] == pytest.approx(
        float(out["BBB"]["ret_1"].iloc[mid]), abs=1e-5
    )
    assert out["BBB"]["peer_ret_1"].iloc[mid] == pytest.approx(
        float(out["AAA"]["ret_1"].iloc[mid]), abs=1e-5
    )
    expected_mkt = 0.5 * (
        float(out["AAA"]["ret_1"].iloc[mid]) + float(out["BBB"]["ret_1"].iloc[mid])
    )
    assert out["AAA"]["mkt_ret_1"].iloc[mid] == pytest.approx(expected_mkt, abs=1e-5)


def test_cross_section_does_not_see_future_peer_return():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
    )
    a = compute_features(_daily_grid(40), cfg)
    b = compute_features(_daily_grid(40), cfg)
    a["symbol"] = "AAA"
    b["symbol"] = "BBB"
    base = attach_cross_section_features({"AAA": a.copy(), "BBB": b.copy()}, cfg)
    spiked = b.copy()
    spiked.loc[spiked.index[-1], "ret_1"] = 9.0
    alt = attach_cross_section_features({"AAA": a.copy(), "BBB": spiked}, cfg)
    assert base["AAA"]["peer_ret_1"].iloc[10] == pytest.approx(
        float(alt["AAA"]["peer_ret_1"].iloc[10]), abs=1e-8
    )


def test_ridge_readout_recovers_linear_target():
    n, f = 80, 4
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, f)).astype(np.float32)
    true_w = np.array([0.4, -0.2, 0.1, 0.0], dtype=np.float32)
    y = x @ true_w + 0.05
    sym = SymbolArrays(
        symbol="T",
        features=x,
        target=y.astype(np.float32),
        scale=np.ones(n, dtype=np.float32),
        valid=np.ones(n, dtype=bool),
    )
    mean = np.zeros(f, dtype=np.float32)
    std = np.ones(f, dtype=np.float32)
    w, b, ic = fit_ridge_readout([sym], mean, std, ridge=1e-4)
    assert ic > 0.99
    assert np.allclose(w, true_w, atol=0.08)
    pred = x @ w + b
    assert float(pred.std()) == pytest.approx(abs(ic) * float(y.std()), rel=0.05)


def test_val_plateau_keeps_counter_so_early_stop_can_fire():
    n, scale, stop, restore, msg = decide_val_plateau(
        improved=False,
        evals_without_gain=3,
        lr_scale=0.5,
        plateau_evals=2,
        plateau_factor=0.5,
        min_scale=0.01,
        early_stop_evals=4,
    )
    assert stop is True
    assert restore is False
    assert n == 4
    assert msg is not None and "early stop" in msg


def test_winsorized_ic_downweights_split_outlier():
    pred = np.array([0.3, 0.2, 0.1, 0.05, 0.0], dtype=np.float64)
    target = np.array([0.3, 0.2, 0.1, 0.05, -40.0], dtype=np.float64)
    scale = np.ones(5, dtype=np.float64)
    metrics = compute_metrics(pred, target, scale, winsor=3.0)
    inlier = compute_metrics(pred[:-1], target[:-1], scale[:-1], winsor=3.0)
    assert inlier["ic_raw"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["ic_spearman"] > metrics["ic_raw"]
    assert metrics["ic"] == pytest.approx(metrics["ic_raw"])
    assert abs(metrics["ic_winsor"] - inlier["ic_raw"]) < abs(
        metrics["ic_raw"] - inlier["ic_raw"]
    )


def test_weekly_cli_uses_week_scale_context():
    args = build_arg_parser().parse_args(["--interval", "weekly"])
    data_cfg, model_cfg, train_cfg = configs_from_cli(args)
    assert data_cfg.seq_len == 52
    assert data_cfg.vol_halflife == 12
    assert data_cfg.z_window == 52
    assert data_cfg.supervise_last == 1
    assert model_cfg.d_model == 32
    assert model_cfg.n_layer == 1
    assert model_cfg.d_state == 8
    assert model_cfg.linear_skip is True
    assert model_cfg.dt_min == pytest.approx(0.05)
    assert train_cfg.ic_loss_weight == pytest.approx(2.0)
    assert train_cfg.rank_loss_weight == pytest.approx(1.0)
    assert train_cfg.early_stop_evals == 24
    assert train_cfg.ridge_skip == pytest.approx(1.0)
    assert train_cfg.freeze_skip is True
    assert train_cfg.ridge_cs_demean is True
    assert train_cfg.cs_center_loss is True
    assert train_cfg.skip_only is False
    assert data_cfg.eval_last_bar is True
    assert data_cfg.global_calendar_split is True
    assert data_cfg.residual_target is True
    assert data_cfg.cs_zscore is True
    assert data_cfg.cross_section_min_names == 30
    assert data_cfg.universe == ""


def test_sequence_dataset_last_bar_is_six_tuple():
    n, f = 20, 4
    dates = np.arange(n, dtype=np.int64)
    sym = SymbolArrays(
        symbol="AAA",
        features=np.ones((n, f), dtype=np.float32),
        target=np.linspace(-1, 1, n, dtype=np.float32),
        scale=np.ones(n, dtype=np.float32),
        valid=np.ones(n, dtype=bool),
        dates=dates,
    )
    ds = SequenceDataset([sym], seq_len=8, stride=1, last_bar_only=True, min_context=2)
    x, y, mask, scale, date_id, sym_idx = ds[0]
    assert x.shape == (8, f)
    assert int(mask.sum()) == 1
    assert bool(mask[-1])
    assert int(date_id) == int(dates[7])
    assert int(sym_idx) == 0


def test_spy_is_mkt_feature_not_equal_weight():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        benchmark_symbol="SPY",
    )
    a = compute_features(_daily_grid(40), cfg)
    b_grid = _daily_grid(40)
    b_grid["close"] = b_grid["close"] * 1.2 + np.linspace(0, 1.0, 40)
    b = compute_features(b_grid, cfg)
    spy_grid = _daily_grid(40)
    spy_grid["close"] = spy_grid["close"] * 0.9 + np.linspace(0, 3.0, 40)
    spy = compute_features(spy_grid, cfg)
    a["symbol"], b["symbol"], spy["symbol"] = "AAPL", "MSFT", "SPY"
    out = attach_cross_section_features({"AAPL": a, "MSFT": b, "SPY": spy}, cfg)
    mid = 20
    assert out["AAPL"]["mkt_ret_1"].iloc[mid] == pytest.approx(
        float(out["SPY"]["ret_1"].iloc[mid]), abs=1e-5
    )
    ew = 0.5 * (
        float(out["AAPL"]["ret_1"].iloc[mid]) + float(out["MSFT"]["ret_1"].iloc[mid])
    )
    assert out["AAPL"]["mkt_ret_1"].iloc[mid] != pytest.approx(ew, abs=1e-4)
    assert out["AAPL"]["peer_ret_1"].iloc[mid] == pytest.approx(
        float(out["MSFT"]["ret_1"].iloc[mid]), abs=1e-5
    )


def test_residual_target_uses_future_spy_in_label_not_features():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        residual_target=True,
        beta_halflife=5,
        benchmark_symbol="SPY",
    )
    a = compute_features(_daily_grid(50), cfg)
    spy = compute_features(_daily_grid(50), cfg)
    a["symbol"] = "AAPL"
    spy["symbol"] = "SPY"
    base = attach_residual_target({"AAPL": a.copy(), "SPY": spy.copy()}, cfg)
    spiked_grid = _daily_grid(50)
    spiked_grid.loc[spiked_grid.index[-1], "close"] = float(spiked_grid["close"].iloc[-1]) * 1.08
    spiked = compute_features(spiked_grid, cfg)
    spiked["symbol"] = "SPY"
    alt = attach_residual_target({"AAPL": a.copy(), "SPY": spiked}, cfg)
    t = 48
    assert base["AAPL"]["ret_1"].iloc[t] == pytest.approx(float(alt["AAPL"]["ret_1"].iloc[t]))
    assert base["AAPL"]["mkt_ret_1"].iloc[t] == pytest.approx(
        float(alt["AAPL"]["mkt_ret_1"].iloc[t])
    )
    assert base["AAPL"]["target"].iloc[t] != pytest.approx(float(alt["AAPL"]["target"].iloc[t]))


def test_mean_cs_ic_averages_per_date_pearson():
    from forecast.training import mean_cs_ic

    pred = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    target = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], dtype=np.float64)
    dates = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    # date 1: +1; date 2: -1
    assert mean_cs_ic(pred, target, dates) == pytest.approx(0.0, abs=1e-6)


def test_selection_score_prefers_cs_ic_when_finite():
    from forecast.training import selection_score

    assert selection_score({"ic": 0.02, "cs_ic": 0.05}) == pytest.approx(0.05)
    assert selection_score({"ic": 0.02, "cs_ic": float("nan")}) == pytest.approx(0.02)
    assert selection_score({"ic": 0.02}) == pytest.approx(0.02)


def test_ranknet_is_within_date_when_dates_have_breadth():
    import torch

    from forecast.training import masked_pairwise_rank_loss

    mean = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    target = torch.tensor([[1.0, 2.0], [0.0, 3.0]])
    mask = torch.ones_like(mean)
    dates = torch.tensor([[10, 10], [11, 11]])
    # Within each date the order is flipped on date 11, aligned on date 10.
    loss = float(masked_pairwise_rank_loss(mean, target, mask, date_ids=dates))
    pooled = float(masked_pairwise_rank_loss(mean, target, mask, date_ids=None))
    assert loss > 0
    assert loss != pytest.approx(pooled, abs=1e-6)


def test_freeze_skip_disables_ridge_gradients():
    import torch

    from forecast.training import apply_ridge_skip
    n, f = 40, len(FEATURE_NAMES)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, f)).astype(np.float32)
    y = (x[:, 0] * 0.4).astype(np.float32)
    sym = SymbolArrays(
        symbol="AAA",
        features=x,
        target=y,
        scale=np.ones(n, dtype=np.float32),
        valid=np.ones(n, dtype=bool),
    )
    model = ReturnForecaster(
        ForecastModelConfig(n_features=f, d_model=16, n_layer=1, d_state=8, linear_skip=True)
    )
    bundle = {
        "train_symbols": [sym],
        "feature_mean": np.zeros(f, dtype=np.float32),
        "feature_std": np.ones(f, dtype=np.float32),
    }
    apply_ridge_skip(
        model, bundle, ForecastTrainConfig(ridge_skip=1.0, freeze_skip=True), torch.device("cpu")
    )
    assert model.skip.weight.requires_grad is False
    assert model.skip.bias.requires_grad is False


def _write_daily_parquet(path: Path, symbol: str, n: int, start: str, drift: float = 0.01) -> None:
    dates = pd.bdate_range(start, periods=n)
    close = 100.0 + drift * np.arange(n)
    pd.DataFrame(
        {
            "datetime": dates,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    ).to_parquet(path)


def test_global_calendar_split_shares_one_train_end(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_parquet(data_dir / "AAA_daily.parquet", "AAA", 200, "2018-01-02", 0.02)
    _write_daily_parquet(data_dir / "BBB_daily.parquet", "BBB", 120, "2019-01-02", 0.03)
    cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=16,
        stride=1,
        min_context=4,
        warmup_bars=8,
        vol_halflife=5,
        z_window=10,
        z_min_periods=5,
        global_calendar_split=True,
        residual_target=False,
        eval_last_bar=True,
        cross_section_min_names=99,
        allow_mixed_prices=True,
    )
    bundle = build_datasets(cfg, log_fn=None)
    ends = {m["train_end"] for m in bundle["meta"]}
    assert len(ends) == 1
    assert bundle["cross_section"] is False


def test_cross_section_dataset_when_enough_names(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(8):
        _write_daily_parquet(
            data_dir / f"S{i}_daily.parquet",
            f"S{i}",
            90,
            "2018-01-02",
            0.01 + 0.002 * i,
        )
    cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=16,
        stride=1,
        min_context=4,
        warmup_bars=8,
        vol_halflife=5,
        z_window=10,
        z_min_periods=5,
        global_calendar_split=True,
        residual_target=False,
        eval_last_bar=True,
        cross_section_min_names=8,
        allow_mixed_prices=True,
    )
    bundle = build_datasets(cfg, log_fn=None)
    assert bundle["cross_section"] is True
    x, y, mask, scale, dates, syms = bundle["datasets"]["train"][0]
    assert x.dim() == 3
    assert int(x.size(0)) >= 8
    assert int(mask[0].sum()) == 1
    assert bool(mask[0, -1])
    assert int(dates.unique().numel()) == 1


def test_cs_zscore_is_same_day_and_excludes_spy():
    cfg = DataConfig(
        interval="daily",
        horizon=1,
        warmup_bars=5,
        vol_halflife=5,
        z_window=10,
        z_min_periods=3,
        benchmark_symbol="SPY",
        cs_zscore=True,
    )
    a = compute_features(_daily_grid(40), cfg)
    b_grid = _daily_grid(40)
    b_grid["close"] = b_grid["close"] * 1.2 + np.linspace(0, 1.0, 40)
    b = compute_features(b_grid, cfg)
    spy = compute_features(_daily_grid(40), cfg)
    a["symbol"], b["symbol"], spy["symbol"] = "AAPL", "MSFT", "SPY"
    out = attach_cross_section_features({"AAPL": a, "MSFT": b, "SPY": spy}, cfg)
    mid = 20
    z_a = float(out["AAPL"]["cs_ret_1"].iloc[mid])
    z_b = float(out["MSFT"]["cs_ret_1"].iloc[mid])
    assert z_a == pytest.approx(-z_b, abs=1e-5)
    assert abs(z_a + z_b) < 1e-5
    spiked = b.copy()
    spiked.loc[spiked.index[-1], "ret_1"] = 9.0
    alt = attach_cross_section_features({"AAPL": a.copy(), "MSFT": spiked, "SPY": spy.copy()}, cfg)
    assert out["AAPL"]["cs_ret_1"].iloc[mid] == pytest.approx(
        float(alt["AAPL"]["cs_ret_1"].iloc[mid]), abs=1e-8
    )


def test_cs_ridge_recovers_within_date_signal_pooled_ridge_misses():
    """Market-dominated pooled OLS vs date-demeaned CS ridge."""
    n_dates, n_names, f = 30, 10, 4
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    mkt = np.repeat(np.linspace(-2, 2, n_dates), n_names)
    rng = np.random.default_rng(1)
    cs = rng.normal(size=n_dates * n_names)
    # Feature 0 = market + a bit of CS; the CS target is in feature 1.
    x = rng.normal(size=(n_dates * n_names, f)).astype(np.float32)
    x[:, 0] = (mkt + 0.05 * cs).astype(np.float32)
    x[:, 1] = cs.astype(np.float32)
    y = (2.0 * mkt + 1.0 * cs).astype(np.float32)
    symbols = []
    for j in range(n_names):
        idx = np.arange(n_dates) * n_names + j
        symbols.append(
            SymbolArrays(
                symbol=f"S{j}",
                features=x[idx],
                target=y[idx],
                scale=np.ones(n_dates, dtype=np.float32),
                valid=np.ones(n_dates, dtype=bool),
                dates=np.arange(n_dates, dtype=np.int64),
            )
        )
    mean = np.zeros(f, dtype=np.float32)
    std = np.ones(f, dtype=np.float32)
    w_cs, _b, ic_cs = fit_ridge_readout(
        symbols, mean, std, ridge=1e-3, cs_demean=True, min_names=8
    )
    w_pool, _b2, ic_pool = fit_ridge_readout(
        symbols, mean, std, ridge=1e-3, cs_demean=False, min_names=8
    )
    assert abs(w_cs[1]) > abs(w_cs[0])
    assert ic_cs > 0.7
    # Pooled fit is allowed to grab the market; it must not beat CS on CS IC.
    assert abs(w_pool[0]) > abs(w_cs[0]) or ic_pool < ic_cs


def test_universe_liquid_drops_non_members(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_daily_parquet(data_dir / "AAPL_daily.parquet", "AAPL", 120, "2018-01-02")
    _write_daily_parquet(data_dir / "ZZZZ_daily.parquet", "ZZZZ", 120, "2018-01-02")
    cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=16,
        stride=1,
        min_context=4,
        warmup_bars=8,
        vol_halflife=5,
        z_window=10,
        z_min_periods=5,
        universe="liquid",
        residual_target=False,
        eval_last_bar=True,
        cross_section_min_names=99,
        allow_mixed_prices=True,
    )
    bundle = build_datasets(cfg, log_fn=None)
    names = {m["symbol"] for m in bundle["meta"]}
    assert names == {"AAPL"}
    assert "ZZZZ" not in names


def test_mean_cs_stats_reports_tstat_and_spearman():
    from forecast.training import mean_cs_stats

    pred = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    target = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    dates = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    stats = mean_cs_stats(pred, target, dates, min_names=3)
    assert stats["cs_ic"] == pytest.approx(1.0)
    assert stats["cs_ic_spearman"] == pytest.approx(1.0)
    assert stats["cs_n_dates"] == pytest.approx(2.0)
