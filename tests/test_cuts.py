"""Date-cut parsing and session-rank recency (labels stay put)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from forecast.config import DataConfig, ForecastTrainConfig
from forecast.cuts import (
    CUTS_SCHEMA,
    CutsError,
    apply_phase,
    cuts_from_mapping,
    default_cuts,
    exclusive_end,
    load_cuts,
    parse_iso_date,
    resolve_cuts_path,
    session_rank_weights,
    val_gate_commands,
)
from forecast.data import build_datasets
from forecast.ridge import _prepare_cs_design, fit_ridge_xy, ridge_kwargs_from_train_cfg
from forecast.training import (
    _fmt,
    _split_has_metrics,
    build_arg_parser,
    configs_from_cli,
)


def _write_daily_parquet(path: Path, symbol: str, n: int, start: str, drift: float = 0.01) -> None:
    import pandas as pd

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


def test_in_repo_cuts_match_authoritative_windows():
    cuts = default_cuts()
    assert cuts.schema == CUTS_SCHEMA
    assert cuts.pretrain.start == "1993-01-29"
    assert cuts.pretrain.train_end == "2022-09-08"
    assert cuts.pretrain.val_end == "2023-09-08"
    assert cuts.pretrain.test_end == "2023-09-11"
    assert cuts.pretrain.time_upweight_recent is False
    assert cuts.pretrain.early_stop_sessions == 252
    assert cuts.finetune.start == "2023-09-11"
    assert cuts.finetune.train_end == "2025-10-17"
    assert cuts.finetune.val_end == "2026-04-01"
    assert cuts.finetune.test_through == "2026-09-11"
    assert cuts.finetune.test_end == "2026-09-12"
    assert cuts.finetune.time_upweight_recent is True
    assert cuts.finetune.halflife_sessions == pytest.approx(126.0)
    assert cuts.dynamic_weights["rewrite_labels"] is False
    assert cuts.val_gate["book"] == "live_locate"
    assert cuts.val_gate["metric"] == "unlevered_net_ir"
    assert float(cuts.val_gate["baseline_ir"]) == pytest.approx(5.58)


def test_parse_iso_date_rejects_garbage():
    assert parse_iso_date("2022-09-08", field="x") == "2022-09-08"
    with pytest.raises(CutsError, match="YYYY-MM-DD"):
        parse_iso_date("09/08/2022", field="x")
    with pytest.raises(CutsError, match="required"):
        parse_iso_date("", field="x")


def test_exclusive_end_from_test_through():
    assert exclusive_end(test_through="2026-09-11", field="ft") == "2026-09-12"
    assert exclusive_end(test_end="2026-09-12", field="ft") == "2026-09-12"
    with pytest.raises(CutsError, match="disagrees"):
        exclusive_end(test_end="2026-09-11", test_through="2026-09-11", field="ft")


def test_cuts_reject_label_rewrite_and_pretrain_overlap(tmp_path: Path):
    payload = json.loads(Path("forecast/pretrain_finetune_cuts.json").read_text())
    payload["dynamic_weights"]["rewrite_labels"] = True
    with pytest.raises(CutsError, match="rewrite"):
        cuts_from_mapping(payload)
    payload = json.loads(Path("forecast/pretrain_finetune_cuts.json").read_text())
    payload["pretrain"]["test_end"] = "2023-09-15"
    payload["pretrain"].pop("test_through", None)
    with pytest.raises(CutsError, match="before FT"):
        cuts_from_mapping(payload)
    payload = json.loads(Path("forecast/pretrain_finetune_cuts.json").read_text())
    payload["pretrain"]["time_upweight_recent"] = True
    payload["pretrain"]["halflife_sessions"] = 126
    with pytest.raises(CutsError, match="frozen"):
        cuts_from_mapping(payload)


def test_session_rank_weights_do_not_touch_labels():
    dates = np.repeat(np.arange(0, 200, dtype=np.int64), 4)
    y = np.linspace(-1.0, 1.0, dates.size)
    y_copy = y.copy()
    w = session_rank_weights(dates, 126.0)
    assert y == pytest.approx(y_copy)
    latest = dates == dates.max()
    oldest = dates == dates.min()
    assert w[latest] == pytest.approx(1.0)
    # 199 sessions earlier than the latest unique date.
    assert w[oldest] == pytest.approx(0.5 ** (199 / 126.0))
    mid = dates == (dates.max() - 126)
    assert w[mid] == pytest.approx(0.5)
    assert session_rank_weights(dates, 0.0) == pytest.approx(np.ones_like(w))


def test_ridge_session_halflife_leaves_y_unchanged():
    rng = np.random.default_rng(0)
    n_dates, n_names, f = 40, 10, 3
    dates = np.repeat(np.arange(n_dates, dtype=np.int64), n_names)
    x = rng.normal(size=(n_dates * n_names, f))
    y = x[:, 0] + 0.05 * rng.normal(size=n_dates * n_names)
    y_before = y.copy()
    prepared = _prepare_cs_design(
        x,
        y,
        dates,
        min_names=8,
        cs_demean=True,
        cs_zscore=False,
        rank_target=False,
        feature_mask_bool=None,
        date_halflife=0.0,
        session_halflife=126.0,
    )
    assert prepared is not None
    _xd, _yd, w, used = prepared
    assert y == pytest.approx(y_before)
    uniq = np.unique(used)
    ranked = session_rank_weights(uniq, 126.0)
    assert ranked[-1] == pytest.approx(1.0)
    assert ranked[0] == pytest.approx(0.5 ** ((n_dates - 1) / 126.0))
    assert w[0] == pytest.approx(ranked[0])
    assert w[-1] == pytest.approx(ranked[-1])
    w0, _, _ = fit_ridge_xy(x, y, dates, ridge=1.0, min_names=8, session_halflife=0.0)
    w1, _, _ = fit_ridge_xy(x, y, dates, ridge=1.0, min_names=8, session_halflife=126.0)
    assert not np.allclose(w0, w1)


def test_apply_phase_sets_windows_and_finetune_weights_only():
    cuts = default_cuts()
    data = DataConfig()
    train = ForecastTrainConfig()
    d0, t0, p0 = apply_phase(data, train, cuts, "pretrain")
    assert d0.train_from == "1993-01-29"
    assert d0.train_end == "2022-09-08"
    assert d0.val_end == "2023-09-08"
    assert d0.test_end == "2023-09-11"
    assert t0.time_upweight_recent is False
    assert t0.time_upweight_halflife_sessions == pytest.approx(0.0)
    assert t0.forecast_phase == "pretrain"
    d1, t1, p1 = apply_phase(data, train, cuts, "finetune")
    assert d1.train_from == "2023-09-11"
    assert d1.train_end == "2025-10-17"
    assert d1.val_end == "2026-04-01"
    assert d1.test_end == "2026-09-12"
    assert t1.time_upweight_recent is True
    assert t1.time_upweight_halflife_sessions == pytest.approx(126.0)
    assert p1.time_upweight_recent is True
    kw = ridge_kwargs_from_train_cfg(t1, {"cs_min_names": 8})
    assert kw["session_halflife"] == pytest.approx(126.0)
    assert kw["date_halflife"] == pytest.approx(0.0)
    kw0 = ridge_kwargs_from_train_cfg(t0, {"cs_min_names": 8})
    assert kw0["session_halflife"] == pytest.approx(0.0)


def test_cli_phase_finetune_reads_cuts_json():
    args = build_arg_parser().parse_args(
        ["--cuts-json", "forecast/pretrain_finetune_cuts.json", "--phase", "finetune"]
    )
    data_cfg, _model, train_cfg = configs_from_cli(args)
    assert data_cfg.train_from == "2023-09-11"
    assert data_cfg.train_end == "2025-10-17"
    assert data_cfg.val_end == "2026-04-01"
    assert data_cfg.test_end == "2026-09-12"
    assert data_cfg.label_return == "overnight"
    assert train_cfg.time_upweight_recent is True
    assert train_cfg.time_upweight_halflife_sessions == pytest.approx(126.0)
    assert train_cfg.forecast_phase == "finetune"
    assert train_cfg.skip_only is True
    assert train_cfg.num_workers == 0


def test_cli_phase_pretrain_has_no_recency():
    args = build_arg_parser().parse_args(
        [
            "--cuts-json",
            "forecast/pretrain_finetune_cuts.json",
            "--phase",
            "pretrain",
            "--label-return",
            "overnight",
        ]
    )
    data_cfg, _model, train_cfg = configs_from_cli(args)
    assert data_cfg.train_end == "2022-09-08"
    assert data_cfg.val_end == "2023-09-08"
    assert data_cfg.test_end == "2023-09-11"
    assert train_cfg.time_upweight_recent is False
    assert train_cfg.forecast_phase == "pretrain"


def test_cli_explicit_dates_override_fractions():
    args = build_arg_parser().parse_args(
        [
            "--train-from",
            "2020-01-02",
            "--train-end",
            "2021-06-01",
            "--val-end",
            "2021-12-01",
            "--test-through",
            "2022-03-01",
            "--time-upweight-recent",
            "--time-upweight-halflife",
            "126",
        ]
    )
    data_cfg, _model, train_cfg = configs_from_cli(args)
    assert data_cfg.train_from == "2020-01-02"
    assert data_cfg.train_end == "2021-06-01"
    assert data_cfg.val_end == "2021-12-01"
    assert data_cfg.test_end == "2022-03-02"
    assert train_cfg.time_upweight_recent is True
    assert train_cfg.time_upweight_halflife_sessions == pytest.approx(126.0)


def test_explicit_cuts_assign_sessions(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, name in enumerate(("AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "PG", "HD")):
        _write_daily_parquet(
            data_dir / f"{name}_daily.parquet",
            name,
            260,
            "2020-01-02",
            0.01 + 0.001 * i,
        )
    cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=8,
        stride=1,
        min_context=2,
        warmup_bars=4,
        vol_halflife=5,
        z_window=8,
        z_min_periods=4,
        global_calendar_split=True,
        residual_target=False,
        eval_last_bar=True,
        cross_section_min_names=8,
        allow_mixed_prices=True,
        equities_only=True,
        sector_residual=False,
        train_from="2020-01-02",
        train_end="2020-06-01",
        val_end="2020-09-01",
        test_end="2020-12-01",
    )
    bundle = build_datasets(cfg, log_fn=None)
    assert bundle["cross_section"] is True
    train_ds, val_ds, test_ds = (
        bundle["datasets"]["train"],
        bundle["datasets"]["val"],
        bundle["datasets"]["test"],
    )
    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0
    train_days = {int(train_ds[i][4][0]) for i in range(len(train_ds))}
    val_days = {int(val_ds[i][4][0]) for i in range(len(val_ds))}
    test_days = {int(test_ds[i][4][0]) for i in range(len(test_ds))}
    assert train_days.isdisjoint(val_days)
    assert val_days.isdisjoint(test_days)
    assert max(train_days) < min(val_days)
    assert max(val_days) < min(test_days)
    ends = {m["train_end"] for m in bundle["meta"]}
    assert ends == {"2020-06-01"}
    assert {m["val_end"] for m in bundle["meta"]} == {"2020-09-01"}
    assert {m["test_end"] for m in bundle["meta"]} == {"2020-12-01"}


def test_empty_test_keeps_cross_section(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, name in enumerate(("AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "PG", "HD")):
        _write_daily_parquet(
            data_dir / f"{name}_daily.parquet",
            name,
            180,
            "2020-01-02",
            0.01 + 0.001 * i,
        )
    cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=8,
        stride=1,
        min_context=2,
        warmup_bars=4,
        vol_halflife=5,
        z_window=8,
        z_min_periods=4,
        global_calendar_split=True,
        residual_target=False,
        eval_last_bar=True,
        cross_section_min_names=8,
        allow_mixed_prices=True,
        equities_only=True,
        sector_residual=False,
        train_from="2020-01-02",
        train_end="2020-07-01",
        val_end="2020-10-01",
        test_end="2020-10-01",
    )
    bundle = build_datasets(cfg, log_fn=None)
    assert bundle["cross_section"] is True
    assert len(bundle["datasets"]["train"]) > 0
    assert len(bundle["datasets"]["val"]) > 0
    assert len(bundle["datasets"]["test"]) == 0


def test_resolve_cuts_prefers_desktop_panel_cache(tmp_path: Path, monkeypatch):
    desktop = tmp_path / "data" / "_panel_cache"
    desktop.mkdir(parents=True)
    payload = json.loads(Path("forecast/pretrain_finetune_cuts.json").read_text())
    (desktop / "pretrain_finetune_cuts.json").write_text(json.dumps(payload))
    found = resolve_cuts_path(data_dir=tmp_path / "data")
    assert found == desktop / "pretrain_finetune_cuts.json"
    loaded = load_cuts(found)
    assert loaded.finetune.halflife_sessions == pytest.approx(126.0)


def test_skip_only_train_uses_phase_val_window_and_writes_cuts(tmp_path: Path):
    from forecast.config import ForecastModelConfig
    from forecast.training import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, name in enumerate(("AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "PG", "HD")):
        _write_daily_parquet(
            data_dir / f"{name}_daily.parquet",
            name,
            220,
            "2020-01-02",
            0.01 + 0.001 * i,
        )
    ckpt_dir = tmp_path / "ckpt_ft"
    data_cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=8,
        stride=1,
        min_context=2,
        warmup_bars=4,
        vol_halflife=5,
        z_window=8,
        z_min_periods=4,
        residual_target=False,
        eval_last_bar=True,
        cross_section_min_names=8,
        allow_mixed_prices=True,
        equities_only=True,
        sector_residual=False,
        train_from="2020-01-02",
        train_end="2020-07-01",
        val_end="2020-10-01",
        test_end="2020-12-01",
    )
    train_cfg = ForecastTrainConfig(
        skip_only=True,
        checkpoint_dir=str(ckpt_dir),
        time_upweight_recent=True,
        time_upweight_halflife_sessions=126.0,
        forecast_phase="finetune",
        num_workers=0,
        precision="fp32",
        eval_train_split=False,
    )
    summary = train(
        data_cfg,
        ForecastModelConfig(d_model=16, n_layer=1, d_state=8),
        train_cfg,
        device=__import__("torch").device("cpu"),
        log_fn=None,
    )
    assert summary["skip_only"] is True
    assert (ckpt_dir / "best.pt").is_file()
    payload = json.loads((ckpt_dir / "phase_cuts.json").read_text())
    assert payload["phase"] == "finetune"
    assert payload["train_end"] == "2020-07-01"
    assert payload["val_end"] == "2020-10-01"
    assert payload["test_end"] == "2020-12-01"
    assert payload["time_upweight_recent"] is True
    assert payload["time_upweight_halflife_sessions"] == pytest.approx(126.0)
    assert payload["rewrite_labels"] is False
    assert "skip_only_val" in summary
    ends = {m["val_end"] for m in summary["symbols"]}
    assert ends == {"2020-10-01"}


def test_fmt_tolerates_empty_and_partial_metrics():
    assert _fmt({}) == "(empty split)"
    assert _fmt(None) == "(empty split)"
    partial = {"ic": float("nan"), "n": 0.0, "loss": float("nan")}
    text = _fmt(partial)
    assert "r2=" in text
    assert "n=0" in text
    assert _split_has_metrics({}) is False
    assert _split_has_metrics(partial) is False
    assert _split_has_metrics({"n": 12.0, "r2": 0.1}) is True


def test_skip_only_empty_test_still_writes_best_pt(tmp_path: Path):
    """Pretrain test windows=0 must not KeyError in _fmt before save."""
    from forecast.config import ForecastModelConfig
    from forecast.training import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, name in enumerate(("AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "PG", "HD")):
        _write_daily_parquet(
            data_dir / f"{name}_daily.parquet",
            name,
            180,
            "2020-01-02",
            0.01 + 0.001 * i,
        )
    ckpt_dir = tmp_path / "ckpt_pretrain"
    logs: list[str] = []
    data_cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=8,
        stride=1,
        min_context=2,
        warmup_bars=4,
        vol_halflife=5,
        z_window=8,
        z_min_periods=4,
        residual_target=False,
        eval_last_bar=True,
        cross_section_min_names=8,
        allow_mixed_prices=True,
        equities_only=True,
        sector_residual=False,
        train_from="2020-01-02",
        train_end="2020-07-01",
        val_end="2020-10-01",
        test_end="2020-10-01",
    )
    summary = train(
        data_cfg,
        ForecastModelConfig(d_model=16, n_layer=1, d_state=8),
        ForecastTrainConfig(
            skip_only=True,
            checkpoint_dir=str(ckpt_dir),
            forecast_phase="pretrain",
            num_workers=0,
            precision="fp32",
            eval_train_split=False,
        ),
        device=__import__("torch").device("cpu"),
        log_fn=logs.append,
    )
    assert (ckpt_dir / "best.pt").is_file()
    assert (ckpt_dir / "last.pt").is_file()
    assert any("skip-only test: (empty split)" in line for line in logs)
    assert not any("KeyError" in line for line in logs)
    assert float(summary["skip_only_test"].get("n", 0.0) or 0.0) == 0.0


def test_val_gate_commands_point_at_ft_live_locate():
    text = val_gate_commands()
    assert "overnight_shorting.py" in text
    assert "live-costs" in text
    assert "5.58" in text
    assert "hit-rate" in text.lower() or "not hit-rate" in text
    assert "forecast_ridge_overnight_finetune" in text
