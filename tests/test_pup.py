"""(2a) P(up) overnight head: labels, PIT mask, cover floor, VAL gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from forecast.config import DataConfig, ForecastModelConfig, ForecastTrainConfig
from forecast.data import FEATURE_NAMES, build_datasets
from forecast.model import ReturnForecaster
from forecast.pit import (
    factor_parquet_path,
    is_member_asof,
    load_liquid_membership,
    load_pit_side_tape,
    mask_membership_asof,
    mask_overnight_pit,
    next_split_days_aligned,
    overnight_up_labels,
)
from forecast.pup import (
    MIN_COVER,
    TARGET_UP_PCT,
    apply_pup_logistic,
    decide_pup_promote,
    fit_pup_logistic,
    pup_sleeve_mask,
    score_pup_sleeve,
)
from forecast.synthetic import write_cs_overnight_universe
from forecast.training import apply_pup_skip, build_arg_parser, configs_from_cli


def test_overnight_up_labels_pit_mask():
    r = np.array([0.01, -0.02, 0.03, 0.04], dtype=np.float64)
    valid = np.array([True, True, True, True])
    days = np.array([0, 0, 1, 0], dtype=np.int16)
    y = overnight_up_labels(r, valid=valid, next_split_days=days)
    assert y[0] == 1.0
    assert y[1] == 0.0
    assert np.isnan(y[2])
    assert y[3] == 1.0
    masked = mask_overnight_pit(valid, days)
    assert masked.tolist() == [True, True, False, True]


def test_pit_side_tape_next_split_days(tmp_path: Path):
    dates = pd.bdate_range("2020-01-02", periods=5)
    dest = factor_parquet_path("AAA", data_dir=tmp_path)
    dest.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": dates,
            "next_split_days": [3, 2, 1, 0, 4],
        }
    ).to_parquet(dest)
    tape = load_pit_side_tape(tmp_path, symbols=["AAA"])
    assert "AAA" in tape
    keys = ((pd.to_datetime(dates) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")).astype(
        np.int64
    )
    aligned = next_split_days_aligned(keys.to_numpy(), tape["AAA"], length=5)
    assert aligned[2] == 1
    assert int(mask_overnight_pit(np.ones(5, dtype=bool), aligned).sum()) == 4
    assert dest == tmp_path / "_pit" / "factors" / "AAA_daily_factors.parquet"


def _day(iso: str) -> int:
    return int((pd.Timestamp(iso) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D"))


def test_membership_asof_ranges_and_by_date(tmp_path: Path):
    path = tmp_path / "liquid_membership.json"
    path.write_text(
        json.dumps(
            {
                "by_date": {
                    "2020-01-01": ["AAA", "BBB"],
                    "2020-06-01": ["BBB"],
                }
            }
        ),
        encoding="utf-8",
    )
    book = load_liquid_membership(path=path)
    assert is_member_asof("AAA", _day("2020-03-01"), book) is True
    assert is_member_asof("AAA", _day("2020-07-01"), book) is False
    assert is_member_asof("BBB", _day("2020-07-01"), book) is True
    dates = np.array([_day("2020-03-01"), _day("2020-07-01")], dtype=np.int64)
    out = mask_membership_asof(np.array([True, True]), dates, "AAA", book)
    np.testing.assert_array_equal(out, np.array([True, False]))

    ranges = {
        "membership": {
            "CCC": {"start": "2020-02-01", "end": "2020-02-28"},
        }
    }
    assert is_member_asof("CCC", _day("2020-02-15"), ranges) is True
    assert is_member_asof("CCC", _day("2020-03-01"), ranges) is False
    assert is_member_asof("AAA", _day("2020-02-15"), ranges) is False


def test_pup_logistic_recovers_planted_feature():
    rng = np.random.default_rng(1)
    n = 400
    z = rng.normal(size=n)
    x = np.column_stack([z, rng.normal(size=n)])
    y = (z > 0).astype(np.float64)
    spec = fit_pup_logistic(x, y, ridge=0.1)
    p = apply_pup_logistic(x, spec)
    assert p.shape == (n,)
    assert float((p[y > 0.5].mean()) - (p[y < 0.5].mean())) > 0.2


def test_pup_sleeve_meets_cover_floor():
    dates = np.repeat(np.arange(20), 10)
    # Within each date, last names have the highest P(up) and go up.
    p = np.tile(np.linspace(0.05, 0.95, 10), 20)
    r = np.where(p >= 0.8, 0.01, -0.01)
    mask = pup_sleeve_mask(p, dates, min_cover=MIN_COVER, min_names=2)
    cover = float(mask.mean())
    assert cover + 1e-12 >= MIN_COVER
    row = score_pup_sleeve(r, mask, n_full=int(dates.size))
    assert row["cover"] + 1e-12 >= MIN_COVER
    assert row["up_pct"] > 50.0


def test_decide_pup_promote_val_only():
    hit = decide_pup_promote({"up_pct": 61.0, "cover": 0.06})
    assert hit["promote"] is True
    assert hit["reached_60"] is True
    miss = decide_pup_promote({"up_pct": 59.07, "cover": 0.067})
    assert miss["promote"] is False
    assert miss["reached_60"] is False
    assert "59.07" in miss["reason"]
    thin = decide_pup_promote({"up_pct": 80.0, "cover": 0.01})
    assert thin["promote"] is False


def test_cli_wires_mmap_and_pup_head():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--pup-head",
            "--mmap-manifest",
            "data/_panel_cache/mmap_manifest.json",
            "--pit-dir",
            "data/_pit",
            "--num-workers",
            "0",
            "--skip-only",
            "--label-return",
            "overnight",
        ]
    )
    data_cfg, _model_cfg, train_cfg = configs_from_cli(args)
    assert train_cfg.pup_head is True
    assert train_cfg.num_workers == 0
    assert data_cfg.mmap_manifest.endswith("mmap_manifest.json")
    assert data_cfg.pit_dir == "data/_pit"
    assert data_cfg.label_return == "overnight"


def test_apply_pup_skip_copies_weights(tmp_path: Path):
    data = tmp_path / "data"
    write_cs_overnight_universe(data, n_names=10, n_days=90, seed=4)
    cfg = DataConfig(
        data_dir=str(data),
        interval="daily",
        horizon=1,
        seq_len=16,
        stride=1,
        min_context=4,
        warmup_bars=8,
        vol_halflife=8,
        z_window=16,
        z_min_periods=6,
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=6,
        allow_mixed_prices=True,
        universe="",
        label_return="overnight",
        use_mmap=False,
    )
    bundle = build_datasets(cfg, log_fn=None)
    model = ReturnForecaster(ForecastModelConfig(n_features=len(FEATURE_NAMES), n_layer=1, d_model=16))
    payload = apply_pup_skip(
        model,
        bundle,
        ForecastTrainConfig(pup_head=True, pup_ridge=1.0, skip_only=True),
        torch.device("cpu"),
    )
    assert payload["kind"] == "p_up_overnight"
    assert payload["test_report_only"] is True
    assert "val" in payload["splits"]
    assert payload["splits"]["val"]["cover"] + 1e-12 >= MIN_COVER or payload["splits"]["val"]["n_sleeve"] == 0
    assert float(model.up_skip.weight.abs().sum()) > 0.0
    gate = payload["gate"]
    assert "reached_60" in gate
    assert TARGET_UP_PCT == 60.0


def test_pit_masks_overnight_on_dataset(tmp_path: Path):
    data = tmp_path / "data"
    write_cs_overnight_universe(data, n_names=8, n_days=70, seed=5)
    cfg = DataConfig(
        data_dir=str(data),
        interval="daily",
        horizon=1,
        seq_len=12,
        stride=1,
        min_context=4,
        warmup_bars=6,
        vol_halflife=6,
        z_window=12,
        z_min_periods=4,
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=5,
        allow_mixed_prices=True,
        universe="",
        label_return="overnight",
        use_mmap=False,
    )
    clean = build_datasets(cfg, log_fn=None)
    dates = clean["train_symbols"][0].dates
    assert dates is not None
    dest = factor_parquet_path("S00", data_dir=data)
    dest.parent.mkdir(parents=True)
    # Data Manager contract: _pit/factors/{SYM}_daily_factors.parquet
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(dates.astype("datetime64[D]")),
            "next_split_days": np.ones(len(dates), dtype=np.int16),
        }
    ).to_parquet(dest)
    masked = build_datasets(cfg, log_fn=None)
    s00 = next(s for s in masked["train_symbols"] if s.symbol == "S00")
    assert int(s00.valid.sum()) == 0
    y = overnight_up_labels(s00.overnight_r, valid=s00.valid, next_split_days=s00.next_split_days)
    assert not np.isfinite(y).any()


def _overnight_cfg(data: Path, **extra) -> DataConfig:
    kwargs = dict(
        data_dir=str(data),
        interval="daily",
        horizon=1,
        seq_len=12,
        stride=1,
        min_context=4,
        warmup_bars=6,
        vol_halflife=6,
        z_window=12,
        z_min_periods=4,
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=5,
        allow_mixed_prices=True,
        universe="",
        label_return="overnight",
        use_mmap=False,
    )
    kwargs.update(extra)
    return DataConfig(**kwargs)


def test_membership_asof_masks_dataset(tmp_path: Path):
    """Optional liquid_membership.json as-of drops a name from overnight labels."""
    data = tmp_path / "data"
    write_cs_overnight_universe(data, n_names=8, n_days=70, seed=5)
    cfg = _overnight_cfg(data)
    pit = data / "_pit"
    pit.mkdir(parents=True)
    members = [f"S{i:02d}" for i in range(1, 8)]
    (pit / "liquid_membership.json").write_text(
        json.dumps({"symbols": members}), encoding="utf-8"
    )
    masked = build_datasets(cfg, log_fn=None)
    s00 = next(s for s in masked["train_symbols"] if s.symbol == "S00")
    s01 = next(s for s in masked["train_symbols"] if s.symbol == "S01")
    assert int(s00.valid.sum()) == 0
    assert int(s01.valid.sum()) > 0
    from forecast.pup import labelled_pup_rows

    x, y, _dates = labelled_pup_rows(
        masked["train_symbols"], masked["feature_mean"], masked["feature_std"]
    )
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] > 0


def test_labelled_pup_rows_drop_factors_split_nights(tmp_path: Path):
    """P(up) training rows omit next_split_days==1 from the factors tape."""
    from forecast.pup import labelled_pup_rows

    data = tmp_path / "data"
    write_cs_overnight_universe(data, n_names=8, n_days=70, seed=5)
    cfg = _overnight_cfg(data)
    clean = build_datasets(cfg, log_fn=None)
    x0, y0, _d0 = labelled_pup_rows(
        clean["train_symbols"], clean["feature_mean"], clean["feature_std"]
    )
    s00_clean = next(s for s in clean["train_symbols"] if s.symbol == "S00")
    x_s00, _y_s00, _d_s00 = labelled_pup_rows(
        [s00_clean], clean["feature_mean"], clean["feature_std"]
    )
    n_s00 = int(x_s00.shape[0])
    assert n_s00 > 0

    dest = factor_parquet_path("S00", data_dir=data)
    dest.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(s00_clean.dates.astype("datetime64[D]")),
            "next_split_days": np.ones(len(s00_clean.dates), dtype=np.int16),
        }
    ).to_parquet(dest)
    masked = build_datasets(cfg, log_fn=None)
    x1, y1, _d1 = labelled_pup_rows(
        masked["train_symbols"], masked["feature_mean"], masked["feature_std"]
    )
    s00 = next(s for s in masked["train_symbols"] if s.symbol == "S00")
    y_s00 = overnight_up_labels(
        s00.overnight_r, valid=s00.valid, next_split_days=s00.next_split_days
    )
    assert not np.isfinite(y_s00).any()
    assert x1.shape[0] == x0.shape[0] - n_s00
    assert y1.shape[0] == x1.shape[0]
