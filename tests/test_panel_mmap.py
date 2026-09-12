"""Mmap panel feed: load API, CS train path, refuse cs_train_*.pt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.config import DataConfig, interval_data_kwargs
from forecast.data import FEATURE_NAMES, SymbolArrays, build_datasets
from forecast.panel_mmap import (
    CsTrainPtForbidden,
    assert_no_cs_train_pt,
    load_manifest,
    load_symbol_mmap,
    write_mmap_cache,
)
from forecast.synthetic import write_cs_overnight_universe


def _tiny_arrays(symbol: str = "AAA", t: int = 12, f: int = 4) -> SymbolArrays:
    rng = np.random.default_rng(0)
    return SymbolArrays(
        symbol=symbol,
        features=rng.normal(size=(t, f)).astype(np.float32),
        target=rng.normal(size=t).astype(np.float32),
        scale=np.full(t, 0.01, dtype=np.float32),
        valid=np.ones(t, dtype=bool),
        dates=np.arange(10_000, 10_000 + t, dtype=np.int64),
        overnight_r=rng.normal(size=t).astype(np.float32),
        next_split_days=np.zeros(t, dtype=np.int16),
    )


def test_assert_no_cs_train_pt():
    assert_no_cs_train_pt("data/_panel_cache/mmap_manifest.json")
    with pytest.raises(CsTrainPtForbidden, match="6.7GB"):
        assert_no_cs_train_pt("data/_panel_cache/cs_train_liquid.pt")
    with pytest.raises(CsTrainPtForbidden):
        load_manifest(Path("cs_train_foo.pt"))


def test_write_and_load_symbol_mmap(tmp_path: Path):
    cache = tmp_path / "_panel_cache"
    dest = write_mmap_cache(
        [_tiny_arrays("AAA"), _tiny_arrays("BBB")],
        cache,
        feature_names=["a", "b", "c", "d"],
        label_return="overnight",
    )
    assert dest.name == "mmap_manifest.json"
    man = load_manifest(dest)
    assert man["schema"] == "panel_mmap/v1"
    assert man["do_not_load"] == ["cs_train_*.pt"]
    packed = load_symbol_mmap(man, "AAA", mmap_mode="r")
    assert isinstance(packed.features, np.memmap)
    assert packed.features.dtype == np.float32
    assert packed.features.shape == (12, 4)
    assert packed.labels["target"].shape == (12,)
    assert float(packed.labels["overnight_r"][0]) == pytest.approx(
        float(_tiny_arrays().overnight_r[0])
    )


def test_build_datasets_reads_mmap_without_rebuilding_frames(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    write_cs_overnight_universe(data, n_names=10, n_days=80, seed=2)
    preset = interval_data_kwargs("daily")
    write_cfg = DataConfig(
        data_dir=str(data),
        interval="daily",
        horizon=1,
        seq_len=16,
        stride=1,
        min_context=4,
        warmup_bars=8,
        vol_halflife=preset["vol_halflife"],
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
        write_mmap=True,
        mmap_cache_dir=str(data / "_panel_cache"),
    )
    first = build_datasets(write_cfg, log_fn=None)
    assert first.get("mmap") is True
    manifest = Path(first["mmap_manifest"])
    assert manifest.is_file()

    def _boom(*_a, **_k):
        raise AssertionError("build_panel must not run on the mmap train path")

    monkeypatch.setattr("forecast.data.build_panel", _boom)
    read_cfg = DataConfig(
        data_dir=str(data),
        interval="daily",
        horizon=1,
        seq_len=16,
        stride=1,
        min_context=4,
        warmup_bars=8,
        vol_halflife=preset["vol_halflife"],
        z_window=16,
        z_min_periods=6,
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=6,
        allow_mixed_prices=True,
        universe="",
        label_return="overnight",
        use_mmap=True,
        write_mmap=False,
        mmap_manifest=str(manifest),
    )
    second = build_datasets(read_cfg, log_fn=None)
    assert second.get("mmap") is True
    assert second["cross_section"] is True
    assert isinstance(second["train_symbols"][0].features, np.memmap)
    assert second["train_symbols"][0].features.dtype == np.float32
    x, y, mask, *_ = second["datasets"]["train"][0]
    assert x.ndim == 3
    assert int(mask.sum()) >= 1
    # Feature count matches the live FEATURE_NAMES contract.
    assert x.shape[-1] == len(FEATURE_NAMES)


def test_mmap_load_applies_live_factors_tape(tmp_path: Path, monkeypatch):
    """Live Data Manager factors tape wins over mmap-cached next_split_days."""
    from forecast.pit import factor_parquet_path

    data = tmp_path / "data"
    write_cs_overnight_universe(data, n_names=8, n_days=70, seed=6)
    preset = interval_data_kwargs("daily")
    write_cfg = DataConfig(
        data_dir=str(data),
        interval="daily",
        horizon=1,
        seq_len=12,
        stride=1,
        min_context=4,
        warmup_bars=6,
        vol_halflife=preset["vol_halflife"],
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
        write_mmap=True,
        mmap_cache_dir=str(data / "_panel_cache"),
    )
    first = build_datasets(write_cfg, log_fn=None)
    s00 = next(s for s in first["train_symbols"] if s.symbol == "S00")
    assert int(s00.valid.sum()) > 0
    dest = factor_parquet_path("S00", data_dir=data)
    dest.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(s00.dates.astype("datetime64[D]")),
            "next_split_days": np.ones(len(s00.dates), dtype=np.int16),
        }
    ).to_parquet(dest)

    def _boom(*_a, **_k):
        raise AssertionError("build_panel must not run on the mmap train path")

    monkeypatch.setattr("forecast.data.build_panel", _boom)
    read_cfg = DataConfig(
        data_dir=str(data),
        interval="daily",
        horizon=1,
        seq_len=12,
        stride=1,
        min_context=4,
        warmup_bars=6,
        vol_halflife=preset["vol_halflife"],
        z_window=12,
        z_min_periods=4,
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=5,
        allow_mixed_prices=True,
        universe="",
        label_return="overnight",
        use_mmap=True,
        write_mmap=False,
        mmap_manifest=str(first["mmap_manifest"]),
    )
    second = build_datasets(read_cfg, log_fn=None)
    s00_m = next(s for s in second["train_symbols"] if s.symbol == "S00")
    assert int(s00_m.valid.sum()) == 0
    assert int((np.asarray(s00_m.next_split_days) == 1).sum()) == len(s00_m.next_split_days)
