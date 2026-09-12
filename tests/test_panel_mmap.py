"""Mmap panel feed: load API, CS train path, refuse cs_train_*.pt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.config import DataConfig, interval_data_kwargs
from forecast.data import FEATURE_NAMES, SymbolArrays, build_datasets
from forecast.panel_mmap import (
    CsTrainPtForbidden,
    DM_SCHEMA,
    MmapError,
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


def _write_raw(path: Path, values: np.ndarray) -> None:
    arr = np.ascontiguousarray(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def test_load_manifest_adapts_numpy_mmap_v1_pointer(tmp_path: Path):
    """DM pointer file + symbols list → in-memory map; hashed store is not rewritten."""
    cache = tmp_path / "data" / "_panel_cache"
    store = cache / "mmap_fc9602c813bb"
    write_mmap_cache(
        [_tiny_arrays("AAA"), _tiny_arrays("BBB")],
        store,
        feature_names=["a", "b", "c", "d"],
        label_return="overnight",
    )
    written = json.loads((store / "mmap_manifest.json").read_text(encoding="utf-8"))
    inner = {
        "schema": DM_SCHEMA,
        "dtype": "float32",
        "n_features": 4,
        "feature_names": ["a", "b", "c", "d"],
        "symbols": ["AAA", "BBB"],
    }
    (store / "manifest.json").write_text(json.dumps(inner), encoding="utf-8")
    pointer = cache / "mmap_manifest.json"
    pointer.write_text(
        json.dumps(
            {
                "schema": DM_SCHEMA,
                "manifest": "mmap_fc9602c813bb/manifest.json",
                "symbols": ["AAA", "BBB"],
            }
        ),
        encoding="utf-8",
    )
    assert (store / "AAA.features.f32").is_file()
    man = load_manifest(pointer)
    assert man["schema"] == DM_SCHEMA
    assert man["_adapted"] is True
    assert set(man["symbols"]) == {"AAA", "BBB"}
    assert isinstance(man["symbols"], dict)
    assert Path(man["_cache_dir"]) == store
    packed = load_symbol_mmap(man, "AAA", mmap_mode="r")
    assert isinstance(packed.features, np.memmap)
    assert packed.features.dtype == np.float32
    assert packed.features.shape == (12, 4)
    assert float(packed.labels["overnight_r"][0]) == pytest.approx(
        float(_tiny_arrays().overnight_r[0])
    )
    # Adapter is read-only: the hashed store stays numpy_mmap_v1 + raw float32.
    assert json.loads((store / "manifest.json").read_text())["schema"] == DM_SCHEMA
    assert written["schema"] == "panel_mmap/v1"


def test_numpy_mmap_v1_symbols_list_of_dicts(tmp_path: Path):
    cache = tmp_path / "_panel_cache"
    dest = write_mmap_cache(
        [_tiny_arrays("AAA")],
        cache,
        feature_names=["a", "b", "c", "d"],
    )
    entry = json.loads(dest.read_text())["symbols"]["AAA"]
    dest.write_text(
        json.dumps(
            {
                "schema": DM_SCHEMA,
                "n_features": 4,
                "feature_names": ["a", "b", "c", "d"],
                "symbols": [{"symbol": "AAA", **entry}],
            }
        ),
        encoding="utf-8",
    )
    packed = load_symbol_mmap(dest, "AAA", mmap_mode="r")
    assert packed.features.shape == (12, 4)
    assert packed.T == 12


def test_numpy_mmap_v1_stacked_arrays(tmp_path: Path):
    store = tmp_path / "mmap_fc9602c813bb"
    store.mkdir()
    a, b = _tiny_arrays("AAA"), _tiny_arrays("BBB")
    _write_raw(store / "features.f32", np.vstack([a.features, b.features]))
    _write_raw(store / "target.f32", np.concatenate([a.target, b.target]))
    _write_raw(store / "scale.f32", np.concatenate([a.scale, b.scale]))
    _write_raw(
        store / "valid.u8",
        np.concatenate([a.valid, b.valid]).astype(np.uint8),
    )
    _write_raw(store / "dates.i64", np.concatenate([a.dates, b.dates]))
    _write_raw(
        store / "overnight_r.f32",
        np.concatenate([a.overnight_r, b.overnight_r]),
    )
    (store / "manifest.json").write_text(
        json.dumps(
            {
                "schema": DM_SCHEMA,
                "n_features": 4,
                "feature_names": ["a", "b", "c", "d"],
                "symbols": ["AAA", "BBB"],
                "arrays": {
                    "features": "features.f32",
                    "target": "target.f32",
                    "scale": "scale.f32",
                    "valid": "valid.u8",
                    "dates": "dates.i64",
                    "overnight_r": "overnight_r.f32",
                },
                "index": {"AAA": [0, 12], "BBB": [12, 24]},
            }
        ),
        encoding="utf-8",
    )
    pointer = tmp_path / "mmap_manifest.json"
    pointer.write_text(
        json.dumps({"schema": DM_SCHEMA, "manifest": "mmap_fc9602c813bb/manifest.json"}),
        encoding="utf-8",
    )
    packed = load_symbol_mmap(pointer, "BBB", mmap_mode="r")
    assert packed.features.shape == (12, 4)
    np.testing.assert_allclose(np.asarray(packed.features), b.features)
    np.testing.assert_allclose(np.asarray(packed.target), b.target)


def test_unsupported_mmap_schema_still_rejected(tmp_path: Path):
    dest = tmp_path / "mmap_manifest.json"
    dest.write_text(json.dumps({"schema": "other/v9", "symbols": {"AAA": {}}}), encoding="utf-8")
    with pytest.raises(MmapError, match="unsupported mmap schema"):
        load_manifest(dest)


def test_build_datasets_reads_numpy_mmap_v1_pointer(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    write_cs_overnight_universe(data, n_names=8, n_days=70, seed=7)
    preset = interval_data_kwargs("daily")
    hashed = data / "_panel_cache" / "mmap_fc9602c813bb"
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
        mmap_cache_dir=str(hashed),
    )
    first = build_datasets(write_cfg, log_fn=None)
    written = json.loads(Path(first["mmap_manifest"]).read_text(encoding="utf-8"))
    names = sorted(written["symbols"])
    (hashed / "manifest.json").write_text(
        json.dumps(
            {
                "schema": DM_SCHEMA,
                "dtype": "float32",
                "n_features": written["n_features"],
                "feature_names": written["feature_names"],
                "label_return": "overnight",
                "symbols": names,
            }
        ),
        encoding="utf-8",
    )
    pointer = data / "_panel_cache" / "mmap_manifest.json"
    pointer.write_text(
        json.dumps(
            {
                "schema": DM_SCHEMA,
                "manifest": "mmap_fc9602c813bb/manifest.json",
                "symbols": names,
            }
        ),
        encoding="utf-8",
    )

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
        mmap_manifest=str(pointer),
    )
    second = build_datasets(read_cfg, log_fn=None)
    assert second.get("mmap") is True
    assert isinstance(second["train_symbols"][0].features, np.memmap)
    assert second["train_symbols"][0].features.dtype == np.float32
    x, y, mask, *_ = second["datasets"]["train"][0]
    assert x.ndim == 3
    assert int(mask.sum()) >= 1
