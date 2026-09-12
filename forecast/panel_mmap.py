"""Data Manager mmap panel feed (desktop contract, in-repo).

Manifest: ``data/_panel_cache/mmap_manifest.json``

Load API::

    manifest = load_manifest(path)
    packed = load_symbol_mmap(manifest, symbol, mmap_mode="r")
    packed.features  # numpy.memmap float32 [T, F]
    packed.labels    # target / scale / valid / dates / overnight_r / ...

Train CS batching must read these memmaps. Do **not** rebuild per-symbol
DataFrames on the hot path, and do **not** load or pickle ``cs_train_*.pt``
(6.7GB Windows double-buffer). ``num_workers`` stays 0.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from mamba_lm.paths import resolve_path

if TYPE_CHECKING:
    from forecast.data import SymbolArrays


SCHEMA = "panel_mmap/v1"
MANIFEST_NAME = "mmap_manifest.json"
DEFAULT_CACHE_REL = Path("data") / "_panel_cache"
DEFAULT_MANIFEST_REL = DEFAULT_CACHE_REL / MANIFEST_NAME
CS_TRAIN_PT_RE = re.compile(r"^cs_train_.*\.pt$", re.IGNORECASE)

FEATURES_SUFFIX = "features.f32"
LABEL_SPECS: tuple[tuple[str, str, np.dtype, bool], ...] = (
    ("target", "target.f32", np.dtype(np.float32), True),
    ("scale", "scale.f32", np.dtype(np.float32), True),
    ("valid", "valid.u8", np.dtype(np.uint8), True),
    ("dates", "dates.i64", np.dtype(np.int64), True),
    ("overnight_r", "overnight_r.f32", np.dtype(np.float32), False),
    ("next_split_days", "next_split_days.i16", np.dtype(np.int16), False),
)


class MmapError(ValueError):
    """Invalid mmap manifest or symbol payload."""


class CsTrainPtForbidden(RuntimeError):
    """``cs_train_*.pt`` is the closed 6.7GB double-buffer. Use mmap instead."""


@dataclass
class SymbolMmap:
    """One symbol's mmap'd panel: float32 features ``[T, F]`` plus labels."""

    symbol: str
    features: np.memmap
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()
    T: int = 0
    F: int = 0
    mmap_mode: str = "r"
    cache_dir: Path | None = None

    @property
    def target(self) -> np.ndarray:
        return self.labels["target"]

    @property
    def scale(self) -> np.ndarray:
        return self.labels["scale"]

    @property
    def valid(self) -> np.ndarray:
        return np.asarray(self.labels["valid"], dtype=bool)

    @property
    def dates(self) -> np.ndarray | None:
        raw = self.labels.get("dates")
        return None if raw is None else np.asarray(raw, dtype=np.int64)


def is_cs_train_pt(path: str | Path) -> bool:
    return bool(CS_TRAIN_PT_RE.match(Path(path).name))


def assert_no_cs_train_pt(path: str | Path | None) -> None:
    """Refuse the materialized CS ``.pt`` double-buffer (Windows / num_workers)."""
    if path is None:
        return
    raw = Path(path)
    if is_cs_train_pt(raw):
        raise CsTrainPtForbidden(
            f"refusing to load/pickle {raw.name} (6.7GB CS double-buffer). "
            "Use data/_panel_cache/mmap_manifest.json and "
            "load_symbol_mmap(..., mmap_mode='r'). Keep --num-workers 0."
        )


def default_manifest_path(data_dir: str | Path = "data") -> Path:
    return resolve_path(Path(data_dir) / "_panel_cache" / MANIFEST_NAME)


def resolve_mmap_manifest(
    data_dir: str | Path = "data",
    explicit: str | Path | None = None,
) -> Path | None:
    """Return an existing manifest path, or None if the feed is absent."""
    if explicit is not None and str(explicit).strip():
        path = resolve_path(explicit)
        assert_no_cs_train_pt(path)
        if not path.is_file():
            raise FileNotFoundError(f"mmap manifest not found: {path}")
        return path
    desktop = default_manifest_path(data_dir)
    if desktop.is_file():
        assert_no_cs_train_pt(desktop)
        return desktop
    return None


def load_manifest(path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate ``mmap_manifest.json``. Mapping input is returned copied."""
    if isinstance(path, Mapping):
        payload = dict(path)
        cache_dir = Path(str(payload.get("cache_dir") or ".")).expanduser()
    else:
        assert_no_cs_train_pt(path)
        loc = resolve_path(path)
        if not loc.is_file():
            raise FileNotFoundError(f"mmap manifest not found: {loc}")
        payload = json.loads(loc.read_text(encoding="utf-8"))
        cache_dir = loc.parent
    if not isinstance(payload, dict):
        raise MmapError("mmap manifest must be a JSON object")
    schema = str(payload.get("schema") or SCHEMA)
    if schema != SCHEMA:
        raise MmapError(f"unsupported mmap schema {schema!r} (want {SCHEMA})")
    symbols = payload.get("symbols")
    if not isinstance(symbols, dict) or not symbols:
        raise MmapError("mmap manifest is missing a non-empty symbols map")
    out = dict(payload)
    out["schema"] = schema
    out["_cache_dir"] = str(cache_dir)
    out["_manifest_path"] = (
        str(resolve_path(path)) if not isinstance(path, Mapping) else ""
    )
    return out


def _symbol_entry(manifest: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    key = str(symbol).upper()
    symbols = manifest.get("symbols") or {}
    entry = symbols.get(key) or symbols.get(symbol)
    if not isinstance(entry, dict):
        raise KeyError(f"symbol {key} is not in the mmap manifest")
    return entry


def _open_memmap(
    path: Path,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    mmap_mode: str,
) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(f"mmap file missing: {path}")
    mode = str(mmap_mode or "r")
    if mode not in ("r", "c", "r+", "w+"):
        raise MmapError(f"unsupported mmap_mode={mode!r}")
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def load_symbol_mmap(
    manifest: str | Path | Mapping[str, Any],
    symbol: str,
    *,
    mmap_mode: str = "r",
) -> SymbolMmap:
    """Map one symbol: float32 features ``[T, F]`` plus label memmaps."""
    payload = load_manifest(manifest) if not isinstance(manifest, Mapping) else (
        load_manifest(manifest) if "_cache_dir" not in manifest else dict(manifest)
    )
    if "_cache_dir" not in payload:
        payload = load_manifest(payload)
    entry = _symbol_entry(payload, symbol)
    cache_dir = Path(str(payload["_cache_dir"]))
    T = int(entry.get("T") or 0)
    F = int(entry.get("F") or payload.get("n_features") or 0)
    if T < 1 or F < 1:
        raise MmapError(f"{symbol}: mmap entry needs T>=1 and F>=1 (got T={T} F={F})")
    feat_name = str(entry.get("features") or f"{str(symbol).upper()}.{FEATURES_SUFFIX}")
    features = _open_memmap(
        cache_dir / feat_name,
        dtype=np.dtype(np.float32),
        shape=(T, F),
        mmap_mode=mmap_mode,
    )
    from forecast.data import FEATURE_NAMES

    names = tuple(payload.get("feature_names") or FEATURE_NAMES)
    labels: dict[str, np.ndarray] = {}
    for key, suffix, dtype, required in LABEL_SPECS:
        rel = entry.get(key)
        if not rel:
            if required:
                raise MmapError(f"{symbol}: mmap entry missing required label {key}")
            continue
        shape: tuple[int, ...] = (T,)
        labels[key] = _open_memmap(
            cache_dir / str(rel),
            dtype=dtype,
            shape=shape,
            mmap_mode=mmap_mode,
        )
    return SymbolMmap(
        symbol=str(symbol).upper(),
        features=features,
        labels=labels,
        feature_names=names,
        T=T,
        F=F,
        mmap_mode=str(mmap_mode or "r"),
        cache_dir=cache_dir,
    )


def symbol_arrays_from_mmap(
    packed: SymbolMmap,
    *,
    valid: np.ndarray | None = None,
) -> "SymbolArrays":
    """Wrap mmap arrays in ``SymbolArrays`` without copying features."""
    from forecast.data import SymbolArrays
    raw_valid = packed.valid if valid is None else np.asarray(valid, dtype=bool)
    overnight = packed.labels.get("overnight_r")
    split_days = packed.labels.get("next_split_days")
    return SymbolArrays(
        symbol=packed.symbol,
        features=packed.features,
        target=np.asarray(packed.target, dtype=np.float32),
        scale=np.asarray(packed.scale, dtype=np.float32),
        valid=np.asarray(raw_valid, dtype=bool),
        dates=packed.dates,
        overnight_r=(
            None
            if overnight is None
            else np.asarray(overnight, dtype=np.float32)
        ),
        next_split_days=(
            None
            if split_days is None
            else np.asarray(split_days, dtype=np.int16)
        ),
    )


def _write_array(path: Path, values: np.ndarray, dtype: np.dtype) -> None:
    arr = np.ascontiguousarray(values, dtype=dtype)
    mm = np.memmap(path, dtype=dtype, mode="w+", shape=arr.shape)
    mm[...] = arr
    mm.flush()
    del mm


def write_symbol_mmap(
    cache_dir: str | Path,
    arrays: "SymbolArrays",
    *,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write one symbol's float32 panel + labels. Returns the manifest entry."""
    del feature_names  # names live on the manifest, not the per-symbol file
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    symbol = str(arrays.symbol).upper()
    feat = np.ascontiguousarray(arrays.features, dtype=np.float32)
    if feat.ndim != 2:
        raise MmapError(f"{symbol}: features must be [T, F], got {feat.shape}")
    T, F = int(feat.shape[0]), int(feat.shape[1])
    feat_name = f"{symbol}.{FEATURES_SUFFIX}"
    _write_array(root / feat_name, feat, np.dtype(np.float32))
    entry: dict[str, Any] = {
        "T": T,
        "F": F,
        "features": feat_name,
    }
    target = np.asarray(arrays.target, dtype=np.float32)
    scale = np.asarray(arrays.scale, dtype=np.float32)
    valid = np.asarray(arrays.valid, dtype=np.uint8)
    if arrays.dates is None:
        dates = np.zeros(T, dtype=np.int64)
    else:
        dates = np.asarray(arrays.dates, dtype=np.int64)
    overnight = arrays.overnight_r
    split_days = arrays.next_split_days
    packed = {
        "target": target,
        "scale": scale,
        "valid": valid,
        "dates": dates,
    }
    if overnight is not None:
        packed["overnight_r"] = np.asarray(overnight, dtype=np.float32)
    if split_days is not None:
        packed["next_split_days"] = np.asarray(split_days, dtype=np.int16)
    for key, suffix, dtype, _required in LABEL_SPECS:
        if key not in packed:
            continue
        if int(np.asarray(packed[key]).shape[0]) != T:
            raise MmapError(f"{symbol}: {key} length != T={T}")
        rel = f"{symbol}.{suffix}"
        _write_array(root / rel, packed[key], dtype)
        entry[key] = rel
    return entry


def write_mmap_cache(
    symbols: Sequence["SymbolArrays"],
    cache_dir: str | Path,
    *,
    feature_names: Sequence[str] | None = None,
    label_return: str = "overnight",
    universe: str = "",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write every symbol and ``mmap_manifest.json``. Returns the manifest path."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    from forecast.data import FEATURE_NAMES

    names = tuple(feature_names or FEATURE_NAMES)
    entries: dict[str, Any] = {}
    n_feat = 0
    for sym in symbols:
        entry = write_symbol_mmap(root, sym, feature_names=names)
        entries[str(sym.symbol).upper()] = entry
        n_feat = int(entry["F"])
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "cache_dir": str(root),
        "dtype": "float32",
        "mmap_mode": "r",
        "n_features": n_feat or len(names),
        "feature_names": list(names),
        "label_return": str(label_return or "overnight"),
        "universe": str(universe or ""),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "do_not_load": ["cs_train_*.pt"],
        "num_workers": 0,
        "symbols": entries,
    }
    if extra:
        for key, value in extra.items():
            if key not in payload:
                payload[key] = value
    dest = root / MANIFEST_NAME
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def list_mmap_symbols(manifest: Mapping[str, Any]) -> list[str]:
    return sorted(str(k).upper() for k in (manifest.get("symbols") or {}))


def iter_symbol_mmap(
    manifest: str | Path | Mapping[str, Any],
    symbols: Iterable[str] | None = None,
    *,
    mmap_mode: str = "r",
) -> list[SymbolMmap]:
    payload = load_manifest(manifest)
    keys = [str(s).upper() for s in symbols] if symbols is not None else list_mmap_symbols(payload)
    return [load_symbol_mmap(payload, key, mmap_mode=mmap_mode) for key in keys]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Write or inspect the Data Manager mmap panel cache. "
            "Train reads mmap_manifest.json; never cs_train_*.pt."
        )
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cache-dir", default="", help="default: <data-dir>/_panel_cache")
    p.add_argument("--universe", default="liquid")
    p.add_argument("--label-return", default="overnight")
    p.add_argument("--write", action="store_true", help="build panels from parquet, write mmap")
    p.add_argument("--inspect", action="store_true", help="print manifest symbol counts")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cache = (
        resolve_path(args.cache_dir)
        if str(args.cache_dir or "").strip()
        else resolve_path(Path(args.data_dir) / "_panel_cache")
    )
    if args.write:
        from forecast.config import DataConfig, interval_data_kwargs
        from forecast.data import build_datasets

        preset = interval_data_kwargs("daily")
        cfg = DataConfig(
            data_dir=args.data_dir,
            interval="daily",
            horizon=1,
            seq_len=int(preset["seq_len"]),
            stride=1,
            min_context=int(preset["min_context"]),
            warmup_bars=int(preset["warmup_bars"]),
            vol_halflife=preset["vol_halflife"],
            z_window=preset["z_window"],
            z_min_periods=preset["z_min_periods"],
            eval_last_bar=True,
            global_calendar_split=True,
            universe=args.universe,
            label_return=args.label_return,
            write_mmap=True,
            use_mmap=False,
            mmap_manifest="",
        )
        # Avoid the chicken-egg: write from parquet, do not read an old mmap.
        bundle = build_datasets(cfg, log_fn=print)
        print(
            f"wrote mmap cache -> {cache / MANIFEST_NAME} "
            f"symbols={bundle.get('n_trading_names')} "
            f"mmap={bundle.get('mmap_manifest')}"
        )
        return
    if args.inspect:
        path = cache / MANIFEST_NAME
        payload = load_manifest(path)
        print(
            f"schema={payload.get('schema')} symbols={len(payload.get('symbols') or {})} "
            f"F={payload.get('n_features')} label_return={payload.get('label_return')}"
        )
        for name in list_mmap_symbols(payload)[:12]:
            packed = load_symbol_mmap(payload, name, mmap_mode="r")
            print(
                f"  {name}: features={tuple(packed.features.shape)} "
                f"dtype={packed.features.dtype} memmap={isinstance(packed.features, np.memmap)}"
            )
        return
    raise SystemExit("pass --write or --inspect")


if __name__ == "__main__":
    main()
