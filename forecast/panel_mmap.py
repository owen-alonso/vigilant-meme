"""Data Manager mmap panel feed (desktop contract, in-repo).

Manifest: ``data/_panel_cache/mmap_manifest.json``

Two on-disk schemas are accepted:

- ``panel_mmap/v1`` — in-repo writer (``symbols`` is a map of file entries)
- ``numpy_mmap_v1`` — Data Manager store (pointer file → hashed
  ``mmap_<hash>/manifest.json``, ``symbols`` is a list). The adapter
  consumes the existing float32 arrays; it does **not** rewrite the
  121MiB store.

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
DM_SCHEMA = "numpy_mmap_v1"
SUPPORTED_SCHEMAS = frozenset({SCHEMA, DM_SCHEMA, "numpy-mmap-v1", "numpy_mmap/v1"})
MANIFEST_NAME = "mmap_manifest.json"
INNER_MANIFEST_NAMES = ("manifest.json", MANIFEST_NAME)
DEFAULT_CACHE_REL = Path("data") / "_panel_cache"
DEFAULT_MANIFEST_REL = DEFAULT_CACHE_REL / MANIFEST_NAME
CS_TRAIN_PT_RE = re.compile(r"^cs_train_.*\.pt$", re.IGNORECASE)
POINTER_KEYS = (
    "manifest",
    "manifest_path",
    "pointer",
    "path",
    "relpath",
    "target",
    "href",
)
STORE_KEYS = ("store", "store_dir", "root", "mmap_dir", "cache", "cache_dir", "dir")

FEATURES_SUFFIX = "features.f32"
LABEL_SPECS: tuple[tuple[str, str, np.dtype, bool], ...] = (
    ("target", "target.f32", np.dtype(np.float32), True),
    ("scale", "scale.f32", np.dtype(np.float32), True),
    ("valid", "valid.u8", np.dtype(np.uint8), True),
    ("dates", "dates.i64", np.dtype(np.int64), True),
    ("overnight_r", "overnight_r.f32", np.dtype(np.float32), False),
    ("next_split_days", "next_split_days.i16", np.dtype(np.int16), False),
)
_FEATURE_TEMPLATES = (
    "{sym}.features.f32",
    "{sym}.features.npy",
    "{sym}.f32",
    "{sym}_features.f32",
    "{sym}/features.f32",
    "{sym}/features.npy",
    "features/{sym}.f32",
    "features/{sym}.npy",
    "features/{sym}.features.f32",
)
_LABEL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "target": (
        "{sym}.target.f32",
        "{sym}.target.npy",
        "{sym}.y.f32",
        "{sym}/target.f32",
        "target/{sym}.f32",
    ),
    "scale": (
        "{sym}.scale.f32",
        "{sym}.scale.npy",
        "{sym}/scale.f32",
        "scale/{sym}.f32",
    ),
    "valid": (
        "{sym}.valid.u8",
        "{sym}.valid.npy",
        "{sym}.mask.u8",
        "{sym}/valid.u8",
        "valid/{sym}.u8",
    ),
    "dates": (
        "{sym}.dates.i64",
        "{sym}.dates.npy",
        "{sym}.date.i64",
        "{sym}/dates.i64",
        "dates/{sym}.i64",
    ),
    "overnight_r": (
        "{sym}.overnight_r.f32",
        "{sym}.overnight.f32",
        "{sym}.overnight_r.npy",
        "{sym}/overnight_r.f32",
        "overnight_r/{sym}.f32",
    ),
    "next_split_days": (
        "{sym}.next_split_days.i16",
        "{sym}.next_split_days.npy",
        "{sym}/next_split_days.i16",
        "next_split_days/{sym}.i16",
    ),
}


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


def _read_json_payload(loc: Path) -> Any:
    text = loc.read_text(encoding="utf-8").strip()
    if not text:
        raise MmapError(f"empty mmap manifest: {loc}")
    return json.loads(text)


def _resolve_rel(base: Path, rel: str | Path) -> Path:
    raw = Path(rel)
    if raw.is_absolute():
        return raw
    here = (base / raw).resolve()
    if here.exists():
        return here
    parent = (base.parent / raw).resolve()
    if parent.exists():
        return parent
    anchored = resolve_path(raw)
    if anchored.exists():
        return anchored
    return here


def _schema_name(payload: Mapping[str, Any] | None) -> str:
    raw = str((payload or {}).get("schema") or "").strip()
    return raw or SCHEMA


def _is_supported_schema(schema: str) -> bool:
    key = str(schema or "").strip()
    return (not key) or key in SUPPORTED_SCHEMAS or key.replace("-", "_") in SUPPORTED_SCHEMAS


def _pointer_target(payload: Mapping[str, Any]) -> str | None:
    for key in POINTER_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and value.strip() != payload.get("schema"):
            if key == "path" and not (
                value.endswith(".json")
                or "manifest" in value
                or value.startswith("mmap_")
            ):
                continue
            return value.strip()
    return None


def _store_dir(payload: Mapping[str, Any], base: Path) -> Path | None:
    for key in STORE_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        loc = _resolve_rel(base, value.strip())
        if loc.is_dir():
            return loc
        if loc.is_file() and loc.name in INNER_MANIFEST_NAMES:
            return loc.parent
    return None


def _inner_manifest_in(store: Path) -> Path | None:
    for name in INNER_MANIFEST_NAMES:
        cand = store / name
        if cand.is_file():
            return cand
    return None


def _merge_payload(outer: Mapping[str, Any], inner: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(outer)
    out.update(inner)
    if "symbols" not in inner and "symbols" in outer:
        out["symbols"] = outer["symbols"]
    if not out.get("feature_names") and outer.get("feature_names"):
        out["feature_names"] = outer["feature_names"]
    if not out.get("n_features") and outer.get("n_features"):
        out["n_features"] = outer["n_features"]
    return out


def _follow_pointers(
    payload: Any,
    *,
    start_dir: Path,
    manifest_path: Path | None,
    seen: set[str] | None = None,
    depth: int = 0,
) -> tuple[dict[str, Any], Path, Path | None]:
    """Resolve DM pointer files (``mmap_manifest.json`` → ``mmap_<hash>/manifest.json``)."""
    if depth > 6:
        raise MmapError("mmap pointer chain is too deep")
    if isinstance(payload, str):
        rel = payload.strip()
        if not rel:
            raise MmapError("mmap pointer is an empty string")
        loc = _resolve_rel(start_dir, rel)
        if not loc.is_file():
            raise MmapError(f"mmap pointer target missing: {loc}")
        assert_no_cs_train_pt(loc)
        return _follow_pointers(
            _read_json_payload(loc),
            start_dir=loc.parent,
            manifest_path=loc,
            seen=seen,
            depth=depth + 1,
        )
    if not isinstance(payload, Mapping):
        raise MmapError("mmap manifest must be a JSON object")
    data = dict(payload)
    loc_key = str(manifest_path) if manifest_path is not None else ""
    visited = set(seen or ())
    if loc_key:
        if loc_key in visited:
            raise MmapError(f"mmap pointer cycle at {loc_key}")
        visited.add(loc_key)

    target = _pointer_target(data)
    if target:
        loc = _resolve_rel(start_dir, target)
        if loc.is_file() and str(loc) != loc_key:
            assert_no_cs_train_pt(loc)
            inner, cache_dir, inner_path = _follow_pointers(
                _read_json_payload(loc),
                start_dir=loc.parent,
                manifest_path=loc,
                seen=visited,
                depth=depth + 1,
            )
            return _merge_payload(data, inner), cache_dir, inner_path
        if loc.is_dir():
            data = dict(data)
            data.setdefault("store", str(loc))

    store = _store_dir(data, start_dir)
    if store is not None:
        inner_path = _inner_manifest_in(store)
        if inner_path is not None and str(inner_path) != loc_key:
            assert_no_cs_train_pt(inner_path)
            inner, cache_dir, resolved = _follow_pointers(
                _read_json_payload(inner_path),
                start_dir=inner_path.parent,
                manifest_path=inner_path,
                seen=visited,
                depth=depth + 1,
            )
            return _merge_payload(data, inner), cache_dir, resolved
        return data, store, manifest_path

    cache_dir = start_dir
    explicit = data.get("cache_dir")
    if isinstance(explicit, str) and explicit.strip():
        resolved = _resolve_rel(start_dir, explicit.strip())
        if resolved.is_dir():
            cache_dir = resolved
    return data, cache_dir, manifest_path


def _f_hint(payload: Mapping[str, Any]) -> int:
    names = payload.get("feature_names")
    if isinstance(names, (list, tuple)) and names:
        return int(len(names))
    try:
        return int(payload.get("n_features") or payload.get("F") or 0)
    except (TypeError, ValueError):
        return 0


def _shape_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _first_existing(root: Path, rels: Iterable[str]) -> str | None:
    for rel in rels:
        if (root / rel).is_file():
            return rel
        lower = rel.lower()
        if lower != rel and (root / lower).is_file():
            return lower
    return None


def _discover_symbol_files(root: Path, symbol: str) -> dict[str, Any]:
    key = str(symbol).upper()
    variants = (key, key.lower(), str(symbol))
    feat_rels: list[str] = []
    for sym in variants:
        feat_rels.extend(tmpl.format(sym=sym) for tmpl in _FEATURE_TEMPLATES)
    features = _first_existing(root, feat_rels)
    if features is None:
        return {}
    entry: dict[str, Any] = {"features": features}
    for label, templates in _LABEL_TEMPLATES.items():
        rels = [tmpl.format(sym=sym) for sym in variants for tmpl in templates]
        found = _first_existing(root, rels)
        if found:
            entry[label] = found
    return entry


def _index_rows(payload: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Return ``{SYM: (offset_rows, length)}`` for a stacked numpy_mmap store."""
    raw = payload.get("index") or payload.get("offsets") or payload.get("slices")
    rows: dict[str, tuple[int, int]] = {}
    if isinstance(raw, Mapping):
        for key, spec in raw.items():
            if isinstance(spec, Mapping):
                start = spec.get("offset", spec.get("start", spec.get("begin", 0)))
                length = spec.get("length", spec.get("T", spec.get("n", spec.get("rows"))))
                if length is None and spec.get("end") is not None:
                    length = int(spec["end"]) - int(start)
                if length is None:
                    continue
                rows[str(key).upper()] = (int(start), int(length))
            elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
                start, end = int(spec[0]), int(spec[1])
                rows[str(key).upper()] = (start, end - start if end > start else int(spec[1]))
    elif isinstance(raw, list):
        for spec in raw:
            if not isinstance(spec, Mapping):
                continue
            name = spec.get("symbol") or spec.get("ticker") or spec.get("name")
            if not name:
                continue
            start = spec.get("offset", spec.get("start", spec.get("begin", 0)))
            length = spec.get("length", spec.get("T", spec.get("n", spec.get("rows"))))
            if length is None and spec.get("end") is not None:
                length = int(spec["end"]) - int(start)
            if length is None:
                continue
            rows[str(name).upper()] = (int(start), int(length))
    return rows


def _shared_array_rel(payload: Mapping[str, Any], key: str) -> str | None:
    arrays = payload.get("arrays") or payload.get("files") or payload.get("mmap")
    if isinstance(arrays, Mapping):
        spec = arrays.get(key)
        if isinstance(spec, str) and spec.strip():
            return spec.strip()
        if isinstance(spec, Mapping):
            rel = spec.get("path") or spec.get("file") or spec.get("rel")
            if isinstance(rel, str) and rel.strip():
                return rel.strip()
    if isinstance(payload.get(key), str) and str(payload[key]).endswith(
        (".f32", ".u8", ".i64", ".i16", ".npy", ".bin", ".mmap")
    ):
        return str(payload[key])
    return None


def _entry_from_item(
    item: Any,
    *,
    cache_dir: Path,
    payload: Mapping[str, Any],
    default_symbol: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    f_hint = _f_hint(payload)
    index = _index_rows(payload)
    if isinstance(item, str):
        key = item.strip().upper()
        if not key:
            return None
        entry = _discover_symbol_files(cache_dir, key)
        if not entry:
            shared = _shared_array_rel(payload, "features")
            if shared and key in index:
                start, length = index[key]
                entry = {
                    "features": shared,
                    "T": length,
                    "F": f_hint,
                    "offset": start,
                }
                for label, _suffix, _dtype, _req in LABEL_SPECS:
                    rel = _shared_array_rel(payload, label)
                    if rel:
                        entry[label] = rel
            else:
                return None
        if f_hint and "F" not in entry:
            entry["F"] = f_hint
        if key in index:
            entry.setdefault("T", index[key][1])
            entry.setdefault("offset", index[key][0])
        return key, entry
    if not isinstance(item, Mapping):
        return None
    name = item.get("symbol") or item.get("ticker") or item.get("name") or default_symbol
    if not name:
        return None
    key = str(name).upper()
    entry = {str(k): v for k, v in item.items() if k not in {"symbol", "ticker", "name"}}
    shape = _shape_pair(item.get("shape") or item.get("features_shape"))
    if shape:
        entry["T"] = shape[0]
        entry["F"] = shape[1]
    for src, dest in (("n_rows", "T"), ("rows", "T"), ("n_cols", "F"), ("cols", "F")):
        if src in item and dest not in entry:
            try:
                entry[dest] = int(item[src])
            except (TypeError, ValueError):
                pass
    if "features" not in entry:
        discovered = _discover_symbol_files(cache_dir, key)
        entry.update({k: v for k, v in discovered.items() if k not in entry})
    if key in index:
        entry.setdefault("T", index[key][1])
        entry.setdefault("offset", index[key][0])
        shared = _shared_array_rel(payload, "features")
        if shared and "features" not in entry:
            entry["features"] = shared
            for label, _suffix, _dtype, _req in LABEL_SPECS:
                rel = _shared_array_rel(payload, label)
                if rel and label not in entry:
                    entry[label] = rel
    if f_hint and not entry.get("F"):
        entry["F"] = f_hint
    return key, entry


def _normalize_symbols_map(
    payload: Mapping[str, Any],
    cache_dir: Path,
) -> dict[str, dict[str, Any]]:
    raw = payload.get("symbols")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for key, spec in raw.items():
            parsed = _entry_from_item(
                spec if isinstance(spec, Mapping) else {"symbol": key, "features": spec},
                cache_dir=cache_dir,
                payload=payload,
                default_symbol=str(key),
            )
            if parsed is None:
                if isinstance(spec, Mapping):
                    out[str(key).upper()] = dict(spec)
                continue
            name, entry = parsed
            out[name] = entry
        return out
    if isinstance(raw, list):
        for item in raw:
            parsed = _entry_from_item(item, cache_dir=cache_dir, payload=payload)
            if parsed is None:
                continue
            name, entry = parsed
            out[name] = entry
        return out
    return out


def load_manifest(path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load ``mmap_manifest.json``, adapting Data Manager ``numpy_mmap_v1``.

    Pointer files (``data/_panel_cache/mmap_manifest.json`` →
    ``mmap_<hash>/manifest.json``) are followed. A ``symbols`` list is
    rewritten to the in-memory map ``load_symbol_mmap`` expects. The
    hashed float32 store is not rewritten.
    """
    manifest_path: Path | None = None
    if isinstance(path, Mapping):
        raw: Any = dict(path)
        start_dir = Path(str(raw.get("cache_dir") or raw.get("_cache_dir") or ".")).expanduser()
        if raw.get("_manifest_path"):
            manifest_path = Path(str(raw["_manifest_path"]))
            start_dir = manifest_path.parent
    else:
        assert_no_cs_train_pt(path)
        loc = resolve_path(path)
        if not loc.is_file():
            raise FileNotFoundError(f"mmap manifest not found: {loc}")
        raw = _read_json_payload(loc)
        start_dir = loc.parent
        manifest_path = loc
    payload, cache_dir, resolved_path = _follow_pointers(
        raw, start_dir=start_dir, manifest_path=manifest_path
    )
    schema = _schema_name(payload)
    if not _is_supported_schema(schema):
        raise MmapError(
            f"unsupported mmap schema {schema!r} "
            f"(want {SCHEMA} or {DM_SCHEMA})"
        )
    symbols = _normalize_symbols_map(payload, cache_dir)
    if not symbols:
        raise MmapError(
            "mmap manifest has no readable symbols "
            f"(schema={schema!r}, cache_dir={cache_dir})"
        )
    out = dict(payload)
    out["schema"] = schema
    out["symbols"] = symbols
    out["_cache_dir"] = str(cache_dir)
    out["_manifest_path"] = str(resolved_path or manifest_path or "")
    out["_source_schema"] = schema
    out["_adapted"] = schema != SCHEMA or not isinstance(payload.get("symbols"), dict)
    return out


def _symbol_entry(manifest: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    key = str(symbol).upper()
    symbols = manifest.get("symbols") or {}
    entry = symbols.get(key) or symbols.get(symbol)
    if not isinstance(entry, dict):
        raise KeyError(f"symbol {key} is not in the mmap manifest")
    return entry


def _resolve_array_path(cache_dir: Path, rel: str) -> Path:
    raw = Path(rel)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            cache_dir / raw,
            cache_dir.parent / raw,
            resolve_path(raw),
        ]
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return cache_dir / raw


def _is_npy(path: Path) -> bool:
    if path.suffix.lower() == ".npy":
        return True
    try:
        with path.open("rb") as fh:
            return fh.read(6) == b"\x93NUMPY"
    except OSError:
        return False


def _infer_rows(path: Path, *, dtype: np.dtype, width: int = 1) -> int:
    if _is_npy(path):
        arr = np.load(path, mmap_mode="r")
        n = int(arr.shape[0])
        del arr
        return n
    item = max(int(np.dtype(dtype).itemsize), 1)
    width = max(int(width), 1)
    n = int(path.stat().st_size) // (item * width)
    return n


def _open_memmap(
    path: Path,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    mmap_mode: str,
    offset_rows: int = 0,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"mmap file missing: {path}")
    mode = str(mmap_mode or "r")
    if mode not in ("r", "c", "r+", "w+"):
        raise MmapError(f"unsupported mmap_mode={mode!r}")
    want = tuple(int(x) for x in shape)
    if _is_npy(path):
        mapped: np.ndarray = np.load(path, mmap_mode=mode)
    else:
        if len(want) == 2 and want[0] > 0 and want[1] > 0 and offset_rows:
            full_t = _infer_rows(path, dtype=dtype, width=want[1])
            mapped = np.memmap(path, dtype=dtype, mode=mode, shape=(full_t, want[1]))
        elif len(want) == 1 and want[0] > 0 and offset_rows:
            full_t = _infer_rows(path, dtype=dtype, width=1)
            mapped = np.memmap(path, dtype=dtype, mode=mode, shape=(full_t,))
        else:
            mapped = np.memmap(path, dtype=dtype, mode=mode, shape=want)
    if offset_rows or (want and mapped.shape[: len(want)] != want and mapped.shape[0] != want[0]):
        start = int(offset_rows)
        stop = start + int(want[0]) if want else mapped.shape[0]
        mapped = mapped[start:stop]
    return mapped


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
    if not isinstance(payload.get("symbols"), dict):
        payload = load_manifest(payload)
    entry = _symbol_entry(payload, symbol)
    cache_dir = Path(str(payload["_cache_dir"]))
    key = str(symbol).upper()
    if not entry.get("features"):
        discovered = _discover_symbol_files(cache_dir, key)
        entry = {**discovered, **entry}
    feat_rel = str(entry.get("features") or f"{key}.{FEATURES_SUFFIX}")
    feat_path = _resolve_array_path(cache_dir, feat_rel)
    F = int(entry.get("F") or payload.get("n_features") or _f_hint(payload) or 0)
    T = int(entry.get("T") or 0)
    offset = int(entry.get("offset") or entry.get("start") or 0)
    if F < 1 and feat_path.is_file() and _is_npy(feat_path):
        peek = np.load(feat_path, mmap_mode="r")
        if peek.ndim >= 2:
            F = int(peek.shape[-1])
            if T < 1:
                T = int(peek.shape[0])
        del peek
    if F < 1:
        from forecast.data import FEATURE_NAMES

        F = int(len(FEATURE_NAMES))
    if T < 1:
        if not feat_path.is_file():
            raise MmapError(f"{key}: mmap features missing: {feat_path}")
        T = _infer_rows(feat_path, dtype=np.dtype(np.float32), width=F)
        if offset:
            T = max(T - offset, 0)
    if T < 1 or F < 1:
        raise MmapError(f"{symbol}: mmap entry needs T>=1 and F>=1 (got T={T} F={F})")
    features = _open_memmap(
        feat_path,
        dtype=np.dtype(np.float32),
        shape=(T, F),
        mmap_mode=mmap_mode,
        offset_rows=offset,
    )
    from forecast.data import FEATURE_NAMES

    names = tuple(payload.get("feature_names") or FEATURE_NAMES)
    labels: dict[str, np.ndarray] = {}
    for label, suffix, dtype, required in LABEL_SPECS:
        rel = entry.get(label)
        if not rel:
            found = _discover_symbol_files(cache_dir, key).get(label)
            rel = found
        if not rel:
            if required:
                raise MmapError(f"{symbol}: mmap entry missing required label {label}")
            continue
        labels[label] = _open_memmap(
            _resolve_array_path(cache_dir, str(rel)),
            dtype=dtype,
            shape=(T,),
            mmap_mode=mmap_mode,
            offset_rows=offset,
        )
        del suffix
    return SymbolMmap(
        symbol=key,
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
            "Write or inspect the mmap panel cache. Reads Data Manager "
            "numpy_mmap_v1 pointer files and in-repo panel_mmap/v1. "
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
            f"schema={payload.get('schema')} adapted={payload.get('_adapted')} "
            f"symbols={len(payload.get('symbols') or {})} "
            f"F={payload.get('n_features')} label_return={payload.get('label_return')} "
            f"store={payload.get('_cache_dir')}"
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
