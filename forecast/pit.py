"""Point-in-time side-tape: mask overnight labels on split nights.

Data Manager contract (2a):

- Drop overnight labels where ``next_split_days == 1`` on
  ``data/_pit/factors/{SYM}_daily_factors.parquet``
- Optional membership as-of: ``data/_pit/liquid_membership.json``

``next_split_days == 1`` means the *next* session is a split day. The
close_t → open_{t+1} gap is then a corporate-action artifact, not a
tradeable overnight return. The P(up) head / CS train path nulls those
labels. Membership as-of drops names that were not in the liquid book
on that date (PIT; missing file is a no-op).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from mamba_lm.paths import resolve_path


def _date_keys_from_series(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    ts = ts.dt.normalize()
    return ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")).astype(np.int64).to_numpy()


PIT_DIRNAME = "_pit"
FACTORS_SUBDIR = "factors"
FACTORS_SUFFIX = "_daily_factors.parquet"
MEMBERSHIP_NAME = "liquid_membership.json"
SPLIT_COL = "next_split_days"
PIT_SPLIT_DAYS = 1


def default_pit_dir(data_dir: str | Path = "data") -> Path:
    return resolve_path(Path(data_dir) / PIT_DIRNAME)


def default_factors_dir(data_dir: str | Path = "data") -> Path:
    return default_pit_dir(data_dir) / FACTORS_SUBDIR


def default_membership_path(data_dir: str | Path = "data") -> Path:
    return default_pit_dir(data_dir) / MEMBERSHIP_NAME


def resolve_pit_root(data_dir: str | Path = "data", pit_dir: str | Path | None = None) -> Path:
    if pit_dir is not None and str(pit_dir).strip():
        return resolve_path(pit_dir)
    return default_pit_dir(data_dir)


def resolve_factors_dir(data_dir: str | Path = "data", pit_dir: str | Path | None = None) -> Path:
    """``data/_pit/factors`` unless ``pit_dir`` already points at ``factors/``."""
    root = resolve_pit_root(data_dir, pit_dir)
    if root.name == FACTORS_SUBDIR:
        return root
    return root / FACTORS_SUBDIR


def factor_parquet_path(
    symbol: str,
    *,
    data_dir: str | Path = "data",
    pit_dir: str | Path | None = None,
) -> Path:
    return resolve_factors_dir(data_dir, pit_dir) / f"{str(symbol).upper()}{FACTORS_SUFFIX}"


def _as_date_keys(values: Any) -> np.ndarray:
    if values is None:
        return np.zeros((0,), dtype=np.int64)
    if isinstance(values, pd.Series):
        return _date_keys_from_series(values)
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64, copy=False)
    return _date_keys_from_series(pd.Series(pd.to_datetime(arr)))


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported PIT tape {path}")


def _normalize_tape(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "date", SPLIT_COL])
    out = df.copy()
    cols = {str(c).strip().lower(): c for c in out.columns}
    date_col = cols.get("datetime") or cols.get("session") or cols.get("date")
    split_col = cols.get(SPLIT_COL) or cols.get("split_days") or cols.get("days_to_split")
    if date_col is None or split_col is None:
        raise ValueError(
            "PIT factors tape needs datetime/session/date and next_split_days"
        )
    symbol_col = cols.get("symbol") or cols.get("ticker")
    dates = _as_date_keys(out[date_col])
    splits = pd.to_numeric(out[split_col], errors="coerce").to_numpy(dtype=np.float64)
    payload: dict[str, Any] = {
        "date": dates,
        SPLIT_COL: splits,
    }
    if symbol_col is not None:
        payload["symbol"] = out[symbol_col].astype(str).str.upper()
    return pd.DataFrame(payload)


def _ingest_frame(
    frame: pd.DataFrame,
    by_symbol: dict[str, dict[int, float]],
    *,
    default_symbol: str | None,
    wanted: set[str] | None,
) -> None:
    if frame.empty:
        return
    if "symbol" in frame.columns:
        groups = frame.groupby(frame["symbol"].astype(str).str.upper(), sort=False)
    elif default_symbol:
        groups = [(str(default_symbol).upper(), frame)]
    else:
        return
    for sym, part in groups:
        key = str(sym).upper()
        if wanted is not None and key not in wanted:
            continue
        bucket = by_symbol.setdefault(key, {})
        dates = part["date"].to_numpy(dtype=np.int64)
        splits = part[SPLIT_COL].to_numpy(dtype=np.float64)
        for day, val in zip(dates, splits):
            if np.isfinite(val):
                bucket[int(day)] = float(val)


def _symbol_from_factors_name(path: Path) -> str | None:
    stem = path.stem
    suffix = FACTORS_SUFFIX[: -len(path.suffix)] if FACTORS_SUFFIX.endswith(path.suffix) else "_daily_factors"
    if stem.endswith(suffix):
        return stem[: -len(suffix)].upper()
    if stem.upper().endswith("_DAILY_FACTORS"):
        return stem[: -len("_daily_factors")].upper()
    return None


def load_pit_side_tape(
    data_dir: str | Path = "data",
    *,
    pit_dir: str | Path | None = None,
    symbols: Iterable[str] | None = None,
) -> dict[str, dict[int, float]]:
    """Return ``{SYMBOL: {date_key: next_split_days}}``. Missing tape → {}.

    Primary: ``<pit>/factors/{SYM}_daily_factors.parquet`` (Data Manager).
    """
    wanted = {str(s).upper() for s in symbols} if symbols is not None else None
    by_symbol: dict[str, dict[int, float]] = {}
    factors_dir = resolve_factors_dir(data_dir, pit_dir)
    if factors_dir.is_dir():
        paths: list[Path] = []
        if wanted is not None:
            paths = [factor_parquet_path(s, data_dir=data_dir, pit_dir=pit_dir) for s in wanted]
        else:
            paths = sorted(factors_dir.glob(f"*{FACTORS_SUFFIX}"))
            paths.extend(p for p in sorted(factors_dir.glob("*_daily_factors.parquet")) if p not in paths)
        for path in paths:
            if not path.is_file():
                continue
            try:
                table = _normalize_tape(_read_table(path))
            except Exception:
                continue
            _ingest_frame(
                table,
                by_symbol,
                default_symbol=_symbol_from_factors_name(path),
                wanted=wanted,
            )
    return by_symbol


def next_split_days_aligned(
    dates: np.ndarray | None,
    tape: Mapping[int, float] | None,
    *,
    length: int,
) -> np.ndarray:
    """Align a per-date tape onto a ``[T]`` date key array. Missing → -1."""
    out = np.full(int(length), -1, dtype=np.int16)
    if dates is None or not tape:
        return out
    keys = np.asarray(dates, dtype=np.int64)
    for i, key in enumerate(keys):
        val = tape.get(int(key))
        if val is None or not np.isfinite(val):
            continue
        out[i] = np.int16(int(round(float(val))))
    return out


def pit_overnight_mask(
    next_split_days: np.ndarray | None,
    *,
    split_days: int = PIT_SPLIT_DAYS,
) -> np.ndarray | None:
    """True where the overnight label must be dropped (``next_split_days==1``)."""
    if next_split_days is None:
        return None
    days = np.asarray(next_split_days)
    return days == int(split_days)


def mask_overnight_pit(
    valid: np.ndarray,
    next_split_days: np.ndarray | None,
    *,
    split_days: int = PIT_SPLIT_DAYS,
) -> np.ndarray:
    """Copy ``valid`` and clear bars whose next session is a split day."""
    out = np.asarray(valid, dtype=bool).copy()
    drop = pit_overnight_mask(next_split_days, split_days=split_days)
    if drop is not None and drop.shape == out.shape:
        out = out & ~drop
    return out


def _day_from_any(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return int((ts.normalize() - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D"))


def load_liquid_membership(
    data_dir: str | Path = "data",
    *,
    pit_dir: str | Path | None = None,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Optional PIT liquid membership. Missing file → None (no-op)."""
    if path is not None and str(path).strip():
        loc = resolve_path(path)
    else:
        root = resolve_pit_root(data_dir, pit_dir)
        loc = root / MEMBERSHIP_NAME if root.name != FACTORS_SUBDIR else root.parent / MEMBERSHIP_NAME
        if not loc.is_file():
            loc = default_membership_path(data_dir)
    if not loc.is_file():
        return None
    payload = json.loads(loc.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"liquid membership must be a JSON object: {loc}")
    out = dict(payload)
    out["_path"] = str(loc)
    return out


def _membership_ranges(payload: Mapping[str, Any]) -> dict[str, list[tuple[int, int | None]]]:
    raw = payload.get("membership") or payload.get("members") or payload.get("ranges")
    if not isinstance(raw, dict):
        return {}
    ranges: dict[str, list[tuple[int, int | None]]] = {}
    for symbol, spec in raw.items():
        key = str(symbol).upper()
        items = spec if isinstance(spec, list) else [spec]
        acc: list[tuple[int, int | None]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            start = _day_from_any(
                item.get("start") or item.get("in_from") or item.get("from")
            )
            end = _day_from_any(item.get("end") or item.get("in_to") or item.get("to"))
            if start is None:
                start = -10**9
            acc.append((int(start), None if end is None else int(end)))
        if acc:
            ranges[key] = acc
    return ranges


def _membership_by_date(payload: Mapping[str, Any]) -> list[tuple[int, frozenset[str]]]:
    raw = payload.get("by_date") or payload.get("as_of_dates") or payload.get("dates")
    if not isinstance(raw, dict):
        return []
    rows: list[tuple[int, frozenset[str]]] = []
    for key, names in raw.items():
        day = _day_from_any(key)
        if day is None or not isinstance(names, (list, tuple)):
            continue
        rows.append((int(day), frozenset(str(n).upper() for n in names)))
    rows.sort(key=lambda kv: kv[0])
    return rows


def _membership_symbols(payload: Mapping[str, Any]) -> frozenset[str] | None:
    raw = payload.get("symbols") or payload.get("universe_symbols")
    if not isinstance(raw, (list, tuple)):
        return None
    return frozenset(str(n).upper() for n in raw)


def is_member_asof(
    symbol: str,
    date_key: int,
    payload: Mapping[str, Any] | None,
) -> bool | None:
    """PIT membership as-of ``date_key``. None = unknown (do not mask)."""
    if payload is None:
        return None
    key = str(symbol).upper()
    by_date = _membership_by_date(payload)
    if by_date:
        chosen: frozenset[str] | None = None
        for day, names in by_date:
            if day <= int(date_key):
                chosen = names
            else:
                break
        if chosen is None:
            return False
        return key in chosen
    ranges = _membership_ranges(payload)
    if ranges:
        windows = ranges.get(key)
        if not windows:
            return False
        for start, end in windows:
            if int(date_key) < start:
                continue
            if end is not None and int(date_key) > end:
                continue
            return True
        return False
    static = _membership_symbols(payload)
    if static is not None:
        return key in static
    return None


def mask_membership_asof(
    valid: np.ndarray,
    dates: np.ndarray | None,
    symbol: str,
    payload: Mapping[str, Any] | None,
) -> np.ndarray:
    """Clear bars where the name was not in the liquid book as-of that date."""
    out = np.asarray(valid, dtype=bool).copy()
    if payload is None or dates is None:
        return out
    keys = np.asarray(dates, dtype=np.int64)
    if keys.shape != out.shape:
        return out
    for i, day in enumerate(keys):
        member = is_member_asof(symbol, int(day), payload)
        if member is False:
            out[i] = False
    return out


def overnight_up_labels(
    overnight_r: np.ndarray | None,
    *,
    valid: np.ndarray | None = None,
    next_split_days: np.ndarray | None = None,
    split_days: int = PIT_SPLIT_DAYS,
) -> np.ndarray:
    """``1`` if raw overnight gap > 0, else ``0``. PIT / invalid → NaN."""
    if overnight_r is None:
        n = 0 if valid is None else int(np.asarray(valid).shape[0])
        return np.full(n, np.nan, dtype=np.float64)
    r = np.asarray(overnight_r, dtype=np.float64)
    y = np.where(np.isfinite(r), (r > 0.0).astype(np.float64), np.nan)
    drop = pit_overnight_mask(next_split_days, split_days=split_days)
    if drop is not None and drop.shape == y.shape:
        y = np.where(drop, np.nan, y)
    if valid is not None:
        keep = np.asarray(valid, dtype=bool)
        if keep.shape == y.shape:
            y = np.where(keep, y, np.nan)
    return y
