"""Point-in-time side-tape: mask overnight labels on split nights.

``next_split_days == 1`` means the *next* session is a split day. The
close_t → open_{t+1} gap is then a corporate-action artifact, not a
tradeable overnight return. (2a) drops / nulls those overnight labels.

Desktop tape lives under ``data/_pit/``:

- ``data/_pit/{SYMBOL}_pit.parquet`` (or ``.csv``) with ``datetime`` / ``session``
  and ``next_split_days``
- or one combined ``data/_pit/next_split_days.parquet`` with a ``symbol`` column
"""

from __future__ import annotations

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
SPLIT_COL = "next_split_days"
PIT_SPLIT_DAYS = 1
COMBINED_STEMS = ("next_split_days", "pit", "splits")


def default_pit_dir(data_dir: str | Path = "data") -> Path:
    return resolve_path(Path(data_dir) / PIT_DIRNAME)


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
            "PIT tape needs datetime/session/date and next_split_days columns"
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


def load_pit_side_tape(
    data_dir: str | Path = "data",
    *,
    pit_dir: str | Path | None = None,
    symbols: Iterable[str] | None = None,
) -> dict[str, dict[int, float]]:
    """Return ``{SYMBOL: {date_key: next_split_days}}``. Missing tape → {}."""
    root = resolve_path(pit_dir) if pit_dir else default_pit_dir(data_dir)
    if not root.is_dir():
        return {}
    wanted = {str(s).upper() for s in symbols} if symbols is not None else None
    by_symbol: dict[str, dict[int, float]] = {}

    def _ingest(frame: pd.DataFrame, default_symbol: str | None = None) -> None:
        if frame.empty:
            return
        if "symbol" in frame.columns:
            groups = frame.groupby(frame["symbol"].astype(str).str.upper(), sort=False)
        elif default_symbol:
            groups = [(str(default_symbol).upper(), frame)]
        else:
            return
        for sym, part in groups:
            if wanted is not None and str(sym).upper() not in wanted:
                continue
            bucket = by_symbol.setdefault(str(sym).upper(), {})
            dates = part["date"].to_numpy(dtype=np.int64)
            splits = part[SPLIT_COL].to_numpy(dtype=np.float64)
            for key, val in zip(dates, splits):
                if np.isfinite(val):
                    bucket[int(key)] = float(val)

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".parquet", ".pq", ".csv", ".txt"}:
            continue
        try:
            raw = _read_table(path)
            table = _normalize_tape(raw)
        except Exception:
            continue
        stem = path.stem.upper()
        default = None
        if stem.endswith("_PIT"):
            default = stem[: -len("_PIT")]
        elif stem not in {s.upper() for s in COMBINED_STEMS}:
            default = stem
        _ingest(table, default_symbol=default)
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
