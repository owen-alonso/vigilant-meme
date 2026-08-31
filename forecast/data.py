"""Turn raw 1-minute OHLCV bars into sequences with a next-hour return target.

Pipeline per symbol:

1.  Snap bars onto a regular 390-bar intraday grid (09:30..15:59). Vendor files
    are sparse for illiquid names, so "60 bars ahead" is only equal to "one hour
    ahead" once the grid is regular.
2.  Forward-fill price into untraded slots and mark them, so the model can tell
    a real print from a stale quote.
3.  Build scale-free, strictly causal features. Nothing here uses information
    from bar ``t + 1`` onwards, so the same code runs at inference time.
4.  Attach the target: the 60-bar-ahead log return, divided by a volatility
    estimate known at ``t``.

The target is deliberately volatility-normalized. Raw one-hour returns swing
between calm and stressed regimes by more than an order of magnitude, and a
plain MSE on them just fits the loudest days. ``generate.py`` multiplies the
prediction back by that same scale to report an expected return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from forecast.config import BARS_PER_SESSION, SESSION_START_MINUTE, DataConfig


FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "ret_390",
    "range_hl",
    "body_co",
    "vol_level",
    "vol_change",
    "volume_z",
    "turnover_z",
    "traded",
    "staleness",
    "new_session",
    "tod_sin",
    "tod_cos",
    "tod_frac",
    "dow_frac",
)

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


# --------------------------------------------------------------------------
# Loading and gridding
# --------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Resolve a relative path against the CWD, then against the repo root.

    Defaults like ``data_dir="data"`` should mean the repo's ``data/`` whether
    the entry point was launched from the repo root, from ``forecast/``, or
    from an IDE with its own working directory.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    from_root = REPO_ROOT / candidate
    return from_root if from_root.exists() else candidate


def discover_symbol_files(data_dir: str | Path) -> list[Path]:
    resolved = resolve_path(data_dir)
    paths = sorted(resolved.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no .parquet files found in {resolved.resolve()} "
            f"(looked for data_dir={str(data_dir)!r} relative to "
            f"{Path.cwd()} and {REPO_ROOT})"
        )
    return paths


def symbol_from_path(path: str | Path) -> str:
    """``data/SPAB_clean_1min.parquet`` -> ``SPAB``."""
    return Path(path).stem.split("_")[0].upper()


def _session_naive_datetime(series: pd.Series) -> pd.Series:
    """US-equity session hours are 09:30–15:59 Eastern.

    Tz-aware stamps are converted to America/New_York then stripped. Naive
    timestamps are left as-is (vendor files in this repo are already session-local).
    """
    ts = pd.to_datetime(series)
    tz = getattr(ts.dtype, "tz", None)
    if tz is not None:
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return ts


def normalize_bars(df: pd.DataFrame, *, origin: str = "bars") -> pd.DataFrame:
    """Normalize an OHLCV frame to the columns ``load_bars`` / generate expect.

    Accepts parquet-style columns or a DatetimeIndex (Yahoo). Extra columns
    such as Adj Close are dropped.
    """
    out = df.copy()
    if "datetime" not in {str(c).lower() for c in out.columns}:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
    renamed: dict[Any, str] = {}
    for col in out.columns:
        key = str(col).lower().replace(" ", "")
        if key in {"date", "datetime", "timestamp", "index"}:
            renamed[col] = "datetime"
        else:
            renamed[col] = str(col).lower()
    out = out.rename(columns=renamed)
    if "datetime" not in out.columns:
        raise ValueError(f"{origin}: expected a 'datetime' column, got {list(out.columns)}")
    missing = [c.lower() for c in OHLCV_COLUMNS if c.lower() not in out.columns]
    if missing:
        raise ValueError(f"{origin}: missing columns {missing}")
    out["datetime"] = _session_naive_datetime(out["datetime"])
    keep = ["datetime", "open", "high", "low", "close", "volume"]
    if "source" in out.columns:
        keep.append("source")
    out = out.loc[:, keep]
    out = out.dropna(subset=["datetime", "close"])
    nonpos = out["close"] <= 0
    if nonpos.any():
        raise ValueError(
            f"{origin}: {int(nonpos.sum())} rows with close <= 0 (log-price is undefined)"
        )
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return out.reset_index(drop=True)


def load_bars(path: str | Path) -> pd.DataFrame:
    """Read one parquet of 1-minute bars and normalize its column names."""
    return normalize_bars(pd.read_parquet(path), origin=str(path))


def build_session_grid(
    df: pd.DataFrame,
    cfg: DataConfig,
    *,
    keep_latest_session: bool = False,
) -> pd.DataFrame:
    """Reindex sparse bars onto a dense ``(session, minute_of_session)`` grid.

    Untraded slots get ``traded=0``, ``volume=0`` and a forward-filled price.
    Sessions with too few real prints are dropped entirely, unless
    ``keep_latest_session`` is set (live inference at the open).
    """
    dt = df["datetime"]
    mos = dt.dt.hour * 60 + dt.dt.minute - SESSION_START_MINUTE
    in_session = (mos >= 0) & (mos < BARS_PER_SESSION)
    df = df.loc[in_session].copy()
    if df.empty:
        raise ValueError("no bars fall inside the 09:30-15:59 session grid")
    df["session"] = dt.loc[in_session].dt.normalize()
    df["mos"] = mos.loc[in_session]

    counts = df.groupby("session")["close"].size()
    keep = counts.index[counts >= cfg.min_session_bars]
    if keep_latest_session and len(counts):
        keep = keep.union(pd.Index([counts.index.max()]))
    if len(keep) == 0:
        raise ValueError(
            f"no session has >= {cfg.min_session_bars} bars "
            f"(busiest has {int(counts.max())})"
        )
    df = df[df["session"].isin(keep)]

    full = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(sorted(keep)), np.arange(BARS_PER_SESSION)],
        names=["session", "mos"],
    )
    grid = df.set_index(["session", "mos"]).reindex(full)

    grid["traded"] = grid["close"].notna().astype(np.float64)
    # Intraday holes ffill within the session only. A dropped week must not
    # become a one-minute return at the next open.
    grid["close"] = grid.groupby(level="session")["close"].ffill()
    for col in ("open", "high", "low"):
        filled = grid.groupby(level="session")[col].ffill()
        grid[col] = filled.fillna(grid["close"])
    grid["volume"] = grid["volume"].fillna(0.0).astype(np.float64)

    grid = grid.reset_index()
    grid["datetime"] = grid["session"] + pd.to_timedelta(
        grid["mos"] + SESSION_START_MINUTE, unit="m"
    )
    # A leading session can still start with no price to carry forward.
    grid = grid.dropna(subset=["close"]).reset_index(drop=True)
    cols = [
        "datetime",
        "session",
        "mos",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "traded",
    ]
    if "source" in grid.columns:
        cols.append("source")
    return grid.loc[:, cols]


# --------------------------------------------------------------------------
# Features and target
# --------------------------------------------------------------------------


def _causal_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Standardize against a trailing window that excludes the current bar."""
    roll = s.rolling(window, min_periods=min_periods)
    mean = roll.mean().shift(1)
    std = roll.std().shift(1)
    return (s - mean) / std.clip(lower=1e-8)


def compute_features(grid: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Attach features, the next-hour target, its scale, and a validity mask."""
    out = grid.copy()
    close = out["close"].astype(np.float64)
    log_close = np.log(close)

    session_dates = pd.to_datetime(out["session"])
    gap_days = session_dates.diff().dt.days.fillna(0)
    large_join = gap_days > cfg.max_session_gap_days
    join_id = large_join.cumsum()

    r1 = log_close.diff().mask(large_join, np.nan)
    # EWM realized vol, shifted so bar t's own return is excluded.
    ewm_var = r1.pow(2).ewm(halflife=cfg.vol_halflife, min_periods=cfg.vol_halflife).mean()
    sigma = np.sqrt(ewm_var).shift(1).clip(lower=cfg.vol_floor)
    out["sigma"] = sigma
    # Scale of a horizon-length return under a random walk.
    out["scale"] = sigma * math.sqrt(cfg.horizon)

    for k in (1, 5, 15, 60, 390):
        crossed = join_id != join_id.shift(k)
        raw = log_close.diff(k).mask(crossed, np.nan)
        out[f"ret_{k}"] = raw / (sigma * math.sqrt(k))

    out["range_hl"] = ((out["high"] - out["low"]) / close) / sigma
    out["body_co"] = ((close - out["open"]) / close) / sigma

    log_sigma = np.log(sigma)
    out["vol_level"] = _causal_zscore(log_sigma, cfg.z_window, cfg.z_min_periods)
    out["vol_change"] = log_sigma.diff(BARS_PER_SESSION)

    log_volume = np.log1p(out["volume"])
    out["volume_z"] = _causal_zscore(log_volume, cfg.z_window, cfg.z_min_periods)
    out["turnover_z"] = _causal_zscore(
        np.log1p(out["volume"] * close), cfg.z_window, cfg.z_min_periods
    )

    # Bars elapsed since the last real print, so stale prices are discountable.
    position = np.arange(len(out), dtype=np.float64)
    last_traded = pd.Series(
        np.where(out["traded"].to_numpy() > 0, position, np.nan), index=out.index
    ).ffill()
    out["staleness"] = np.log1p(position - last_traded) / math.log(BARS_PER_SESSION)

    out["new_session"] = (out["mos"] == 0).astype(np.float64)
    tod = out["mos"].astype(np.float64) / (BARS_PER_SESSION - 1)
    out["tod_frac"] = tod
    out["tod_sin"] = np.sin(2.0 * math.pi * tod)
    out["tod_cos"] = np.cos(2.0 * math.pi * tod)
    out["dow_frac"] = out["datetime"].dt.dayofweek.astype(np.float64) / 4.0

    # Target: log return realized `horizon` bars later, in volatility units.
    forward = log_close.shift(-cfg.horizon) - log_close
    out["target_raw"] = forward
    out["target"] = forward / out["scale"]
    horizon_traded = out["traded"].shift(-cfg.horizon)
    out["horizon_traded"] = horizon_traded.fillna(0.0)

    # A bar is trainable only if it is a real print, the horizon lands inside
    # the same session, and every feature has enough history behind it.
    same_session = (out["mos"] + cfg.horizon) < BARS_PER_SESSION
    finite = out[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    features_ok = np.isfinite(finite).all(axis=1)
    valid = (
        (out["traded"] > 0)
        & same_session
        & out["target"].notna()
        & np.isfinite(out["target"].to_numpy())
        & features_ok
        & (position >= cfg.warmup_bars)
    )
    if cfg.require_horizon_traded:
        valid = valid & (horizon_traded > 0)
    out["valid"] = valid

    for name in FEATURE_NAMES:
        out[name] = out[name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[name] = out[name].clip(-cfg.clip, cfg.clip)
    out["target"] = out["target"].fillna(0.0)
    out["target_raw"] = out["target_raw"].fillna(0.0)
    return out


def assemble_panel(
    bars: pd.DataFrame,
    cfg: DataConfig,
    symbol: str,
    *,
    keep_latest_session: bool = False,
) -> pd.DataFrame:
    """Session-grid + features for an already-normalized OHLCV frame."""
    grid = build_session_grid(
        bars, cfg, keep_latest_session=keep_latest_session
    )
    panel = compute_features(grid, cfg)
    panel["symbol"] = str(symbol).upper()
    return panel


def build_panel_from_bars(
    df: pd.DataFrame,
    cfg: DataConfig,
    symbol: str,
    *,
    keep_latest_session: bool = False,
) -> pd.DataFrame:
    """OHLCV frame (Yahoo or parquet-like) -> feature panel."""
    return assemble_panel(
        normalize_bars(df, origin=symbol),
        cfg,
        symbol,
        keep_latest_session=keep_latest_session,
    )


def build_panel(
    path: str | Path,
    cfg: DataConfig,
    *,
    keep_latest_session: bool = False,
) -> pd.DataFrame:
    """Full raw-parquet -> feature-frame pipeline for one symbol."""
    return assemble_panel(
        load_bars(path),
        cfg,
        symbol_from_path(path),
        keep_latest_session=keep_latest_session,
    )


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


@dataclass
class SymbolArrays:
    """Contiguous per-symbol arrays for one chronological split."""

    symbol: str
    features: np.ndarray  # [T, F] float32
    target: np.ndarray  # [T]    float32, volatility units
    scale: np.ndarray  # [T]    float32, target -> log-return multiplier
    valid: np.ndarray  # [T]    bool


def panel_to_arrays(panel: pd.DataFrame, symbol: str) -> SymbolArrays:
    return SymbolArrays(
        symbol=symbol,
        features=panel[list(FEATURE_NAMES)].to_numpy(dtype=np.float32),
        target=panel["target"].to_numpy(dtype=np.float32),
        scale=panel["scale"].to_numpy(dtype=np.float32),
        valid=panel["valid"].to_numpy(dtype=bool),
    )


class SequenceDataset(Dataset):
    """Fixed-length windows over one or more symbols.

    ``__getitem__`` returns ``(x, y, mask, scale)`` where ``x`` is ``[L, F]`` and
    the rest are ``[L]``. Supervision is dense: the model predicts at every bar
    and ``mask`` zeroes out the ones that are not trainable.
    """

    def __init__(
        self,
        symbols: Sequence[SymbolArrays],
        seq_len: int,
        stride: int,
        *,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        min_context: int = 0,
        require_valid: int = 1,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if not 0 <= min_context < seq_len:
            raise ValueError(f"min_context must be in [0, seq_len); got {min_context}")
        self.symbols = list(symbols)
        self.seq_len = seq_len
        self.min_context = min_context
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.windows: list[tuple[int, int]] = []
        for i, sym in enumerate(self.symbols):
            length = len(sym.target)
            for start in range(0, max(0, length - seq_len + 1), stride):
                scorable = sym.valid[start + min_context : start + seq_len]
                if int(scorable.sum()) >= require_valid:
                    self.windows.append((i, start))

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def n_valid_bars(self) -> int:
        return int(sum(int(s.valid.sum()) for s in self.symbols))

    def __getitem__(self, index: int):
        sym_idx, start = self.windows[index]
        sym = self.symbols[sym_idx]
        stop = start + self.seq_len
        x = sym.features[start:stop]
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std
        mask = sym.valid[start:stop].copy()
        mask[: self.min_context] = False
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(sym.target[start:stop].copy()),
            torch.from_numpy(mask),
            torch.from_numpy(sym.scale[start:stop].copy()),
        )


def _source_mix(frame: pd.DataFrame) -> dict[str, float]:
    if "source" not in frame.columns:
        return {}
    traded = frame.loc[frame["traded"] > 0, "source"].dropna()
    if traded.empty:
        return {}
    counts = traded.astype(str).value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def _split_bounds(n_sessions: int, cfg: DataConfig) -> tuple[int, int]:
    n_test = int(round(n_sessions * cfg.test_fraction))
    n_val = int(round(n_sessions * cfg.val_fraction))
    n_train = n_sessions - n_val - n_test
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"{n_sessions} sessions cannot be split into train/val/test with "
            f"val_fraction={cfg.val_fraction}, test_fraction={cfg.test_fraction}"
        )
    return n_train, n_train + n_val


def build_datasets(
    cfg: DataConfig,
    *,
    paths: Sequence[str | Path] | None = None,
    log_fn: Any | None = print,
) -> dict[str, Any]:
    """Load every symbol, split chronologically, and standardize on train only.

    Splits are cut at session boundaries. Because a target never reaches past
    the end of its own session, no label can straddle a split boundary.
    """
    paths = list(paths) if paths is not None else discover_symbol_files(cfg.data_dir)

    train_syms: list[SymbolArrays] = []
    val_syms: list[SymbolArrays] = []
    test_syms: list[SymbolArrays] = []
    meta: list[dict[str, Any]] = []

    for path in paths:
        panel = build_panel(path, cfg)
        symbol = str(panel["symbol"].iloc[0])
        sessions = panel["session"].drop_duplicates().sort_values().to_numpy()
        cut_train, cut_val = _split_bounds(len(sessions), cfg)
        train_end, val_end = sessions[cut_train], sessions[cut_val]

        is_train = panel["session"] < train_end
        is_val = (panel["session"] >= train_end) & (panel["session"] < val_end)
        is_test = panel["session"] >= val_end

        train_syms.append(panel_to_arrays(panel[is_train], symbol))
        val_syms.append(panel_to_arrays(panel[is_val], symbol))
        test_syms.append(panel_to_arrays(panel[is_test], symbol))
        train_mix = _source_mix(panel[is_train])
        val_mix = _source_mix(panel[is_val])
        test_mix = _source_mix(panel[is_test])
        meta.append(
            {
                "symbol": symbol,
                "path": str(path),
                "sessions": int(len(sessions)),
                "grid_bars": int(len(panel)),
                "traded_bars": int(panel["traded"].sum()),
                "valid_bars": int(panel["valid"].sum()),
                "first": str(panel["datetime"].iloc[0]),
                "last": str(panel["datetime"].iloc[-1]),
                "train_end": str(pd.Timestamp(train_end).date()),
                "val_end": str(pd.Timestamp(val_end).date()),
                "train_source_mix": train_mix,
                "val_source_mix": val_mix,
                "test_source_mix": test_mix,
            }
        )
        if log_fn:
            m = meta[-1]
            log_fn(
                f"{symbol}: {m['sessions']} sessions, {m['grid_bars']} grid bars, "
                f"{m['traded_bars']} traded ({m['traded_bars'] / max(1, m['grid_bars']):.1%}), "
                f"{m['valid_bars']} labelled | train<{m['train_end']} "
                f"val<{m['val_end']} test>= {m['val_end']}"
            )
            if train_mix and test_mix:
                train_top = max(train_mix, key=train_mix.get)
                test_top = max(test_mix, key=test_mix.get)
                if train_top != test_top:
                    log_fn(
                        f"WARNING {symbol}: train vendor is mostly {train_top} "
                        f"({train_mix[train_top]:.0%}) but test is mostly {test_top} "
                        f"({test_mix[test_top]:.0%}). Test IC is not the same "
                        "data-generating process as train."
                    )

    mean, std = feature_stats(train_syms)

    # Evaluation windows overlap by exactly the warmup region, so the scored
    # segments tile the split end to end and each bar is counted once.
    eval_stride = max(1, cfg.seq_len - cfg.min_context)
    common = {"feature_mean": mean, "feature_std": std, "min_context": cfg.min_context}
    datasets = {
        "train": SequenceDataset(train_syms, cfg.seq_len, cfg.stride, **common),
        "val": SequenceDataset(val_syms, cfg.seq_len, eval_stride, **common),
        "test": SequenceDataset(test_syms, cfg.seq_len, eval_stride, **common),
    }
    if log_fn:
        for name, ds in datasets.items():
            log_fn(f"{name}: {len(ds)} windows, {ds.n_valid_bars} labelled bars")
    return {
        "datasets": datasets,
        "feature_mean": mean,
        "feature_std": std,
        "feature_names": list(FEATURE_NAMES),
        "meta": meta,
    }


def feature_stats(symbols: Sequence[SymbolArrays]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over labelled training bars only."""
    chunks = [s.features[s.valid] for s in symbols if s.valid.any()]
    if not chunks:
        raise ValueError("no labelled training bars; check min_session_bars / warmup_bars")
    stacked = np.concatenate(chunks, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    # Constant features (e.g. a symbol that always trades) must not blow up.
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std
