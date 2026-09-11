"""Turn raw OHLCV bars into sequences with a forward-return target.

Alpha Vantage daily bars (default): one row per trading day, next-day target.

1-minute bars (premium TIME_SERIES_INTRADAY): snap onto a regular 390-bar
intraday grid (09:30..15:59), forward-fill untraded slots, and target the
same-session 60-bar return.

Then:

- Build scale-free, strictly causal features. Nothing here uses information
  from bar ``t + 1`` onwards, so the same code runs at inference time.
- Attach the target: the horizon-bar-ahead log return, divided by a volatility
  estimate known at ``t``.
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
from forecast.universe import allowed_symbols, hedge_symbol_for, is_equity_name
from mamba_lm.paths import REPO_ROOT, resolve_path


FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "ret_390",
    "range_hl",
    "body_co",
    "close_loc",
    "wick_up",
    "wick_dn",
    "vol_level",
    "vol_change",
    "volume_z",
    "turnover_z",
    "ret_vol",
    "peer_ret_1",
    "mkt_ret_1",
    "idio_ret_1",
    "sector_ret_1",
    "idio_sector",
    "cs_rank_1",
    "cs_ret_1",
    "cs_ret_5",
    "cs_ret_15",
    "cs_ret_60",
    "cs_volume",
    "cs_vol",
    "traded",
    "staleness",
    "new_session",
    "tod_sin",
    "tod_cos",
    "tod_frac",
    "dow_frac",
)

CROSS_SECTION_FEATURES: tuple[str, ...] = (
    "peer_ret_1",
    "mkt_ret_1",
    "idio_ret_1",
    "sector_ret_1",
    "idio_sector",
    "cs_rank_1",
    "cs_ret_1",
    "cs_ret_5",
    "cs_ret_15",
    "cs_ret_60",
    "cs_volume",
    "cs_vol",
)

# Same-bar CS z-scores (source column -> feature name). Known at close; not labels.
CS_ZSCORE_SOURCES: tuple[tuple[str, str], ...] = (
    ("ret_1", "cs_ret_1"),
    ("ret_5", "cs_ret_5"),
    ("ret_15", "cs_ret_15"),
    ("ret_60", "cs_ret_60"),
    ("volume_z", "cs_volume"),
    ("vol_level", "cs_vol"),
)

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


# --------------------------------------------------------------------------
# Loading and gridding
# --------------------------------------------------------------------------


def discover_symbol_files(
    data_dir: str | Path, interval: str | None = None
) -> list[Path]:
    resolved = resolve_path(data_dir)
    paths: list[Path] = []
    if interval:
        paths = sorted(resolved.glob(f"*_{interval}.parquet"))
    if not paths:
        paths = sorted(resolved.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no .parquet files found in {resolved.resolve()} "
            f"(looked for data_dir={str(data_dir)!r} relative to "
            f"{Path.cwd()} and {REPO_ROOT}). "
            "Pull bars with: python -m forecast.download --symbols AAPL"
        )
    return paths


def symbol_from_path(path: str | Path) -> str:
    """``data/AAPL_daily.parquet`` -> ``AAPL``."""
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

    Accepts the Alpha Vantage cache schema or a DatetimeIndex. Extra columns
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
    if "interval" in out.columns:
        keep.append("interval")
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
    """Read one parquet of OHLCV bars and normalize its column names."""
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


def build_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """One trading day per row. No session grid; weekends are simply absent."""
    out = df.copy()
    out["session"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["mos"] = 0
    out["traded"] = 1.0
    out = out.sort_values("datetime").drop_duplicates("session", keep="last")
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
    if "source" in out.columns:
        cols.append("source")
    return out.loc[:, cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Features and target
# --------------------------------------------------------------------------


def _causal_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Standardize against a trailing window that excludes the current bar."""
    roll = s.rolling(window, min_periods=min_periods)
    mean = roll.mean().shift(1)
    std = roll.std().shift(1)
    return (s - mean) / std.clip(lower=1e-8)


def _return_lags(cfg: DataConfig) -> tuple[int, ...]:
    if cfg.interval == "weekly":
        return (1, 4, 13, 26, 52)
    if cfg.interval == "monthly":
        return (1, 3, 6, 12, 24)
    if cfg.is_calendar():
        return (1, 5, 10, 15, 21)
    return (1, 5, 15, 60, 390)


def _vol_lag(cfg: DataConfig) -> int:
    return {"daily": 21, "weekly": 12, "monthly": 6}.get(cfg.interval, BARS_PER_SESSION)


def _stale_denom(cfg: DataConfig) -> float:
    periods = {"daily": 252.0, "weekly": 52.0, "monthly": 12.0}.get(
        cfg.interval, float(BARS_PER_SESSION)
    )
    return math.log(periods)


def _gap_limit_days(cfg: DataConfig) -> int:
    if cfg.interval == "weekly":
        return max(cfg.max_session_gap_days, 21)
    if cfg.interval == "monthly":
        return max(cfg.max_session_gap_days, 45)
    return cfg.max_session_gap_days


def compute_features(grid: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Attach features, the forward-return target, its scale, and a validity mask."""
    out = grid.copy()
    close = out["close"].astype(np.float64)
    log_close = np.log(close)

    session_dates = pd.to_datetime(out["session"])
    gap_days = session_dates.diff().dt.days.fillna(0)
    large_join = gap_days > _gap_limit_days(cfg)
    join_id = large_join.cumsum()

    r1 = log_close.diff().mask(large_join, np.nan)
    out["ret_raw"] = r1.fillna(0.0)
    # EWM realized vol, shifted so bar t's own return is excluded.
    ewm_var = r1.pow(2).ewm(halflife=cfg.vol_halflife, min_periods=cfg.vol_halflife).mean()
    sigma = np.sqrt(ewm_var).shift(1).clip(lower=cfg.vol_floor)
    out["sigma"] = sigma
    # Scale of a horizon-length return under a random walk.
    out["scale"] = sigma * math.sqrt(cfg.horizon)

    lags = _return_lags(cfg)
    for name, k in zip(("ret_1", "ret_5", "ret_15", "ret_60", "ret_390"), lags):
        crossed = join_id != join_id.shift(k)
        raw = log_close.diff(k).mask(crossed, np.nan)
        out[name] = raw / (sigma * math.sqrt(k))

    out["range_hl"] = ((out["high"] - out["low"]) / close) / sigma
    out["body_co"] = ((close - out["open"]) / close) / sigma
    hl = (out["high"] - out["low"]).clip(lower=1e-12)
    out["close_loc"] = (2.0 * (close - out["low"]) / hl) - 1.0
    upper = np.maximum(close, out["open"])
    lower = np.minimum(close, out["open"])
    out["wick_up"] = ((out["high"] - upper) / close) / sigma
    out["wick_dn"] = ((lower - out["low"]) / close) / sigma

    log_sigma = np.log(sigma)
    out["vol_level"] = _causal_zscore(log_sigma, cfg.z_window, cfg.z_min_periods)
    out["vol_change"] = log_sigma.diff(_vol_lag(cfg))

    log_volume = np.log1p(out["volume"])
    out["volume_z"] = _causal_zscore(log_volume, cfg.z_window, cfg.z_min_periods)
    out["turnover_z"] = _causal_zscore(
        np.log1p(out["volume"] * close), cfg.z_window, cfg.z_min_periods
    )
    out["ret_vol"] = out["ret_1"] * out["volume_z"]
    # Filled by attach_cross_section_features when more than one symbol exists.
    out["peer_ret_1"] = 0.0
    out["mkt_ret_1"] = 0.0
    out["idio_ret_1"] = 0.0
    out["sector_ret_1"] = 0.0
    out["idio_sector"] = 0.0
    out["cs_rank_1"] = 0.0
    for _src, dest in CS_ZSCORE_SOURCES:
        out[dest] = 0.0

    # Bars elapsed since the last real print, so stale prices are discountable.
    position = np.arange(len(out), dtype=np.float64)
    last_traded = pd.Series(
        np.where(out["traded"].to_numpy() > 0, position, np.nan), index=out.index
    ).ffill()
    out["staleness"] = np.log1p(position - last_traded) / _stale_denom(cfg)

    if cfg.is_calendar():
        month = pd.to_datetime(out["datetime"]).dt.to_period("M")
        out["new_session"] = (month != month.shift(1)).fillna(True).astype(np.float64)
        doy = (pd.to_datetime(out["datetime"]).dt.dayofyear.astype(np.float64) - 1.0) / 365.0
        out["tod_frac"] = doy
        out["tod_sin"] = np.sin(2.0 * math.pi * doy)
        out["tod_cos"] = np.cos(2.0 * math.pi * doy)
        same_session = np.ones(len(out), dtype=bool)
    else:
        out["new_session"] = (out["mos"] == 0).astype(np.float64)
        tod = out["mos"].astype(np.float64) / (BARS_PER_SESSION - 1)
        out["tod_frac"] = tod
        out["tod_sin"] = np.sin(2.0 * math.pi * tod)
        out["tod_cos"] = np.cos(2.0 * math.pi * tod)
        same_session = (out["mos"] + cfg.horizon) < BARS_PER_SESSION
    out["dow_frac"] = out["datetime"].dt.dayofweek.astype(np.float64) / 4.0

    # Target: log return realized `horizon` bars later, in volatility units.
    forward = log_close.shift(-cfg.horizon) - log_close
    out["target_raw"] = forward
    out["target"] = forward / out["scale"]
    horizon_traded = out["traded"].shift(-cfg.horizon)
    out["horizon_traded"] = horizon_traded.fillna(0.0)

    # A bar is trainable only if it is a real print, the horizon lands inside
    # the same session (intraday), and every feature has enough history.
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
    if cfg.max_abs_log_return > 0:
        valid = valid & (out["target_raw"].abs() <= cfg.max_abs_log_return)
    if cfg.max_abs_target > 0:
        valid = valid & (out["target"].abs() <= cfg.max_abs_target)
    out["valid"] = valid

    for name in FEATURE_NAMES:
        out[name] = out[name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[name] = out[name].clip(-cfg.clip, cfg.clip)
    out["target"] = out["target"].fillna(0.0)
    out["target_raw"] = out["target_raw"].fillna(0.0)
    return out


def _cross_section_key(panel: pd.DataFrame, cfg: DataConfig) -> pd.Series:
    """Align names on session dates (calendar) or exact timestamps (intraday)."""
    if cfg.is_calendar():
        return pd.to_datetime(panel["session"]).dt.tz_localize(None).dt.normalize()
    return pd.to_datetime(panel["datetime"])


def _traded_wide(
    panels: dict[str, pd.DataFrame],
    symbols: list[str],
    cfg: DataConfig,
    column: str,
) -> pd.DataFrame | None:
    """Date x symbol matrix of a feature on real prints only."""
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        panel = panels[sym]
        if column not in panel.columns:
            continue
        traded = panel["traded"].to_numpy(dtype=np.float64) > 0
        if not bool(traded.any()):
            continue
        parts.append(
            pd.DataFrame(
                {
                    "key": _cross_section_key(panel, cfg).to_numpy()[traded],
                    "symbol": sym,
                    "val": panel[column].to_numpy(dtype=np.float64)[traded],
                }
            )
        )
    if len(parts) < 2:
        return None
    long = pd.concat(parts, ignore_index=True)
    return long.pivot_table(index="key", columns="symbol", values="val", aggfunc="mean")


def _cs_zscore_wide(wide: pd.DataFrame) -> pd.DataFrame:
    """Z-score each date across names. Constant rows become 0."""
    mean = wide.mean(axis=1, skipna=True)
    std = wide.std(axis=1, skipna=True)
    std = std.where(std >= 1e-8, np.nan)
    return wide.sub(mean, axis=0).div(std, axis=0)


def _cs_rank_wide(wide: pd.DataFrame) -> pd.DataFrame:
    """Centered within-date percentile rank of ``ret_1`` in [-1, 1]."""
    ranks = wide.rank(axis=1, method="average")
    n = wide.notna().sum(axis=1).astype(np.float64)
    denom = (n - 1.0).where(n >= 2, np.nan)
    return ranks.sub(1.0, axis=0).div(denom, axis=0).mul(2.0).sub(1.0)


def _indexed_col(panel: pd.DataFrame, cfg: DataConfig, column: str) -> pd.Series:
    key = pd.Index(_cross_section_key(panel, cfg).to_numpy())
    return pd.Series(
        panel[column].to_numpy(dtype=np.float64), index=key
    ).groupby(level=0).last()


def trading_panel_symbols(panels: dict[str, pd.DataFrame], cfg: DataConfig) -> list[str]:
    """Names that enter the CS book (benchmark always excluded)."""
    bench = str(cfg.benchmark_symbol or "").upper()
    names = [s for s in panels if s != bench]
    if bool(getattr(cfg, "equities_only", False)):
        names = [s for s in names if is_equity_name(s)]
    return names


def attach_cross_section_features(
    panels: dict[str, pd.DataFrame],
    cfg: DataConfig,
) -> dict[str, pd.DataFrame]:
    """Peer / market / CS-z features known at bar ``t``. Strictly causal.

    ``mkt_ret_1`` is the benchmark ticker's ``ret_1`` when that parquet is in
    ``panels``, otherwise the equal-weight mean of trading names.
    ``peer_ret_1`` averages the other *trading* names (benchmark excluded).
    ``idio_ret_1`` is own ``ret_1`` minus ``mkt_ret_1``.
    ``sector_ret_1`` / ``idio_sector`` use the mapped sector ETF's same-bar
    ``ret_1`` (SPY if that ETF is missing). Never uses t+1.
    ``cs_*`` columns are same-day z-scores across trading names (not SPY).
    ``cs_rank_1`` is the centered within-date rank of ``ret_1``.
    """
    if not panels:
        return panels
    bench = str(cfg.benchmark_symbol or "").upper()
    trade_syms = trading_panel_symbols(panels, cfg)
    clip = float(cfg.clip)

    mkt: pd.Series | None = None
    if bench in panels:
        mkt = _indexed_col(panels[bench], cfg, "ret_1")
    elif len(trade_syms) >= 2:
        wide_mkt = _traded_wide(panels, trade_syms, cfg, "ret_1")
        if wide_mkt is not None:
            mkt = wide_mkt.mean(axis=1, skipna=True)

    hedge_ret: dict[str, pd.Series] = {}
    for hedge in {hedge_symbol_for(
        s,
        sector_residual=bool(getattr(cfg, "sector_residual", False)),
        benchmark=bench,
    ) for s in panels}:
        if hedge in panels:
            hedge_ret[hedge] = _indexed_col(panels[hedge], cfg, "ret_1")

    peer_wide = _traded_wide(panels, trade_syms, cfg, "ret_1") if len(trade_syms) >= 2 else None
    rank_wide = _cs_rank_wide(peer_wide) if peer_wide is not None else None
    cs_wides: dict[str, pd.DataFrame] = {}
    if bool(cfg.cs_zscore) and peer_wide is not None:
        for src, dest in CS_ZSCORE_SOURCES:
            src_wide = peer_wide if src == "ret_1" else _traded_wide(
                panels, trade_syms, cfg, src
            )
            if src_wide is None:
                continue
            cs_wides[dest] = _cs_zscore_wide(src_wide)

    if (
        mkt is None
        and peer_wide is None
        and not cs_wides
        and not hedge_ret
        and rank_wide is None
    ):
        return panels

    out: dict[str, pd.DataFrame] = {}
    for sym, panel in panels.items():
        p = panel.copy()
        keys = _cross_section_key(p, cfg)
        if mkt is not None:
            mkt_vals = keys.map(mkt).to_numpy(dtype=np.float64)
            mkt_vals = np.where(np.isfinite(mkt_vals), mkt_vals, 0.0)
        else:
            mkt_vals = np.zeros(len(p), dtype=np.float64)
        own = p["ret_1"].to_numpy(dtype=np.float64)
        peer = np.zeros(len(p), dtype=np.float64)
        if peer_wide is not None and sym in getattr(peer_wide, "columns", []):
            others = peer_wide.drop(columns=[sym], errors="ignore")
            if others.shape[1] > 0:
                peer_s = others.mean(axis=1, skipna=True)
                peer = keys.map(peer_s).to_numpy(dtype=np.float64)
                peer = np.where(np.isfinite(peer), peer, 0.0)
        hedge = hedge_symbol_for(
            sym,
            sector_residual=bool(getattr(cfg, "sector_residual", False)),
            benchmark=bench,
        )
        if hedge not in hedge_ret and bench in hedge_ret:
            hedge = bench
        if hedge in hedge_ret:
            sec_vals = keys.map(hedge_ret[hedge]).to_numpy(dtype=np.float64)
            sec_vals = np.where(np.isfinite(sec_vals), sec_vals, 0.0)
        else:
            sec_vals = np.zeros(len(p), dtype=np.float64)
        rank_vals = np.zeros(len(p), dtype=np.float64)
        if rank_wide is not None and sym in getattr(rank_wide, "columns", []):
            rank_vals = keys.map(rank_wide[sym]).to_numpy(dtype=np.float64)
            rank_vals = np.where(np.isfinite(rank_vals), rank_vals, 0.0)
        p["peer_ret_1"] = np.clip(peer, -clip, clip)
        p["mkt_ret_1"] = np.clip(mkt_vals, -clip, clip)
        p["idio_ret_1"] = np.clip(own - mkt_vals, -clip, clip)
        p["sector_ret_1"] = np.clip(sec_vals, -clip, clip)
        p["idio_sector"] = np.clip(own - sec_vals, -clip, clip)
        p["cs_rank_1"] = np.clip(rank_vals, -clip, clip)
        for _src, dest in CS_ZSCORE_SOURCES:
            if dest not in p.columns:
                p[dest] = 0.0
            wide = cs_wides.get(dest)
            if wide is None or sym not in getattr(wide, "columns", []):
                p[dest] = 0.0
                continue
            vals = keys.map(wide[sym]).to_numpy(dtype=np.float64)
            p[dest] = np.clip(np.where(np.isfinite(vals), vals, 0.0), -clip, clip)
        out[sym] = p
    return out


def attach_residual_target(
    panels: dict[str, pd.DataFrame],
    cfg: DataConfig,
) -> dict[str, pd.DataFrame]:
    """Replace the label with trailing-beta residual vs a hedge forward return.

    ``beta_t`` uses same-bar returns through ``t`` only. The hedge's *forward*
    return enters the label, never ``FEATURE_NAMES``. Default hedge is SPY;
    ``sector_residual`` uses the mapped sector ETF when that parquet exists.
    """
    bench = str(cfg.benchmark_symbol or "").upper()
    if not cfg.residual_target:
        return panels
    if bench not in panels and not any(
        hedge_symbol_for(
            s,
            sector_residual=bool(getattr(cfg, "sector_residual", False)),
            benchmark=bench,
        )
        in panels
        for s in panels
    ):
        return panels

    cache_fwd: dict[str, pd.Series] = {}
    cache_r: dict[str, pd.Series] = {}

    def _hedge_series(name: str) -> tuple[pd.Series, pd.Series] | None:
        if name not in panels:
            return None
        if name not in cache_fwd:
            cache_fwd[name] = _indexed_col(panels[name], cfg, "target_raw")
            cache_r[name] = _indexed_col(panels[name], cfg, "ret_raw")
        return cache_fwd[name], cache_r[name]

    hl = max(2, int(cfg.beta_halflife))
    out: dict[str, pd.DataFrame] = {}
    for sym, panel in panels.items():
        p = panel.copy()
        if sym == bench:
            out[sym] = p
            continue
        hedge = hedge_symbol_for(
            sym,
            sector_residual=bool(getattr(cfg, "sector_residual", False)),
            benchmark=bench,
        )
        series = _hedge_series(hedge) or _hedge_series(bench)
        if series is None:
            out[sym] = p
            continue
        hedge_fwd, hedge_r = series
        keys = _cross_section_key(p, cfg)
        x = keys.map(hedge_r).to_numpy(dtype=np.float64)
        fwd = keys.map(hedge_fwd).to_numpy(dtype=np.float64)
        own_r = p["ret_raw"].to_numpy(dtype=np.float64)
        frame = pd.DataFrame({"y": own_r, "x": x})
        cov = frame["y"].ewm(halflife=hl, min_periods=hl).cov(frame["x"])
        var = frame["x"].ewm(halflife=hl, min_periods=hl).var()
        beta = (cov / var.replace(0.0, np.nan)).fillna(0.0).clip(-5.0, 5.0).to_numpy()
        own_fwd = p["target_raw"].to_numpy(dtype=np.float64)
        fwd = np.where(np.isfinite(fwd), fwd, 0.0)
        resid = own_fwd - beta * fwd
        scale = p["scale"].to_numpy(dtype=np.float64)
        p["target_raw"] = resid
        p["target"] = np.divide(resid, scale, out=np.zeros_like(resid), where=scale > 0)
        valid = p["valid"].to_numpy(dtype=bool).copy()
        if cfg.max_abs_log_return > 0:
            valid &= np.abs(resid) <= cfg.max_abs_log_return
        if cfg.max_abs_target > 0:
            valid &= np.abs(p["target"].to_numpy()) <= cfg.max_abs_target
        p["valid"] = valid
        out[sym] = p
    return out


def assemble_panel(
    bars: pd.DataFrame,
    cfg: DataConfig,
    symbol: str,
    *,
    keep_latest_session: bool = False,
) -> pd.DataFrame:
    """Grid + features for an already-normalized OHLCV frame."""
    if cfg.is_calendar():
        grid = build_daily_index(bars)
    else:
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
    """OHLCV frame (Alpha Vantage cache or parquet-like) -> feature panel."""
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
    dates: np.ndarray | None = None  # [T] int64 days since epoch


def _date_keys(panel: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(panel["datetime"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    ts = ts.dt.normalize()
    return ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")).astype(np.int64).to_numpy()


def panel_to_arrays(panel: pd.DataFrame, symbol: str) -> SymbolArrays:
    return SymbolArrays(
        symbol=symbol,
        features=panel[list(FEATURE_NAMES)].to_numpy(dtype=np.float32),
        target=panel["target"].to_numpy(dtype=np.float32),
        scale=panel["scale"].to_numpy(dtype=np.float32),
        valid=panel["valid"].to_numpy(dtype=bool),
        dates=_date_keys(panel) if "datetime" in panel.columns else None,
    )


class SequenceDataset(Dataset):
    """Fixed-length windows over one or more symbols.

    ``__getitem__`` returns ``(x, y, mask, scale, date_id, sym_idx)``.
    Last-bar mode scores the trading object: only the final index is labelled.
    Windows are built on the full series so val/test last bars can see
    earlier-split history.
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
        supervise_last: int = 0,
        last_bar_only: bool = False,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if not 0 <= min_context < seq_len:
            raise ValueError(f"min_context must be in [0, seq_len); got {min_context}")
        if supervise_last < 0:
            raise ValueError("supervise_last must be >= 0")
        self.symbols = list(symbols)
        self.seq_len = seq_len
        self.min_context = min_context
        self.supervise_last = min(int(supervise_last), seq_len)
        self.last_bar_only = bool(last_bar_only)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.windows: list[tuple[int, int]] = []
        for i, sym in enumerate(self.symbols):
            length = len(sym.target)
            if length < seq_len:
                continue
            if self.last_bar_only or self.supervise_last == 1:
                for end in range(seq_len - 1, length, stride):
                    start = end - seq_len + 1
                    if bool(sym.valid[end]):
                        self.windows.append((i, start))
                continue
            for start in range(0, max(0, length - seq_len + 1), stride):
                if self.supervise_last > 0:
                    scorable = sym.valid[start + seq_len - self.supervise_last : start + seq_len]
                else:
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
        if self.last_bar_only:
            last = mask[-1]
            mask[:] = False
            mask[-1] = last
        elif self.supervise_last > 0:
            mask[: -self.supervise_last] = False
        date_id = 0
        if sym.dates is not None and len(sym.dates) >= stop:
            date_id = int(sym.dates[stop - 1])
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(sym.target[start:stop].copy()),
            torch.from_numpy(mask),
            torch.from_numpy(sym.scale[start:stop].copy()),
            torch.tensor(date_id, dtype=torch.int64),
            torch.tensor(sym_idx, dtype=torch.int64),
        )


class CrossSectionDataset(Dataset):
    """One item is every name that prints on a date, last-bar windows aligned."""

    def __init__(
        self,
        symbols: Sequence[SymbolArrays],
        seq_len: int,
        *,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        min_names: int = 8,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.symbols = list(symbols)
        self.seq_len = seq_len
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        buckets: dict[int, list[tuple[int, int]]] = {}
        for i, sym in enumerate(self.symbols):
            length = len(sym.target)
            if length < seq_len or sym.dates is None:
                continue
            for end in range(seq_len - 1, length):
                if not bool(sym.valid[end]):
                    continue
                key = int(sym.dates[end])
                buckets.setdefault(key, []).append((i, end))
        self.items: list[tuple[int, list[tuple[int, int]]]] = [
            (d, pairs)
            for d, pairs in sorted(buckets.items())
            if len(pairs) >= int(min_names)
        ]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def n_valid_bars(self) -> int:
        return int(sum(len(pairs) for _d, pairs in self.items))

    def __getitem__(self, index: int):
        date_id, pairs = self.items[index]
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        dates: list[int] = []
        syms: list[int] = []
        for sym_idx, end in pairs:
            sym = self.symbols[sym_idx]
            start = end - self.seq_len + 1
            x = sym.features[start : end + 1]
            if self.feature_mean is not None and self.feature_std is not None:
                x = (x - self.feature_mean) / self.feature_std
            mask = np.zeros(self.seq_len, dtype=bool)
            mask[-1] = bool(sym.valid[end])
            y = sym.target[start : end + 1].copy()
            sc = sym.scale[start : end + 1].copy()
            xs.append(np.ascontiguousarray(x, dtype=np.float32))
            ys.append(y)
            masks.append(mask)
            scales.append(sc)
            dates.append(int(date_id))
            syms.append(int(sym_idx))
        return (
            torch.from_numpy(np.stack(xs, axis=0)),
            torch.from_numpy(np.stack(ys, axis=0)),
            torch.from_numpy(np.stack(masks, axis=0)),
            torch.from_numpy(np.stack(scales, axis=0)),
            torch.tensor(dates, dtype=torch.int64),
            torch.tensor(syms, dtype=torch.int64),
        )


def collate_forecast(batch: list[tuple]) -> tuple[torch.Tensor, ...]:
    """Stack SequenceDataset items or concat CrossSectionDataset dates."""
    x0 = batch[0][0]
    if x0.dim() == 3:
        xs, ys, masks, scales, dates, syms = zip(*batch)
        return (
            torch.cat(xs, dim=0),
            torch.cat(ys, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(scales, dim=0),
            torch.cat(dates, dim=0),
            torch.cat(syms, dim=0),
        )
    return torch.utils.data.default_collate(batch)


def _source_mix(frame: pd.DataFrame) -> dict[str, float]:
    if "source" not in frame.columns:
        return {}
    traded = frame.loc[frame["traded"] > 0, "source"].dropna()
    if traded.empty:
        return {}
    counts = traded.astype(str).value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def split_session_bounds(n_sessions: int, cfg: DataConfig) -> tuple[int, int]:
    """Return ``(train_end_idx, val_end_idx)`` for chronological session splits."""
    n_test = int(round(n_sessions * cfg.test_fraction))
    n_val = int(round(n_sessions * cfg.val_fraction))
    n_train = n_sessions - n_val - n_test
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"{n_sessions} sessions cannot be split into train/val/test with "
            f"val_fraction={cfg.val_fraction}, test_fraction={cfg.test_fraction}"
        )
    return n_train, n_train + n_val


def embargo_calendar_horizon(panel: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Drop labels whose horizon close sits outside this split.

    Intraday targets cannot leave their session, so a session-boundary cut is
    already an embargo. Daily/weekly/monthly bars *are* sessions, so the last
    ``horizon`` train labels would otherwise be the first val/test returns.
    """
    if panel.empty or not cfg.is_calendar() or int(cfg.horizon) < 1:
        return panel
    out = panel.copy()
    valid = out["valid"].to_numpy(dtype=bool).copy()
    valid[max(0, len(out) - int(cfg.horizon)) :] = False
    out["valid"] = valid
    return out


def _mask_split_valid(
    base_valid: np.ndarray,
    in_split: np.ndarray,
    cfg: DataConfig,
) -> np.ndarray:
    valid = np.asarray(base_valid, dtype=bool).copy() & np.asarray(in_split, dtype=bool)
    if cfg.is_calendar() and int(cfg.horizon) >= 1:
        idx = np.flatnonzero(in_split)
        if idx.size:
            valid[idx[-int(cfg.horizon) :]] = False
    return valid


def _arrays_with_valid(src: SymbolArrays, valid: np.ndarray) -> SymbolArrays:
    return SymbolArrays(
        symbol=src.symbol,
        features=src.features,
        target=src.target,
        scale=src.scale,
        valid=valid,
        dates=src.dates,
    )


def global_session_cuts(
    panels: dict[str, pd.DataFrame],
    cfg: DataConfig,
) -> tuple[Any, Any]:
    """Union of session dates, then the usual 70/15/15 cuts."""
    sessions = pd.concat(
        [p["session"].drop_duplicates() for p in panels.values()],
        ignore_index=True,
    ).drop_duplicates().sort_values().to_numpy()
    cut_train, cut_val = split_session_bounds(len(sessions), cfg)
    return sessions[cut_train], sessions[cut_val]


def build_datasets(
    cfg: DataConfig,
    *,
    paths: Sequence[str | Path] | None = None,
    log_fn: Any | None = print,
) -> dict[str, Any]:
    """Load every symbol, split chronologically, and standardize on train only.

    Calendar splits share one ``train_end`` / ``val_end``. Windows run over the
    full series so a val last bar can use train history. The benchmark ticker
    (default SPY) is a market feature / residual label, not a training name.
    """
    from forecast.diagnostics import assert_calendar_price_quality

    paths = list(paths) if paths is not None else discover_symbol_files(
        cfg.data_dir, interval=cfg.interval
    )
    allowed = allowed_symbols(cfg.universe)
    if allowed is not None:
        kept = [p for p in paths if symbol_from_path(p) in allowed]
        skipped = len(paths) - len(kept)
        if log_fn:
            log_fn(
                f"universe={cfg.universe!r}: {len(kept)} parquet(s) "
                f"(dropped {skipped} non-members)"
            )
        paths = kept
        if not paths:
            raise FileNotFoundError(
                f"no parquets in {cfg.data_dir} match universe={cfg.universe!r}. "
                "Download with: python -m forecast.download --universe liquid "
                "--source yahoo --replace --interval daily"
            )

    raw_panels: dict[str, pd.DataFrame] = {}
    path_by_symbol: dict[str, Path] = {}
    for path in paths:
        if cfg.is_calendar():
            try:
                assert_calendar_price_quality(path, cfg, log_fn=log_fn)
            except ValueError:
                raise
        panel = build_panel(path, cfg)
        symbol = str(panel["symbol"].iloc[0])
        min_needed = int(cfg.seq_len) + max(int(cfg.horizon), 1)
        n_sess = int(panel["session"].nunique())
        if n_sess < min_needed:
            if log_fn:
                log_fn(
                    f"skip {symbol}: {n_sess} sessions < seq_len+horizon={min_needed} "
                    "(compact vendor slice, not usable)"
                )
            continue
        raw_panels[symbol] = panel
        path_by_symbol[symbol] = Path(path)
    if not raw_panels:
        raise FileNotFoundError("no symbol panels long enough to window")

    raw_panels = attach_cross_section_features(raw_panels, cfg)
    raw_panels = attach_residual_target(raw_panels, cfg)

    bench = str(cfg.benchmark_symbol or "").upper()
    trade_names = trading_panel_symbols(raw_panels, cfg)
    trade_panels = {s: raw_panels[s] for s in trade_names}
    train_from_ts: pd.Timestamp | None = None
    raw_from = str(getattr(cfg, "train_from", "") or "").strip()
    if raw_from:
        train_from_ts = pd.Timestamp(raw_from)
    if not trade_panels:
        raise ValueError(
            f"no trading names after filters (benchmark={bench}, "
            f"equities_only={bool(getattr(cfg, 'equities_only', False))}); "
            "add single-name equity parquets"
        )
    if log_fn:
        dropped = [s for s in raw_panels if s not in trade_panels and s != bench]
        log_fn(
            f"trading names={len(trade_panels)} equities_only="
            f"{bool(getattr(cfg, 'equities_only', False))} "
            f"sector_residual={bool(getattr(cfg, 'sector_residual', False))} "
            f"train_from={raw_from or 'all'}"
            + (f" (held out of book: {','.join(sorted(dropped))})" if dropped else "")
        )

    if cfg.global_calendar_split and cfg.is_calendar() and len(trade_panels) >= 1:
        train_end, val_end = global_session_cuts(trade_panels, cfg)
    else:
        train_end, val_end = None, None

    train_syms: list[SymbolArrays] = []
    val_syms: list[SymbolArrays] = []
    test_syms: list[SymbolArrays] = []
    meta: list[dict[str, Any]] = []

    for symbol, panel in trade_panels.items():
        path = path_by_symbol[symbol]
        sessions = panel["session"].drop_duplicates().sort_values().to_numpy()
        if train_end is None:
            cut_train, cut_val = split_session_bounds(len(sessions), cfg)
            sym_train_end, sym_val_end = sessions[cut_train], sessions[cut_val]
        else:
            sym_train_end, sym_val_end = train_end, val_end

        is_train = (panel["session"] < sym_train_end).to_numpy()
        if train_from_ts is not None:
            is_train = is_train & (pd.to_datetime(panel["session"]) >= train_from_ts).to_numpy()
        is_val = (
            (panel["session"] >= sym_train_end) & (panel["session"] < sym_val_end)
        ).to_numpy()
        is_test = (panel["session"] >= sym_val_end).to_numpy()
        base = panel_to_arrays(panel, symbol)
        base_valid = panel["valid"].to_numpy(dtype=bool)
        train_syms.append(
            _arrays_with_valid(base, _mask_split_valid(base_valid, is_train, cfg))
        )
        val_syms.append(
            _arrays_with_valid(base, _mask_split_valid(base_valid, is_val, cfg))
        )
        test_syms.append(
            _arrays_with_valid(base, _mask_split_valid(base_valid, is_test, cfg))
        )
        train_mix = _source_mix(panel.loc[is_train])
        val_mix = _source_mix(panel.loc[is_val])
        test_mix = _source_mix(panel.loc[is_test])
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
                "train_end": str(pd.Timestamp(sym_train_end).date()),
                "val_end": str(pd.Timestamp(sym_val_end).date()),
                "train_source_mix": train_mix,
                "val_source_mix": val_mix,
                "test_source_mix": test_mix,
                "residual_target": bool(cfg.residual_target and bench in raw_panels),
                "benchmark": bench if bench in raw_panels else "",
                "sector_residual": bool(getattr(cfg, "sector_residual", False)),
                "hedge": hedge_symbol_for(
                    symbol,
                    sector_residual=bool(getattr(cfg, "sector_residual", False)),
                    benchmark=bench,
                ),
                "train_from": raw_from,
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
    last_bar = bool(cfg.eval_last_bar) and cfg.is_calendar()
    eval_stride = 1 if last_bar else max(1, cfg.seq_len - cfg.min_context)
    common = {
        "feature_mean": mean,
        "feature_std": std,
        "min_context": cfg.min_context,
        "last_bar_only": last_bar,
    }
    use_cs = (
        last_bar
        and cfg.is_calendar()
        and len(train_syms) >= int(cfg.cross_section_min_names)
    )
    if use_cs:
        datasets = {
            "train": CrossSectionDataset(
                train_syms,
                cfg.seq_len,
                feature_mean=mean,
                feature_std=std,
                min_names=cfg.cross_section_min_names,
            ),
            "val": CrossSectionDataset(
                val_syms,
                cfg.seq_len,
                feature_mean=mean,
                feature_std=std,
                min_names=cfg.cross_section_min_names,
            ),
            "test": CrossSectionDataset(
                test_syms,
                cfg.seq_len,
                feature_mean=mean,
                feature_std=std,
                min_names=cfg.cross_section_min_names,
            ),
        }
        if log_fn:
            log_fn(
                f"cross-section dates: train={len(datasets['train'])} "
                f"val={len(datasets['val'])} test={len(datasets['test'])} "
                f"(min_names={cfg.cross_section_min_names})"
            )
        if any(len(datasets[k]) == 0 for k in ("train", "val", "test")):
            if log_fn:
                log_fn(
                    "cross-section empty on a split; falling back to last-bar sequences"
                )
            use_cs = False
            datasets = None
    if not use_cs:
        datasets = {
            "train": SequenceDataset(
                train_syms,
                cfg.seq_len,
                cfg.stride,
                supervise_last=cfg.supervise_last,
                **common,
            ),
            "val": SequenceDataset(
                val_syms, cfg.seq_len, eval_stride, supervise_last=1, **common
            ),
            "test": SequenceDataset(
                test_syms, cfg.seq_len, eval_stride, supervise_last=1, **common
            ),
        }
    if log_fn:
        for name, ds in datasets.items():
            log_fn(f"{name}: {len(ds)} windows, {ds.n_valid_bars} labelled bars")
    return {
        "datasets": datasets,
        "train_symbols": train_syms,
        "val_symbols": val_syms,
        "test_symbols": test_syms,
        "feature_mean": mean,
        "feature_std": std,
        "feature_names": list(FEATURE_NAMES),
        "meta": meta,
        "cross_section": use_cs,
        "cs_min_names": int(cfg.cross_section_min_names),
        "universe": cfg.universe,
        "equities_only": bool(getattr(cfg, "equities_only", False)),
        "train_from": raw_from,
        "n_trading_names": int(len(trade_panels)),
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


def fit_ridge_readout(
    symbols: Sequence[SymbolArrays],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    ridge: float = 1.0,
    cs_demean: bool = False,
    min_names: int = 8,
) -> tuple[np.ndarray, float, float]:
    """Train-only ridge of target on normalized features. Returns weight, bias, IC.

    ``cs_demean=True`` date-demeans X and y (the CS linear baseline). Inference
    still applies ``w·x + b`` without demeaning: within-date Pearson is invariant
    to a per-date additive shift, so CS IC matches the demeaned fit.
    """
    n_features = int(np.asarray(feature_mean).shape[0])
    mean = np.asarray(feature_mean, dtype=np.float64)
    std = np.asarray(feature_std, dtype=np.float64)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    rows_d: list[np.ndarray] = []
    have_dates = True
    for sym in symbols:
        if not bool(sym.valid.any()):
            continue
        raw = sym.features[sym.valid].astype(np.float64, copy=False)
        rows_x.append((raw - mean) / std)
        rows_y.append(sym.target[sym.valid].astype(np.float64, copy=False))
        if sym.dates is None:
            have_dates = False
            rows_d.append(np.full(int(sym.valid.sum()), -1, dtype=np.int64))
        else:
            rows_d.append(sym.dates[sym.valid].astype(np.int64, copy=False))
    if not rows_x:
        return np.zeros(n_features, dtype=np.float32), 0.0, float("nan")

    x_all = np.concatenate(rows_x, axis=0)
    y_all = np.concatenate(rows_y, axis=0)
    d_all = np.concatenate(rows_d, axis=0)
    x, y = x_all, y_all
    used_cs = False
    if cs_demean and have_dates:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for key in np.unique(d_all):
            sel = d_all == key
            if int(sel.sum()) < int(min_names):
                continue
            xd = x_all[sel]
            yd = y_all[sel]
            xs.append(xd - xd.mean(axis=0, keepdims=True))
            ys.append(yd - yd.mean())
        if xs:
            x = np.concatenate(xs, axis=0)
            y = np.concatenate(ys, axis=0)
            used_cs = True
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    lam = max(0.0, float(ridge))
    xtx = design.T @ design
    xtx.flat[:: xtx.shape[0] + 1] += lam
    try:
        coef = np.linalg.solve(xtx, design.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx, design.T @ y, rcond=None)[0]
    weights = coef[:-1].astype(np.float32)
    bias = float(coef[-1])
    pred_fit = x @ coef[:-1] + coef[-1]
    ic = float("nan")
    if pred_fit.size >= 2:
        pc = pred_fit - pred_fit.mean()
        yc = y - y.mean()
        denom = float(np.sqrt((pc * pc).sum() * (yc * yc).sum()))
        if denom > 1e-12:
            ic = float((pc * yc).sum() / denom)
    pred_std = float(pred_fit.std())
    y_std = float(y.std())
    if pred_std > 1e-8 and y_std > 1e-8 and np.isfinite(ic):
        amp = abs(ic) * y_std / pred_std
        weights = (weights * amp).astype(np.float32)
        bias = float(bias * amp)
    if used_cs and have_dates:
        pred_raw = x_all @ weights.astype(np.float64) + bias
        cs_ics: list[float] = []
        demeaned_p: list[np.ndarray] = []
        demeaned_y: list[np.ndarray] = []
        for key in np.unique(d_all):
            sel = d_all == key
            if int(sel.sum()) < int(min_names):
                continue
            a = pred_raw[sel]
            b = y_all[sel]
            a_c = a - a.mean()
            b_c = b - b.mean()
            demeaned_p.append(a_c)
            demeaned_y.append(b_c)
            denom = float(np.sqrt((a_c * a_c).sum() * (b_c * b_c).sum()))
            if denom > 1e-12:
                cs_ics.append(float((a_c * b_c).sum() / denom))
        if cs_ics:
            ic = float(np.mean(cs_ics))
        if demeaned_p:
            p_cat = np.concatenate(demeaned_p)
            y_cat = np.concatenate(demeaned_y)
            p_std = float(p_cat.std())
            y_std_cs = float(y_cat.std())
            if p_std > 1e-8 and y_std_cs > 1e-8 and np.isfinite(ic):
                amp = abs(ic) * y_std_cs / p_std
                weights = (weights * amp).astype(np.float32)
                bias = float(bias * amp)
        # Drop the global intercept: CS scores are relative.
        bias = 0.0
    return weights, bias, ic
