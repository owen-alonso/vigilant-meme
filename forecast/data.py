"""Turn raw OHLCV bars into sequences with a forward-return target.

Alpha Vantage daily bars (default): one row per trading day, next-day target.

1-minute bars (premium TIME_SERIES_INTRADAY): snap onto a regular 390-bar
intraday grid (09:30..15:59), forward-fill untraded slots, and target the
same-session 60-bar return.

Then:

- Build scale-free, strictly causal features. Nothing here uses information
  from bar ``t + 1`` onwards, so the same code runs at inference time.
- Attach the target: the horizon-bar-ahead log return, divided by a volatility
  estimate known at ``t``.     ``label_return='overnight'`` is
    ``log(open_{t+h}) - log(close_t)`` — next open is a label, never a feature.
    ``open_fill`` adds ``(fill_minutes/390)`` of the next session.
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
from forecast.overnight import (
    forward_log_return,
    formula_log_line,
    next_open_valid,
    normalize_label_return,
    same_bar_open_for_features,
    uses_next_open,
)
from forecast.universe import (
    allowed_symbols,
    hedge_symbol_for,
    industry_symbol_for,
    is_equity_name,
)
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
    "cs_ret1_x_vol",
    "cs_rank_x_vol",
    "idio_x_csvol",
    "cs_ret1_x_idio",
    "cs_rank_x_idio",
    "cs_ret1_x_rank",
    "cs_vol_sq",
    "idio_sq",
    "cs_ret1_sq",
    "cs_rank_sq",
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
    "cs_ret1_x_vol",
    "cs_rank_x_vol",
    "idio_x_csvol",
    "cs_ret1_x_idio",
    "cs_rank_x_idio",
    "cs_ret1_x_rank",
    "cs_vol_sq",
    "idio_sq",
    "cs_ret1_sq",
    "cs_rank_sq",
)

# Pairwise products of CS columns (known at close; not labels).
CS_PRODUCTS: tuple[tuple[str, str, str], ...] = (
    ("cs_ret_1", "cs_vol", "cs_ret1_x_vol"),
    ("cs_rank_1", "cs_vol", "cs_rank_x_vol"),
    ("idio_sector", "cs_vol", "idio_x_csvol"),
    ("cs_ret_1", "idio_sector", "cs_ret1_x_idio"),
    ("cs_rank_1", "idio_sector", "cs_rank_x_idio"),
    ("cs_ret_1", "cs_rank_1", "cs_ret1_x_rank"),
    ("cs_vol", "cs_vol", "cs_vol_sq"),
    ("idio_sector", "idio_sector", "idio_sq"),
    ("cs_ret_1", "cs_ret_1", "cs_ret1_sq"),
    ("cs_rank_1", "cs_rank_1", "cs_rank_sq"),
)
CS_PRODUCT_FEATURES: tuple[str, ...] = tuple(name for _, _, name in CS_PRODUCTS)
VOL_FEATURES: tuple[str, ...] = (
    "vol_level",
    "vol_change",
    "volume_z",
    "turnover_z",
    "ret_vol",
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
    for col in ("open", "high", "low"):
        px = pd.to_numeric(out[col], errors="coerce")
        out[col] = px.where(px > 0)
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


def _forward_log_return(out: pd.DataFrame, log_close: pd.Series, cfg: DataConfig) -> pd.Series:
    """Causal label return. Features never see these future prices.

    Overnight is ``log(open_{t+h}) - log(close_t)``. Invalid / non-positive
    next opens are NaN (not clipped to 1e-12). Same-bar ``open_t`` is unused
    here; it only appears in candle features via ``same_bar_open_for_features``.
    """
    del log_close  # recomputed inside forward_log_return from raw close
    return forward_log_return(
        close=out["close"],
        open_px=out["open"],
        kind=str(getattr(cfg, "label_return", "close") or "close"),
        horizon=int(cfg.horizon),
        fill_minutes=int(getattr(cfg, "fill_minutes", 0) or 0),
    )


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
    open_feat = same_bar_open_for_features(out["open"], close)
    out["body_co"] = ((close - open_feat) / close) / sigma
    hl = (out["high"] - out["low"]).clip(lower=1e-12)
    out["close_loc"] = (2.0 * (close - out["low"]) / hl) - 1.0
    upper = np.maximum(close, open_feat)
    lower = np.minimum(close, open_feat)
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
    for _a, _b, dest in CS_PRODUCTS:
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
    forward = _forward_log_return(out, log_close, cfg)
    out["target_raw"] = forward
    out["target"] = forward / out["scale"]
    # Raw overnight gap is always stored (P(up) / PIT). Residualization
    # must not overwrite it — next open is a label, never a feature.
    out["overnight_r"] = forward_log_return(
        close=out["close"],
        open_px=out["open"],
        kind="overnight",
        horizon=int(cfg.horizon),
        fill_minutes=0,
    )
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
    if uses_next_open(getattr(cfg, "label_return", "close")):
        valid = valid & next_open_valid(out["open"], int(cfg.horizon)).to_numpy()
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
        for a, b, dest in CS_PRODUCTS:
            left = p[a].to_numpy(dtype=np.float64) if a in p.columns else np.zeros(len(p))
            right = p[b].to_numpy(dtype=np.float64) if b in p.columns else np.zeros(len(p))
            p[dest] = np.clip(left * right, -clip, clip)
        out[sym] = p
    return out


def _ewm_beta(y: np.ndarray, x: np.ndarray, hl: int) -> np.ndarray:
    frame = pd.DataFrame({"y": y, "x": x})
    cov = frame["y"].ewm(halflife=hl, min_periods=hl).cov(frame["x"])
    var = frame["x"].ewm(halflife=hl, min_periods=hl).var()
    return (cov / var.replace(0.0, np.nan)).fillna(0.0).clip(-5.0, 5.0).to_numpy()


def _ewm_multi_beta(y: np.ndarray, xs: list[np.ndarray], hl: int) -> list[np.ndarray]:
    """Causal EWM betas of ``y`` on one or more ``xs``. Falls back if singular."""
    if len(xs) == 1:
        return [_ewm_beta(y, xs[0], hl)]
    cols = {f"x{i}": xs[i] for i in range(len(xs))}
    cols["y"] = y
    df = pd.DataFrame(cols)
    y_s = df["y"]
    k = len(xs)
    cov_yx = [
        y_s.ewm(halflife=hl, min_periods=hl).cov(df[f"x{i}"]).to_numpy()
        for i in range(k)
    ]
    cov_xx = np.zeros((len(df), k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                v = df[f"x{i}"].ewm(halflife=hl, min_periods=hl).var().to_numpy()
                cov_xx[:, i, j] = v
            else:
                c = df[f"x{i}"].ewm(halflife=hl, min_periods=hl).cov(df[f"x{j}"]).to_numpy()
                cov_xx[:, i, j] = c
                cov_xx[:, j, i] = c
    rhs = np.stack(cov_yx, axis=1)
    betas = np.zeros((len(df), k), dtype=np.float64)
    eye = 1e-8 * np.eye(k)
    for t in range(len(df)):
        a = cov_xx[t] + eye
        try:
            betas[t] = np.linalg.solve(a, rhs[t])
        except np.linalg.LinAlgError:
            betas[t] = 0.0
    return [np.clip(betas[:, i], -5.0, 5.0) for i in range(k)]


def attach_residual_target(
    panels: dict[str, pd.DataFrame],
    cfg: DataConfig,
) -> dict[str, pd.DataFrame]:
    """Replace the label with trailing-beta residual vs hedge forward return(s).

    ``beta_t`` uses same-bar returns through ``t`` only. Hedge *forward* return
    enters the label, never ``FEATURE_NAMES``. Default hedge is SPY;
    ``sector_residual`` uses the mapped sector ETF when that parquet exists.
    ``double_residual`` adds SPY as a second factor next to the sector hedge.
    ``industry_residual`` adds a mapped industry ETF when present.
    ``residualize_features`` subtracts the same causal betas times same-bar
    hedge ``ret_*`` from the name's own ``ret_*`` (not a label leak).
    ``label_return`` selects which forward log-return is residualized.
    Overnight is ``log(open_{t+h})-log(close_t)``; the hedge's *forward*
    overnight return is a label term. Features stay at close ``t``.
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
    cache_ret: dict[tuple[str, str], pd.Series] = {}

    def _hedge_series(name: str) -> tuple[pd.Series, pd.Series] | None:
        if name not in panels:
            return None
        if name not in cache_fwd:
            cache_fwd[name] = _indexed_col(panels[name], cfg, "target_raw")
            cache_r[name] = _indexed_col(panels[name], cfg, "ret_raw")
        return cache_fwd[name], cache_r[name]

    def _hedge_ret(name: str, col: str) -> pd.Series | None:
        if name not in panels or col not in panels[name].columns:
            return None
        key = (name, col)
        if key not in cache_ret:
            cache_ret[key] = _indexed_col(panels[name], cfg, col)
        return cache_ret[key]

    hl = max(2, int(cfg.beta_halflife))
    double = bool(getattr(cfg, "double_residual", False))
    industry = bool(getattr(cfg, "industry_residual", False))
    resid_feat = bool(getattr(cfg, "residualize_features", False))
    feat_cols = ("ret_1", "ret_5", "ret_15", "ret_60", "ret_390")
    out: dict[str, pd.DataFrame] = {}
    for sym, panel in panels.items():
        p = panel.copy()
        if sym == bench:
            out[sym] = p
            continue
        names: list[str] = []
        sector = hedge_symbol_for(
            sym,
            sector_residual=bool(getattr(cfg, "sector_residual", False)),
            benchmark=bench,
        )
        if double:
            if bench in panels:
                names.append(bench)
            if sector != bench and sector in panels:
                names.append(sector)
        else:
            if sector in panels:
                names.append(sector)
            elif bench in panels:
                names.append(bench)
        if industry:
            ind = industry_symbol_for(sym)
            if ind and ind in panels and ind not in names:
                names.append(ind)
        series = [_hedge_series(n) for n in names]
        series = [s for s in series if s is not None]
        if not series:
            out[sym] = p
            continue
        keys = _cross_section_key(p, cfg)
        own_r = p["ret_raw"].to_numpy(dtype=np.float64)
        xs = []
        fwds = []
        for hedge_fwd, hedge_r in series:
            xs.append(keys.map(hedge_r).to_numpy(dtype=np.float64))
            fwd = keys.map(hedge_fwd).to_numpy(dtype=np.float64)
            fwds.append(np.where(np.isfinite(fwd), fwd, 0.0))
        betas = _ewm_multi_beta(own_r, xs, hl) if len(xs) > 1 else [_ewm_beta(own_r, xs[0], hl)]
        own_fwd = p["target_raw"].to_numpy(dtype=np.float64)
        resid = own_fwd.astype(np.float64, copy=True)
        for b, fwd in zip(betas, fwds):
            resid = resid - b * fwd
        scale = p["scale"].to_numpy(dtype=np.float64)
        p["target_raw"] = resid
        p["target"] = np.divide(resid, scale, out=np.zeros_like(resid), where=scale > 0)
        if resid_feat:
            for col in feat_cols:
                if col not in p.columns:
                    continue
                own = p[col].to_numpy(dtype=np.float64)
                adj = own.copy()
                for b, hedge_name in zip(betas, names):
                    hs = _hedge_ret(hedge_name, col)
                    if hs is None:
                        continue
                    hx = keys.map(hs).to_numpy(dtype=np.float64)
                    hx = np.where(np.isfinite(hx), hx, 0.0)
                    adj = adj - b * hx
                p[col] = adj
            if "idio_ret_1" in p.columns and "ret_1" in p.columns and "mkt_ret_1" in p.columns:
                p["idio_ret_1"] = p["ret_1"] - p["mkt_ret_1"]
            if "idio_sector" in p.columns and "ret_1" in p.columns and "sector_ret_1" in p.columns:
                p["idio_sector"] = p["ret_1"] - p["sector_ret_1"]
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
    features: np.ndarray  # [T, F] float32 (may be a read-only memmap)
    target: np.ndarray  # [T]    float32, volatility units
    scale: np.ndarray  # [T]    float32, target -> log-return multiplier
    valid: np.ndarray  # [T]    bool
    dates: np.ndarray | None = None  # [T] int64 days since epoch
    overnight_r: np.ndarray | None = None  # [T] raw close_t -> open_{t+h}
    next_split_days: np.ndarray | None = None  # [T] PIT side-tape; 1 = drop


def _date_keys(panel: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(panel["datetime"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    ts = ts.dt.normalize()
    return ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")).astype(np.int64).to_numpy()


def panel_to_arrays(panel: pd.DataFrame, symbol: str) -> SymbolArrays:
    overnight = None
    if "overnight_r" in panel.columns:
        overnight = panel["overnight_r"].to_numpy(dtype=np.float32)
    split_days = None
    if "next_split_days" in panel.columns:
        split_days = panel["next_split_days"].to_numpy(dtype=np.int16)
    return SymbolArrays(
        symbol=symbol,
        features=panel[list(FEATURE_NAMES)].to_numpy(dtype=np.float32),
        target=panel["target"].to_numpy(dtype=np.float32),
        scale=panel["scale"].to_numpy(dtype=np.float32),
        valid=panel["valid"].to_numpy(dtype=bool),
        dates=_date_keys(panel) if "datetime" in panel.columns else None,
        overnight_r=overnight,
        next_split_days=split_days,
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
    """Drop labels whose horizon bar sits outside this split.

    Overnight uses ``open_{t+h}`` on that same next bar, so the last
    ``horizon`` train dates would otherwise label the first val open.
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
        overnight_r=src.overnight_r,
        next_split_days=src.next_split_days,
    )


def _ts_to_day(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    ts = pd.Timestamp(value)
    return int((ts.normalize() - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D"))


def _apply_pit_to_base(
    base: SymbolArrays,
    tape: dict[int, float] | None,
    membership: dict[str, Any] | None = None,
) -> SymbolArrays:
    from forecast.pit import (
        mask_membership_asof,
        mask_overnight_pit,
        next_split_days_aligned,
    )

    length = int(len(base.target))
    days = base.next_split_days
    # Live Data Manager factors tape wins over mmap-cached next_split_days.
    if tape:
        days = next_split_days_aligned(base.dates, tape, length=length)
    valid = np.asarray(base.valid, dtype=bool)
    if days is not None:
        valid = mask_overnight_pit(valid, days)
    if membership is not None:
        valid = mask_membership_asof(valid, base.dates, base.symbol, membership)
    if (
        days is None
        and membership is None
        and np.array_equal(valid, np.asarray(base.valid, dtype=bool))
    ):
        return base
    return SymbolArrays(
        symbol=base.symbol,
        features=base.features,
        target=base.target,
        scale=base.scale,
        valid=valid,
        dates=base.dates,
        overnight_r=base.overnight_r,
        next_split_days=(None if days is None else np.asarray(days, dtype=np.int16)),
    )


def _parse_cut_ts(value: Any) -> pd.Timestamp | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return pd.Timestamp(raw)


def explicit_calendar_cuts(cfg: DataConfig) -> tuple[Any, Any, Any] | None:
    """Return ``(train_end, val_end, test_end)`` when both train/val cuts are set.

    Ends are exclusive timestamps. ``test_end`` may be None (no upper bound).
    """
    train_end = _parse_cut_ts(getattr(cfg, "train_end", ""))
    val_end = _parse_cut_ts(getattr(cfg, "val_end", ""))
    test_end = _parse_cut_ts(getattr(cfg, "test_end", ""))
    if train_end is None and val_end is None and test_end is None:
        return None
    if train_end is None or val_end is None:
        raise ValueError(
            "explicit calendar cuts need both train_end and val_end "
            "(YYYY-MM-DD, exclusive); test_end is optional"
        )
    if val_end <= train_end:
        raise ValueError(f"val_end {val_end.date()} must be after train_end {train_end.date()}")
    if test_end is not None and test_end < val_end:
        raise ValueError(f"test_end {test_end.date()} must be >= val_end {val_end.date()}")
    return train_end, val_end, test_end


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


def _global_date_cuts(bases: dict[str, SymbolArrays], cfg: DataConfig) -> tuple[int, int]:
    chunks = [np.unique(b.dates) for b in bases.values() if b.dates is not None and len(b.dates)]
    if not chunks:
        raise ValueError("no dates on mmap/array bases for a calendar split")
    sessions = np.unique(np.concatenate(chunks))
    cut_train, cut_val = split_session_bounds(len(sessions), cfg)
    return int(sessions[cut_train]), int(sessions[cut_val])


def _split_bases(
    bases: dict[str, SymbolArrays],
    cfg: DataConfig,
    *,
    path_by_symbol: dict[str, Path] | None = None,
    log_fn: Any | None = None,
) -> tuple[list[SymbolArrays], list[SymbolArrays], list[SymbolArrays], list[dict[str, Any]]]:
    """Apply calendar cuts + PIT-aware valid masks. Features stay mmap-backed."""
    raw_from = str(getattr(cfg, "train_from", "") or "").strip()
    train_from_day = _ts_to_day(raw_from) if raw_from else None
    pinned = explicit_calendar_cuts(cfg)
    if pinned is not None:
        train_end_day = _ts_to_day(pinned[0])
        val_end_day = _ts_to_day(pinned[1])
        test_end_day = _ts_to_day(pinned[2]) if pinned[2] is not None else None
        per_symbol = False
    elif cfg.global_calendar_split and cfg.is_calendar() and len(bases) >= 1:
        train_end_day, val_end_day = _global_date_cuts(bases, cfg)
        test_end_day = None
        per_symbol = False
    else:
        train_end_day = val_end_day = test_end_day = None
        per_symbol = True

    train_syms: list[SymbolArrays] = []
    val_syms: list[SymbolArrays] = []
    test_syms: list[SymbolArrays] = []
    meta: list[dict[str, Any]] = []
    paths = path_by_symbol or {}
    bench = str(cfg.benchmark_symbol or "").upper()

    for symbol, base in bases.items():
        dates = base.dates
        if dates is None:
            raise ValueError(f"{symbol}: panel arrays need dates for CS splits")
        dates_i = np.asarray(dates, dtype=np.int64)
        sessions = np.unique(dates_i)
        if per_symbol:
            cut_train, cut_val = split_session_bounds(len(sessions), cfg)
            sym_train_end, sym_val_end = int(sessions[cut_train]), int(sessions[cut_val])
        else:
            sym_train_end, sym_val_end = int(train_end_day), int(val_end_day)
        is_train = dates_i < sym_train_end
        if train_from_day is not None:
            is_train = is_train & (dates_i >= int(train_from_day))
        is_val = (dates_i >= sym_train_end) & (dates_i < sym_val_end)
        is_test = dates_i >= sym_val_end
        if test_end_day is not None:
            is_test = is_test & (dates_i < int(test_end_day))
        train_syms.append(
            _arrays_with_valid(base, _mask_split_valid(base.valid, is_train, cfg))
        )
        val_syms.append(
            _arrays_with_valid(base, _mask_split_valid(base.valid, is_val, cfg))
        )
        test_syms.append(
            _arrays_with_valid(base, _mask_split_valid(base.valid, is_test, cfg))
        )
        meta.append(
            {
                "symbol": symbol,
                "path": str(paths.get(symbol, "")),
                "sessions": int(len(sessions)),
                "grid_bars": int(len(base.target)),
                "valid_bars": int(np.asarray(base.valid).sum()),
                "train_end": str(pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(sym_train_end)))[:10],
                "val_end": str(pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(sym_val_end)))[:10],
                "test_end": (
                    str(pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(test_end_day)))[:10]
                    if test_end_day is not None
                    else ""
                ),
                "benchmark": bench,
                "sector_residual": bool(getattr(cfg, "sector_residual", False)),
                "hedge": hedge_symbol_for(
                    symbol,
                    sector_residual=bool(getattr(cfg, "sector_residual", False)),
                    benchmark=bench,
                ),
                "train_from": raw_from,
                "mmap": bool(isinstance(base.features, np.memmap)),
            }
        )
        if log_fn:
            m = meta[-1]
            log_fn(
                f"{symbol}: {m['sessions']} sessions, {m['grid_bars']} grid bars, "
                f"{m['valid_bars']} labelled | train<{m['train_end']} "
                f"val<{m['val_end']} test"
                + (f"<{m['test_end']}" if m.get("test_end") else f">= {m['val_end']}")
                + (" [mmap]" if m.get("mmap") else "")
            )
    return train_syms, val_syms, test_syms, meta


def _finalize_bundle(
    cfg: DataConfig,
    train_syms: list[SymbolArrays],
    val_syms: list[SymbolArrays],
    test_syms: list[SymbolArrays],
    meta: list[dict[str, Any]],
    *,
    log_fn: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    datasets: dict[str, Any] | None = None
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
        if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
            if log_fn:
                log_fn(
                    "cross-section empty on train/val; falling back to last-bar sequences"
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
    raw_from = str(getattr(cfg, "train_from", "") or "").strip()
    out = {
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
        "train_end": str(getattr(cfg, "train_end", "") or ""),
        "val_end": str(getattr(cfg, "val_end", "") or ""),
        "test_end": str(getattr(cfg, "test_end", "") or ""),
        "n_trading_names": int(len(train_syms)),
        "label_return": normalize_label_return(getattr(cfg, "label_return", "close")),
    }
    if extra:
        out.update(extra)
    return out


def _pit_tape_for_cfg(cfg: DataConfig, symbols: Sequence[str]) -> dict[str, dict[int, float]]:
    from forecast.pit import load_pit_side_tape

    pit_dir = str(getattr(cfg, "pit_dir", "") or "").strip() or None
    return load_pit_side_tape(cfg.data_dir, pit_dir=pit_dir, symbols=symbols)


def _pit_membership_for_cfg(cfg: DataConfig) -> dict[str, Any] | None:
    from forecast.pit import load_liquid_membership

    pit_dir = str(getattr(cfg, "pit_dir", "") or "").strip() or None
    return load_liquid_membership(cfg.data_dir, pit_dir=pit_dir)


def _build_datasets_from_mmap(
    cfg: DataConfig,
    manifest_path: Path,
    *,
    log_fn: Any | None = None,
) -> dict[str, Any]:
    """CS train path: memmap panels only. No DataFrame rebuild, no ``.pt``."""
    from forecast.panel_mmap import (
        list_mmap_symbols,
        load_manifest,
        load_symbol_mmap,
        symbol_arrays_from_mmap,
    )
    from forecast.universe import is_equity_name

    payload = load_manifest(manifest_path)
    allowed = allowed_symbols(cfg.universe)
    bench = str(cfg.benchmark_symbol or "").upper()
    tape = _pit_tape_for_cfg(cfg, list_mmap_symbols(payload))
    membership = _pit_membership_for_cfg(cfg)
    bases: dict[str, SymbolArrays] = {}
    for symbol in list_mmap_symbols(payload):
        if allowed is not None and symbol not in allowed:
            continue
        if symbol == bench:
            continue
        if bool(getattr(cfg, "equities_only", False)) and not is_equity_name(symbol):
            continue
        packed = load_symbol_mmap(payload, symbol, mmap_mode="r")
        if not isinstance(packed.features, np.memmap):
            raise RuntimeError(f"{symbol}: load_symbol_mmap did not return a memmap")
        if packed.features.dtype != np.float32:
            raise RuntimeError(f"{symbol}: mmap features must be float32")
        base = symbol_arrays_from_mmap(packed)
        bases[symbol] = _apply_pit_to_base(base, tape.get(symbol), membership)
    if not bases:
        raise FileNotFoundError(
            f"mmap manifest {manifest_path} had no trading names "
            f"(universe={cfg.universe!r})"
        )
    if log_fn:
        log_fn(
            f"mmap feed {manifest_path}: {len(bases)} trading names "
            f"(no DataFrame rebuild, mmap_mode=r, num_workers=0)"
            + (f" PIT factors={len(tape)} names" if tape else " PIT factors=none")
            + (
                f" membership={membership.get('_path')}"
                if membership
                else " membership=off"
            )
        )
    train_syms, val_syms, test_syms, meta = _split_bases(bases, cfg, log_fn=log_fn)
    return _finalize_bundle(
        cfg,
        train_syms,
        val_syms,
        test_syms,
        meta,
        log_fn=log_fn,
        extra={
            "mmap_manifest": str(manifest_path),
            "mmap": True,
            "cs_train_pt": False,
        },
    )


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

    When ``data/_panel_cache/mmap_manifest.json`` exists (or
    ``cfg.mmap_manifest``), the CS train path reads mmap'd panels and does
    **not** rebuild DataFrames or load ``cs_train_*.pt``.
    """
    from forecast.diagnostics import assert_calendar_price_quality
    from forecast.panel_mmap import (
        assert_no_cs_train_pt,
        resolve_mmap_manifest,
        write_mmap_cache,
        load_manifest,
        load_symbol_mmap,
        symbol_arrays_from_mmap,
    )

    assert_no_cs_train_pt(getattr(cfg, "mmap_manifest", "") or None)
    if bool(getattr(cfg, "use_mmap", True)):
        try:
            manifest_path = resolve_mmap_manifest(
                cfg.data_dir,
                str(getattr(cfg, "mmap_manifest", "") or "") or None,
            )
        except FileNotFoundError:
            if str(getattr(cfg, "mmap_manifest", "") or "").strip():
                raise
            manifest_path = None
        if manifest_path is not None and not bool(getattr(cfg, "write_mmap", False)):
            return _build_datasets_from_mmap(cfg, manifest_path, log_fn=log_fn)

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
    tape = _pit_tape_for_cfg(cfg, list(trade_panels))
    membership = _pit_membership_for_cfg(cfg)
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
            f"train_from={raw_from or 'all'} "
            f"{formula_log_line(getattr(cfg, 'label_return', 'close'), horizon=int(cfg.horizon), fill_minutes=int(getattr(cfg, 'fill_minutes', 0) or 0))}"
            + (f" (held out of book: {','.join(sorted(dropped))})" if dropped else "")
            + (
                f" PIT factors={len(tape)} names"
                if tape
                else " PIT factors=none"
            )
            + (
                f" membership={membership.get('_path')}"
                if membership
                else " membership=off"
            )
        )

    pinned = explicit_calendar_cuts(cfg)
    if pinned is not None:
        train_end, val_end, test_end = pinned
    elif cfg.global_calendar_split and cfg.is_calendar() and len(trade_panels) >= 1:
        train_end, val_end = global_session_cuts(trade_panels, cfg)
        test_end = None
    else:
        train_end, val_end, test_end = None, None, None

    train_syms: list[SymbolArrays] = []
    val_syms: list[SymbolArrays] = []
    test_syms: list[SymbolArrays] = []
    meta: list[dict[str, Any]] = []
    bases: dict[str, SymbolArrays] = {}

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
        if test_end is not None:
            is_test = is_test & (panel["session"] < test_end).to_numpy()
        base = _apply_pit_to_base(
            panel_to_arrays(panel, symbol), tape.get(symbol), membership
        )
        bases[symbol] = base
        base_valid = base.valid
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
                "test_end": (
                    str(pd.Timestamp(test_end).date()) if test_end is not None else ""
                ),
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
                f"val<{m['val_end']} test"
                + (
                    f"<{m['test_end']}"
                    if m.get("test_end")
                    else f">= {m['val_end']}"
                )
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

    extra: dict[str, Any] = {"mmap": False, "cs_train_pt": False}
    if bool(getattr(cfg, "write_mmap", False)) and bases:
        cache_raw = str(getattr(cfg, "mmap_cache_dir", "") or "").strip()
        cache_dir = (
            resolve_path(cache_raw)
            if cache_raw
            else resolve_path(Path(cfg.data_dir) / "_panel_cache")
        )
        dest = write_mmap_cache(
            list(bases.values()),
            cache_dir,
            feature_names=FEATURE_NAMES,
            label_return=normalize_label_return(getattr(cfg, "label_return", "close")),
            universe=str(cfg.universe or ""),
        )
        man = load_manifest(dest)

        def _swap(split: list[SymbolArrays]) -> list[SymbolArrays]:
            out: list[SymbolArrays] = []
            for src in split:
                packed = load_symbol_mmap(man, src.symbol, mmap_mode="r")
                out.append(symbol_arrays_from_mmap(packed, valid=src.valid))
            return out

        train_syms = _swap(train_syms)
        val_syms = _swap(val_syms)
        test_syms = _swap(test_syms)
        extra["mmap_manifest"] = str(dest)
        extra["mmap"] = True
        if log_fn:
            log_fn(
                f"wrote mmap panel cache -> {dest} "
                "(train CS now reads memmaps; not cs_train_*.pt)"
            )
    del raw_panels, trade_panels, bases
    return _finalize_bundle(
        cfg,
        train_syms,
        val_syms,
        test_syms,
        meta,
        log_fn=log_fn,
        extra=extra,
    )


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
    cs_zscore: bool = False,
    rank_target: bool = False,
    feature_mask_bool: np.ndarray | None = None,
    date_halflife: float = 0.0,
    session_halflife: float = 0.0,
    y_winsor: float = 0.0,
    feat_winsor: float = 0.0,
    drop_disp_q: float = 0.0,
    huber_delta: float = 0.0,
    sign_constrain: bool = False,
    drop_crashes: bool = False,
    exclude_dates: set[int] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Train-only ridge of target on normalized features. Returns weight, bias, IC.

    ``cs_demean=True`` date-demeans X and y (the CS linear baseline). Inference
    still applies ``w·x + b`` without demeaning: within-date Pearson is invariant
    to a per-date additive shift, so CS IC matches the demeaned fit.
    """
    from forecast.ridge import fit_ridge_xy, labelled_rows

    x, y, dates = labelled_rows(symbols, feature_mean, feature_std)
    return fit_ridge_xy(
        x,
        y,
        dates,
        ridge=ridge,
        min_names=min_names,
        cs_demean=cs_demean,
        cs_zscore=cs_zscore,
        rank_target=rank_target,
        feature_mask_bool=feature_mask_bool,
        date_halflife=date_halflife,
        session_halflife=session_halflife,
        y_winsor=y_winsor,
        feat_winsor=feat_winsor,
        drop_disp_q=drop_disp_q,
        huber_delta=huber_delta,
        sign_constrain=sign_constrain,
        drop_crashes=drop_crashes,
        exclude_dates=exclude_dates,
    )
