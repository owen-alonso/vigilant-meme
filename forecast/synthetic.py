"""Synthetic daily panels with a planted cross-sectional residual signal.

Used by ablations and tests. Not market data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_cs_momentum_universe(
    data_dir: str | Path,
    *,
    n_names: int = 12,
    n_days: int = 160,
    seed: int = 0,
    rho: float = 0.35,
    include_spy: bool = True,
) -> list[str]:
    """Write daily parquets where next-day residual ranks follow today's CS residual.

    ``idio[t] = rho * cs_z(idio[t-1]) + noise``, plus a shared market factor.
    SPY is approximately the market (beta ~ 1, small residual).
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    mkt = rng.normal(scale=0.006, size=n_days)
    names = [f"S{i:02d}" for i in range(n_names)]
    idio = np.zeros((n_days, n_names), dtype=np.float64)
    noise = rng.normal(scale=0.008, size=(n_days, n_names))
    for t in range(1, n_days):
        prev = idio[t - 1]
        z = (prev - prev.mean()) / max(float(prev.std()), 1e-8)
        idio[t] = np.clip(rho * 0.008 * z + noise[t], -0.04, 0.04)
    betas = 0.7 + 0.6 * rng.random(n_names)

    def _write(symbol: str, close: np.ndarray) -> None:
        pd.DataFrame(
            {
                "datetime": dates,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(n_days, 1_000_000.0),
                "source": "synthetic",
                "interval": "daily",
            }
        ).to_parquet(root / f"{symbol}_daily.parquet")

    for j, name in enumerate(names):
        r = np.clip(betas[j] * mkt + idio[:, j], -0.06, 0.06)
        close = 50.0 * (1.0 + j) * np.exp(np.cumsum(r))
        _write(name, close)
    if include_spy:
        spy_r = np.clip(mkt + rng.normal(scale=0.001, size=n_days), -0.06, 0.06)
        _write("SPY", 200.0 * np.exp(np.cumsum(spy_r)))
    return names


def write_cs_overnight_universe(
    data_dir: str | Path,
    *,
    n_names: int = 12,
    n_days: int = 200,
    seed: int = 1,
    rho: float = 0.6,
    include_spy: bool = True,
) -> list[str]:
    """Write daily parquets whose *overnight gap* residual ranks are planted.

    ``open[t+1] / close[t]`` carries CS-momentum in the idiosyncratic gap.
    The next session ``close[t+1] / open[t+1]`` is independent noise, so a
    close-to-close skip should not recover the same object.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    mkt_gap = rng.normal(scale=0.004, size=n_days)
    mkt_sess = rng.normal(scale=0.006, size=n_days)
    names = [f"S{i:02d}" for i in range(n_names)]
    idio_gap = np.zeros((n_days, n_names), dtype=np.float64)
    noise_gap = rng.normal(scale=0.006, size=(n_days, n_names))
    noise_sess = rng.normal(scale=0.008, size=(n_days, n_names))
    for t in range(1, n_days):
        prev = idio_gap[t - 1]
        z = (prev - prev.mean()) / max(float(prev.std()), 1e-8)
        idio_gap[t] = np.clip(rho * 0.006 * z + noise_gap[t], -0.03, 0.03)
    betas = 0.7 + 0.6 * rng.random(n_names)

    def _write_ohlc(symbol: str, open_px: np.ndarray, close: np.ndarray) -> None:
        high = np.maximum(open_px, close) * 1.001
        low = np.minimum(open_px, close) * 0.999
        pd.DataFrame(
            {
                "datetime": dates,
                "open": open_px,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n_days, 1_000_000.0),
                "source": "synthetic",
                "interval": "daily",
            }
        ).to_parquet(root / f"{symbol}_daily.parquet")

    for j, name in enumerate(names):
        gap = np.clip(betas[j] * mkt_gap + idio_gap[:, j], -0.05, 0.05)
        sess = np.clip(betas[j] * mkt_sess + noise_sess[:, j], -0.05, 0.05)
        close = np.empty(n_days, dtype=np.float64)
        open_px = np.empty(n_days, dtype=np.float64)
        close[0] = 50.0 * (1.0 + j)
        open_px[0] = close[0] * np.exp(-0.001)
        for t in range(1, n_days):
            open_px[t] = close[t - 1] * np.exp(gap[t])
            close[t] = open_px[t] * np.exp(sess[t])
        _write_ohlc(name, open_px, close)
    if include_spy:
        spy_close = np.empty(n_days, dtype=np.float64)
        spy_open = np.empty(n_days, dtype=np.float64)
        spy_close[0] = 200.0
        spy_open[0] = 200.0
        for t in range(1, n_days):
            spy_open[t] = spy_close[t - 1] * np.exp(mkt_gap[t] + rng.normal(scale=0.0005))
            spy_close[t] = spy_open[t] * np.exp(mkt_sess[t] + rng.normal(scale=0.0005))
        _write_ohlc("SPY", spy_open, spy_close)
    return names
