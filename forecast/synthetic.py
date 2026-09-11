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
