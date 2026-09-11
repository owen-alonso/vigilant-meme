"""Price-cache quality checks. Mixed adjusted/raw weekly files fake reversal IC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.config import DataConfig


MIX_AUTOCORR_FAIL = -0.2
MIX_ABS_LOG = 0.25


def close_return_diagnostics(close: np.ndarray) -> dict[str, float]:
    """Lag-1 log-return autocorr and jump counts on a close series."""
    px = np.asarray(close, dtype=np.float64)
    px = px[np.isfinite(px) & (px > 0)]
    if px.size < 3:
        return {
            "n": float(px.size),
            "lag1_autocorr": float("nan"),
            "n_big_jumps": 0.0,
            "max_abs_log_return": 0.0,
        }
    log_r = np.diff(np.log(px))
    log_r = log_r[np.isfinite(log_r)]
    n_big = float(np.sum(np.abs(log_r) > MIX_ABS_LOG))
    max_abs = float(np.max(np.abs(log_r))) if log_r.size else 0.0
    autocorr = float("nan")
    if log_r.size >= 3:
        a = log_r[:-1]
        b = log_r[1:]
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if denom > 1e-12:
            autocorr = float((a * b).sum() / denom)
    return {
        "n": float(log_r.size),
        "lag1_autocorr": autocorr,
        "n_big_jumps": n_big,
        "max_abs_log_return": max_abs,
    }


def parquet_close_diagnostics(path: str | Path) -> dict[str, Any]:
    bars = pd.read_parquet(path)
    close_col = "close" if "close" in bars.columns else "Close"
    time_col = "datetime" if "datetime" in bars.columns else "Date"
    stats = close_return_diagnostics(bars[close_col].to_numpy(dtype=np.float64))
    stats["path"] = str(path)
    stats["first"] = str(bars[time_col].iloc[0]) if time_col in bars.columns else ""
    stats["last"] = str(bars[time_col].iloc[-1]) if time_col in bars.columns else ""
    return stats


def looks_like_mixed_scale(stats: dict[str, Any]) -> bool:
    """True when lag-1 autocorr is the mixed adjusted/raw weekly signature."""
    rho = stats.get("lag1_autocorr", float("nan"))
    return bool(np.isfinite(rho) and float(rho) < MIX_AUTOCORR_FAIL)


def assert_calendar_price_quality(
    path: str | Path,
    cfg: DataConfig,
    *,
    log_fn: Any | None = None,
) -> dict[str, Any]:
    """Raise on weekly mix-toggle caches. Daily/intraday are logged only."""
    stats = parquet_close_diagnostics(path)
    mixed = looks_like_mixed_scale(stats)
    if log_fn:
        log_fn(
            f"price quality {Path(path).name}: lag1_autocorr="
            f"{stats['lag1_autocorr']:+.3f} big_jumps={int(stats['n_big_jumps'])} "
            f"max_|r|={stats['max_abs_log_return']:.3f}"
        )
    if cfg.interval == "weekly" and mixed and not cfg.allow_mixed_prices:
        raise ValueError(
            f"{path} looks like mixed split-adjusted and raw weekly closes "
            f"(lag-1 autocorr={stats['lag1_autocorr']:+.3f} < {MIX_AUTOCORR_FAIL}). "
            "Delete the parquet and re-download with "
            "`python -m forecast.download --source yahoo --interval weekly "
            "--replace --symbols AAPL,MSFT`. Do not upsert onto the mixed file."
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    """Print close-return diagnostics for every parquet in a data dir."""
    import argparse
    import sys

    from forecast.data import discover_symbol_files

    p = argparse.ArgumentParser(description="Mix-jump / scale-toggle diagnostics for price caches.")
    p.add_argument("--data-dir", default=DataConfig().data_dir)
    p.add_argument("--interval", default="weekly")
    p.add_argument(
        "--delete-mixed",
        action="store_true",
        help="delete weekly parquets that fail the lag-1 mix-toggle gate",
    )
    args = p.parse_args(argv)
    cfg = DataConfig(interval=args.interval, allow_mixed_prices=True)
    try:
        files = discover_symbol_files(args.data_dir, interval=args.interval)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    mixed_any = False
    for path in files:
        stats = parquet_close_diagnostics(path)
        mixed = looks_like_mixed_scale(stats)
        mixed_any = mixed_any or mixed
        flag = "MIXED" if mixed else "ok"
        print(
            f"{flag:5} {Path(path).name}  lag1={stats['lag1_autocorr']:+.3f}  "
            f"jumps={int(stats['n_big_jumps'])}  max_|r|={stats['max_abs_log_return']:.3f}  "
            f"{stats['first']} -> {stats['last']}"
        )
        if cfg.interval == "weekly" and mixed:
            print(
                "      fail: weekly lag-1 autocorr < -0.2 is the mixed adjusted/raw signature",
                file=sys.stderr,
            )
            if args.delete_mixed:
                Path(path).unlink()
                print(f"      deleted {path}", file=sys.stderr)
    return 1 if mixed_any and str(args.interval) == "weekly" and not args.delete_mixed else 0


if __name__ == "__main__":
    raise SystemExit(main())
