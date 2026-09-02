"""Report bar density and vendor mix per chronological split.

The forecaster splits by date, so a change in data provenance across those dates
shows up as train/test distribution shift rather than as a modelling problem.
Run this before trusting a test-set number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from forecast.config import DataConfig
from forecast.data import build_session_grid, load_bars, split_session_bounds
from mamba_lm.paths import resolve_path


def main(path: str) -> None:
    cfg = DataConfig()
    resolved = resolve_path(path)
    if not resolved.exists():
        raise SystemExit(
            f"data file not found: {resolved}\n"
            "Pass a parquet path, e.g. scripts/split_report.py data/YOUR_SYMBOL_clean_1min.parquet"
        )
    raw = load_bars(resolved)
    grid = build_session_grid(raw, cfg)
    sessions = grid["session"].drop_duplicates().sort_values().to_numpy()
    n = len(sessions)
    cut_train, cut_val = split_session_bounds(n, cfg)
    train_end, val_end = sessions[cut_train], sessions[cut_val]

    source = (
        pd.read_parquet(resolved, columns=["datetime", "source"])
        if _has_source(resolved)
        else None
    )

    print(
        f"{resolved}: {n} sessions, split at {pd.Timestamp(train_end).date()} / "
        f"{pd.Timestamp(val_end).date()}\n"
    )
    bounds = [
        ("train", grid["session"] < train_end),
        ("val", (grid["session"] >= train_end) & (grid["session"] < val_end)),
        ("test", grid["session"] >= val_end),
    ]
    for name, sel in bounds:
        part = grid[sel]
        days = part["session"].nunique()
        traded = float(part["traded"].sum())
        line = (
            f"{name:5s} sessions={days:5d} traded_bars={int(traded):7d} "
            f"per_day={traded / max(1, days):6.1f} density={traded / max(1, len(part)):6.1%}"
        )
        if source is not None:
            lo = part["datetime"].min()
            hi = part["datetime"].max()
            mix = source[(source["datetime"] >= lo) & (source["datetime"] <= hi)]
            counts = mix["source"].value_counts(normalize=True)
            line += "  " + ", ".join(f"{k}={v:.0%}" for k, v in counts.items())
        print(line)


def _has_source(path) -> bool:
    return "source" in pd.read_parquet(path).columns


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report bar density per train/val/test split.")
    p.add_argument(
        "path",
        nargs="?",
        default="data/SPAB_clean_1min.parquet",
        help="parquet file to inspect (default: data/SPAB_clean_1min.parquet)",
    )
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    try:
        main(args.path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
