"""Report bar density and vendor mix per chronological split.

The forecaster splits by date, so a change in data provenance across those dates
shows up as train/test distribution shift rather than as a modelling problem.
Run this before trusting a test-set number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast.config import DataConfig
from forecast.data import build_session_grid, load_bars


def main(path: str = "data/SPAB_clean_1min.parquet") -> None:
    cfg = DataConfig()
    raw = load_bars(path)
    grid = build_session_grid(raw, cfg)
    sessions = grid["session"].drop_duplicates().sort_values().to_numpy()
    n = len(sessions)
    n_test = int(round(n * cfg.test_fraction))
    n_val = int(round(n * cfg.val_fraction))
    train_end, val_end = sessions[n - n_val - n_test], sessions[n - n_test]

    source = pd.read_parquet(path, columns=["datetime", "source"]) if _has_source(path) else None

    print(f"{path}: {n} sessions, split at {pd.Timestamp(train_end).date()} / "
          f"{pd.Timestamp(val_end).date()}\n")
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


def _has_source(path: str) -> bool:
    return "source" in pd.read_parquet(path).columns


if __name__ == "__main__":
    main(*sys.argv[1:])
