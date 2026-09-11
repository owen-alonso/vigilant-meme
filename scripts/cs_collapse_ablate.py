"""Sweep frozen vs walk-forward CS ridge on the locked val/test window.

Builds last-bar rows once (slow), then ridge variants are seconds.

    python scripts/cs_collapse_ablate.py --data-dir /tmp/liquid_daily --universe liquid
    python scripts/cs_collapse_ablate.py --cache /tmp/cs_lastbars.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from forecast.config import DataConfig, interval_data_kwargs
from forecast.data import build_datasets
from forecast.ridge import (
    cs_stats,
    feature_mask,
    fit_ridge_xy,
    labelled_rows,
    walk_forward_predict,
)


def _cfg(data_dir: str, universe: str, train_from: str) -> DataConfig:
    preset = interval_data_kwargs("daily")
    return DataConfig(
        data_dir=data_dir,
        interval="daily",
        horizon=1,
        seq_len=32,
        stride=1,
        min_context=8,
        warmup_bars=16,
        vol_halflife=preset["vol_halflife"],
        z_window=preset["z_window"],
        z_min_periods=preset["z_min_periods"],
        max_abs_log_return=preset["max_abs_log_return"],
        eval_last_bar=True,
        global_calendar_split=True,
        residual_target=True,
        cross_section_min_names=30,
        allow_mixed_prices=True,
        cs_zscore=True,
        universe=universe,
        sector_residual=True,
        equities_only=True,
        train_from=train_from,
    )


def _dump(bundle: dict[str, Any], path: Path) -> None:
    payload: dict[str, Any] = {
        "feature_mean": bundle["feature_mean"],
        "feature_std": bundle["feature_std"],
        "cs_min_names": np.array([bundle.get("cs_min_names", 30)]),
    }
    for split in ("train", "val", "test"):
        x, y, d = labelled_rows(
            bundle[f"{split}_symbols"],
            bundle["feature_mean"],
            bundle["feature_std"],
        )
        payload[f"{split}_x"] = x.astype(np.float32)
        payload[f"{split}_y"] = y.astype(np.float32)
        payload[f"{split}_d"] = d.astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load(path: Path) -> dict[str, Any]:
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out["cs_min_names"] = int(out["cs_min_names"][0])
    return out


def _eval(pred: np.ndarray, y: np.ndarray, d: np.ndarray, min_names: int) -> dict[str, float]:
    sel = np.isfinite(pred)
    return cs_stats(pred[sel], y[sel], d[sel], min_names=min_names)


def _frozen(
    cache: dict[str, Any],
    *,
    train_from_days: int | None,
    ridge: float,
    rank_target: bool,
    cs_zscore: bool,
    mask_mode: str,
    date_halflife: float,
) -> dict[str, Any]:
    min_names = int(cache["cs_min_names"])
    x_tr = cache["train_x"].astype(np.float64)
    y_tr = cache["train_y"].astype(np.float64)
    d_tr = cache["train_d"].astype(np.int64)
    if train_from_days is not None:
        keep = d_tr >= int(train_from_days)
        x_tr, y_tr, d_tr = x_tr[keep], y_tr[keep], d_tr[keep]
    w, b, train_ic = fit_ridge_xy(
        x_tr,
        y_tr,
        d_tr,
        ridge=ridge,
        min_names=min_names,
        cs_demean=True,
        cs_zscore=cs_zscore,
        rank_target=rank_target,
        feature_mask_bool=feature_mask(mask_mode),
        date_halflife=date_halflife,
    )
    row: dict[str, Any] = {"train_is_cs_ic": train_ic, "n_train": int(y_tr.size)}
    for split in ("val", "test"):
        x = cache[f"{split}_x"].astype(np.float64)
        y = cache[f"{split}_y"].astype(np.float64)
        d = cache[f"{split}_d"].astype(np.int64)
        pred = x @ w.astype(np.float64) + b
        stats = cs_stats(pred, y, d, min_names=min_names)
        row[split] = stats
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CS train→test collapse ablations.")
    p.add_argument("--data-dir", default="")
    p.add_argument("--universe", default="liquid", choices=("", "liquid"))
    p.add_argument("--cache", default="/tmp/cs_lastbars.npz")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--train-from", default="", help="used only when building the cache")
    p.add_argument("--out", default="/tmp/cs_collapse_ablate.json")
    args = p.parse_args(argv)
    cache_path = Path(args.cache)
    if args.rebuild or not cache_path.exists():
        if not args.data_dir:
            print("need --data-dir to build the last-bar cache", file=sys.stderr)
            return 2
        print(f"building last bars from {args.data_dir} ...", flush=True)
        bundle = build_datasets(
            _cfg(args.data_dir, args.universe, args.train_from),
            log_fn=print,
        )
        _dump(bundle, cache_path)
        print(f"wrote {cache_path}", flush=True)
    cache = _load(cache_path)
    min_d = int(cache["train_d"].min()) if cache["train_d"].size else 0
    # days since epoch helpers
    def ymd(year: int, month: int = 1, day: int = 1) -> int:
        return int((np.datetime64(f"{year:04d}-{month:02d}-{day:02d}") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))

    rows: list[dict[str, Any]] = []
    frozen_cases = [
        ("frozen_1999", dict(train_from_days=ymd(1999))),
        ("frozen_2004", dict(train_from_days=ymd(2004))),
        ("frozen_2007", dict(train_from_days=ymd(2007))),
        ("frozen_all_train", dict(train_from_days=None if min_d < ymd(1999) else min_d)),
        ("frozen_rank", dict(train_from_days=ymd(1999), rank_target=True)),
        ("frozen_cs_z", dict(train_from_days=ymd(1999), cs_zscore=True)),
        ("frozen_cs_feats", dict(train_from_days=ymd(1999), mask_mode="cs")),
        ("frozen_no_cal", dict(train_from_days=ymd(1999), mask_mode="no_calendar")),
        ("frozen_lam10", dict(train_from_days=ymd(1999), ridge=10.0)),
        ("frozen_lam100", dict(train_from_days=ymd(1999), ridge=100.0)),
        ("frozen_hl504", dict(train_from_days=ymd(1999), date_halflife=504.0)),
        ("frozen_hl1260", dict(train_from_days=ymd(1999), date_halflife=1260.0)),
        ("frozen_2004_rank_cs", dict(train_from_days=ymd(2004), rank_target=True, mask_mode="cs")),
    ]
    base = dict(ridge=1.0, rank_target=False, cs_zscore=False, mask_mode="all", date_halflife=0.0)
    print(f"{'case':<28} {'val':>8} {'val_t':>7} {'test':>8} {'test_t':>7} {'ntr':>8}")
    for name, kw in frozen_cases:
        cfg = {**base, **kw}
        stats = _frozen(cache, **cfg)
        stats["name"] = name
        stats["kind"] = "frozen"
        rows.append(stats)
        val, test = stats["val"], stats["test"]
        print(
            f"{name:<28} {val['cs_ic']:+8.4f} {val['cs_ic_tstat']:7.2f} "
            f"{test['cs_ic']:+8.4f} {test['cs_ic_tstat']:7.2f} {stats['n_train']:8d}",
            flush=True,
        )

    x = np.concatenate([cache["train_x"], cache["val_x"], cache["test_x"]], axis=0).astype(np.float64)
    y = np.concatenate([cache["train_y"], cache["val_y"], cache["test_y"]], axis=0).astype(np.float64)
    d = np.concatenate([cache["train_d"], cache["val_d"], cache["test_d"]], axis=0).astype(np.int64)
    val_dates = np.unique(cache["val_d"])
    test_dates = np.unique(cache["test_d"])
    score_dates = np.unique(np.concatenate([val_dates, test_dates]))
    min_names = int(cache["cs_min_names"])
    wf_cases = [
        ("wf_expanding", None, "all", False),
        ("wf_roll_2y", 504, "all", False),
        ("wf_roll_5y", 1260, "all", False),
        ("wf_roll_8y", 2016, "all", False),
        ("wf_roll_5y_cs", 1260, "cs", False),
        ("wf_roll_5y_rank", 1260, "all", True),
        ("wf_expanding_rank", None, "all", True),
    ]
    for name, lookback, mask_mode, rank_target in wf_cases:
        pred = walk_forward_predict(
            x,
            y,
            d,
            score_dates=score_dates,
            lookback_days=lookback,
            ridge=1.0,
            min_names=min_names,
            cs_demean=True,
            rank_target=rank_target,
            feature_mask_bool=feature_mask(mask_mode),
            min_train_dates=60,
        )
        val_sel = np.isin(d, val_dates)
        test_sel = np.isin(d, test_dates)
        stats = {
            "name": name,
            "kind": "walk_forward",
            "lookback_days": lookback,
            "val": _eval(pred[val_sel], y[val_sel], d[val_sel], min_names),
            "test": _eval(pred[test_sel], y[test_sel], d[test_sel], min_names),
        }
        rows.append(stats)
        val, test = stats["val"], stats["test"]
        print(
            f"{name:<28} {val['cs_ic']:+8.4f} {val['cs_ic_tstat']:7.2f} "
            f"{test['cs_ic']:+8.4f} {test['cs_ic_tstat']:7.2f} {'wf':>8}",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
