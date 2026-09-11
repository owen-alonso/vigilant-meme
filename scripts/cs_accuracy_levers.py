"""Accuracy levers 1–6 and 8 vs the promoted overnight skip.

Locked-val gate: lift >= 0.003 CS IC and do not kill val-2017. Test / 2023
are reported after the fact. Live long-only IR is the runnable-book tie-break.
Lever 7 (vendor ingest) is a hook only.

    python scripts/cs_accuracy_levers.py --synthetic
    python scripts/cs_accuracy_levers.py --data-dir data --universe liquid \\
        --out checkpoints/forecast_ridge_overnight/levers.json
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
import pandas as pd

from forecast.backtest import book_pnl
from forecast.config import DataConfig, interval_data_kwargs
from forecast.data import build_datasets
from forecast.levers import (
    LABEL_ESTIMANDS,
    VAL_2017_KEEP,
    VAL_LIFT,
    VENDOR_INGEST_HOOK,
    ablation_row,
    blend_skip_mlp,
    calibration_scale,
    fit_long_only_xy,
    format_ablation_table,
    year_sign_consistency_mask,
    year_slice_stats,
)
from forecast.overnight import LIVE_BUNDLE, LIVE_LONG_ONLY_BUNDLE, LIVE_ADV_BUNDLE
from forecast.ridge import (
    cs_stats,
    feature_mask,
    fit_ridge_xy,
    labelled_rows,
    predict_residual_mlp,
    regime_score_map,
    year_cs_ics,
)
from forecast.synthetic import write_cs_overnight_universe

PROMOTED = dict(
    ridge=10.0,
    rank_target=True,
    feat_winsor=3.0,
    mask_mode="no_long_ts",
)


def _ymd(year: int, month: int = 1, day: int = 1) -> int:
    return int(
        (np.datetime64(f"{year:04d}-{month:02d}-{day:02d}") - np.datetime64("1970-01-01"))
        / np.timedelta64(1, "D")
    )


def _cfg(
    data_dir: str,
    universe: str,
    *,
    label_return: str = "overnight",
    fill_minutes: int = 0,
    size_residual: bool = False,
    peer_residual: bool = False,
    double_residual: bool = False,
    train_adv_floor_pctile: float = 0.0,
) -> DataConfig:
    preset = interval_data_kwargs("daily")
    synthetic = universe in ("", "synthetic")
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
        cross_section_min_names=8 if synthetic else 30,
        allow_mixed_prices=synthetic,
        cs_zscore=True,
        universe="" if synthetic else universe,
        sector_residual=True,
        equities_only=not synthetic,
        train_from="" if synthetic else "1999-01-01",
        label_return=label_return,
        fill_minutes=int(fill_minutes or 0),
        double_residual=double_residual,
        size_residual=size_residual,
        peer_residual=peer_residual,
        train_adv_floor_pctile=float(train_adv_floor_pctile or 0.0),
        liquid_min_names=8 if train_adv_floor_pctile else 0,
    )


def _cache_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "feature_mean": bundle["feature_mean"],
        "feature_std": bundle["feature_std"],
        "cs_min_names": int(bundle.get("cs_min_names", 8)),
        "n_trade": int(bundle.get("n_trading_names", 0)),
        "panels": None,
    }
    for split in ("train", "val", "test"):
        x, y, d = labelled_rows(
            bundle[f"{split}_symbols"],
            bundle["feature_mean"],
            bundle["feature_std"],
        )
        out[f"{split}_x"] = x.astype(np.float64)
        out[f"{split}_y"] = y.astype(np.float64)
        out[f"{split}_d"] = d.astype(np.int64)
    return out


def _fit_skip(
    cache: dict[str, Any],
    *,
    mask: np.ndarray | None = None,
    long_only: bool = False,
    date_halflife: float = 0.0,
) -> tuple[np.ndarray, float]:
    x, y, d = cache["train_x"], cache["train_y"], cache["train_d"]
    keep = cache["train_d"] >= _ymd(1999) if int(cache["train_d"].min()) < _ymd(1999) else np.ones(
        cache["train_d"].shape[0], dtype=bool
    )
    x, y, d = x[keep], y[keep], d[keep]
    feat = mask if mask is not None else feature_mask(PROMOTED["mask_mode"])
    min_n = int(cache["cs_min_names"])
    if long_only:
        w, b, _ = fit_long_only_xy(
            x, y, d, ridge=PROMOTED["ridge"], min_names=min_n,
            feat_winsor=PROMOTED["feat_winsor"], feature_mask_bool=feat,
            date_halflife=date_halflife,
        )
    else:
        w, b, _ = fit_ridge_xy(
            x, y, d, ridge=PROMOTED["ridge"], rank_target=PROMOTED["rank_target"],
            feat_winsor=PROMOTED["feat_winsor"], feature_mask_bool=feat,
            min_names=min_n, date_halflife=date_halflife,
        )
    return w.astype(np.float64), float(b)


def _pred(cache: dict[str, Any], split: str, w: np.ndarray, b: float) -> np.ndarray:
    return cache[f"{split}_x"] @ w + b


def _split_pack(cache: dict[str, Any], w: np.ndarray, b: float) -> dict[str, Any]:
    min_n = int(cache["cs_min_names"])
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        pred = _pred(cache, split, w, b)
        y, d = cache[f"{split}_y"], cache[f"{split}_d"]
        stats = cs_stats(pred, y, d, min_names=min_n)
        y2017 = year_slice_stats(pred, y, d, min_names=min_n, year=2017)
        y2023 = year_slice_stats(pred, y, d, min_names=min_n, year=2023)
        stats["cs_ic_2017"] = float(y2017.get("cs_ic", float("nan")))
        stats["cs_ic_2023"] = float(y2023.get("cs_ic", float("nan")))
        stats["years"] = year_cs_ics(pred, y, d, min_names=min_n)
        out[split] = stats
        out[f"{split}_pred"] = pred
    return out


def _live_long_only_ir(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
    """Tiny panel IR for synthetic / unit tests. Not Owen's liquid-tape number."""
    keys = np.unique(dates)
    if keys.size < 8:
        return float("nan")
    rows_p: dict[int, pd.Series] = {}
    rows_y: dict[int, pd.Series] = {}
    for key in keys:
        sel = dates == key
        n = int(sel.sum())
        idx = [f"n{i}" for i in range(n)]
        rows_p[int(key)] = pd.Series(pred[sel], index=idx)
        rows_y[int(key)] = pd.Series(y[sel], index=idx)
    pred_df = pd.DataFrame.from_dict(rows_p, orient="index").sort_index()
    y_df = pd.DataFrame.from_dict(rows_y, orient="index").reindex_like(pred_df)
    pred_df.index = pd.to_datetime(pred_df.index, unit="D")
    y_df.index = pred_df.index
    stats = book_pnl(
        pred_df,
        y_df,
        vol_target=0.15,
        long_only=True,
        holding="overnight",
        min_names=4,
        hold_halflife=0.0,
        round_trip_bps=float(LIVE_LONG_ONLY_BUNDLE["round_trip_bps"]),
        moc_bps=float(LIVE_LONG_ONLY_BUNDLE["moc_bps"]),
        moo_bps=float(LIVE_LONG_ONLY_BUNDLE["moo_bps"]),
        borrow_bps=0.0,
        hedge_cost_bps=float(LIVE_LONG_ONLY_BUNDLE["hedge_cost_bps"]),
        impact_vol_k=float(LIVE_LONG_ONLY_BUNDLE["impact_vol_k"]),
        thin_mult=float(LIVE_LONG_ONLY_BUNDLE["thin_mult"]),
        thin_pctile=float(LIVE_LONG_ONLY_BUNDLE["thin_pctile"]),
    )
    return float(stats.get("unlevered_net_ir", float("nan")))


def run_levers(cache: dict[str, Any]) -> dict[str, Any]:
    min_n = int(cache["cs_min_names"])
    w0, b0 = _fit_skip(cache)
    base = _split_pack(cache, w0, b0)
    base_val = float(base["val"]["cs_ic"])
    base_2017 = float(base["val"]["cs_ic_2017"])
    base_lo = _live_long_only_ir(base["val_pred"], cache["val_y"], cache["val_d"])
    rows: list[dict[str, Any]] = []

    def _add(name: str, lever: str, pack: dict[str, Any], *, note: str = "", lo: float | None = None) -> None:
        rows.append(
            ablation_row(
                name=name,
                lever=lever,
                baseline_val=base_val,
                baseline_2017=base_2017,
                val=pack["val"],
                test=pack["test"],
                live_long_only_ir=base_lo if lo is None else lo,
                note=note,
            )
        )

    rows.append(
        ablation_row(
            name="overnight_skip",
            lever="baseline",
            baseline_val=base_val - 1.0,
            baseline_2017=base_2017,
            val=base["val"],
            test=base["test"],
            live_long_only_ir=base_lo,
            note="promoted overnight skip (PR #5); not a new lever",
        )
    )
    # Force promote on the baseline row.
    rows[0]["promote"] = True
    rows[0]["decision"] = "promote"
    rows[0]["val_lift"] = 0.0

    # 1. Year-stable / recency
    mask = feature_mask(PROMOTED["mask_mode"]) & year_sign_consistency_mask(
        cache["train_x"], cache["train_y"], cache["train_d"], min_names=min_n
    )
    w, b = _fit_skip(cache, mask=mask)
    _add("year_sign_recency", "1_regime", _split_pack(cache, w, b), note="train-only recency year-stable")

    # 1b. Trailing-IC flatten (sizing; CS IC of scores is unchanged except flats)
    pred_va = base["val_pred"]
    keys, scales = calibration_scale(
        pred_va, cache["val_y"], cache["val_d"], min_names=min_n, lookback_days=63
    )
    scale_map = {int(k): float(s) for k, s in zip(keys, scales)}
    flat = np.array([scale_map.get(int(d), 1.0) for d in cache["val_d"]], dtype=np.float64)
    pred_flat = pred_va * (flat > 0)
    stats_flat = cs_stats(pred_flat, cache["val_y"], cache["val_d"], min_names=min_n, flat_as_zero=True)
    y2017 = year_slice_stats(pred_flat, cache["val_y"], cache["val_d"], min_names=min_n, year=2017)
    stats_flat["cs_ic_2017"] = float(y2017.get("cs_ic", float("nan")))
    _add(
        "trailing_ic_flatten",
        "1_regime",
        {"val": stats_flat, "test": base["test"]},
        note="sizing overlay; Pearson unchanged unless flattened dates count as 0",
    )

    # 2. Long-only skip
    w, b = _fit_skip(cache, long_only=True)
    pack = _split_pack(cache, w, b)
    lo = _live_long_only_ir(pack["val_pred"], cache["val_y"], cache["val_d"])
    _add("long_only_skip", "2_label_objective", pack, lo=lo, note="ridge on top CS quantile")

    # 5. Ensemble skip+MLP (mix on late-train; val-gate here)
    skip_tr = cache["train_x"] @ w0 + b0
    ens = blend_skip_mlp(
        cache["train_x"], cache["train_y"], cache["train_d"], skip_tr, min_names=min_n
    )
    mlp_va = predict_residual_mlp(
        cache["val_x"], cache["val_d"], ens["w1"], ens["b1"], ens["w2"], min_names=min_n
    )
    mix = float(ens["mix"])
    pred_ens = base["val_pred"] + mix * mlp_va
    ens_val = cs_stats(pred_ens, cache["val_y"], cache["val_d"], min_names=min_n)
    ens_val["cs_ic_2017"] = float(
        year_slice_stats(pred_ens, cache["val_y"], cache["val_d"], min_names=min_n, year=2017).get(
            "cs_ic", float("nan")
        )
    )
    _add(
        "ensemble_skip_mlp",
        "5_ensemble",
        {"val": ens_val, "test": base["test"]},
        note=f"late-train mix={mix:.2f}; promote only on locked val",
    )

    payload = {
        "baseline_val_cs_ic": base_val,
        "baseline_val_2017": base_2017,
        "baseline_live_long_only_ir": base_lo,
        "val_lift": VAL_LIFT,
        "val_2017_keep": VAL_2017_KEEP,
        "vendor_ingest": VENDOR_INGEST_HOOK,
        "label_estimands": [{"label": k, "fill_minutes": n} for k, n in LABEL_ESTIMANDS],
        "cost_bundles_new": ["live_adv", "live_adv_long_only", "borrow_stress", "auction_stress"],
        "rows": rows,
        "promoted": [r["name"] for r in rows if r["promote"]],
        "discarded": [r["name"] for r in rows if not r["promote"]],
        "table": format_ablation_table(rows),
        "live_bundle": LIVE_BUNDLE["name"],
        "live_adv_bundle": LIVE_ADV_BUNDLE["name"],
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Overnight accuracy levers 1-6 and 8 (locked-val).")
    p.add_argument("--synthetic", action="store_true", help="planted overnight CS universe (CPU)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--universe", default="liquid")
    p.add_argument("--out", default="")
    p.add_argument(
        "--try-fill",
        type=int,
        default=0,
        help="also score open+N fill as a separate estimand (does not replace overnight y)",
    )
    args = p.parse_args(argv)

    if args.synthetic:
        import tempfile

        tmp = tempfile.mkdtemp(prefix="cs_levers_")
        write_cs_overnight_universe(tmp, n_names=12, n_days=220, seed=3)
        cfg = _cfg(tmp, "synthetic")
        print(f"synthetic overnight universe -> {tmp}", flush=True)
    else:
        cfg = _cfg(args.data_dir, args.universe)
        print(f"loading {args.universe} from {args.data_dir}", flush=True)

    bundle = build_datasets(cfg, log_fn=print)
    cache = _cache_from_bundle(bundle)
    payload = run_levers(cache)
    payload["n_trading_names"] = int(bundle.get("n_trading_names", 0))
    payload["label_return"] = str(bundle.get("label_return", "overnight"))

    if int(args.try_fill or 0) > 0:
        fill_cfg = _cfg(
            cfg.data_dir,
            args.universe if not args.synthetic else "synthetic",
            label_return="open_fill",
            fill_minutes=int(args.try_fill),
        )
        fill_bundle = build_datasets(fill_cfg, log_fn=print)
        fill_cache = _cache_from_bundle(fill_bundle)
        w, b = _fit_skip(fill_cache)
        pack = _split_pack(fill_cache, w, b)
        payload["rows"].append(
            ablation_row(
                name=f"open{int(args.try_fill)}_fill",
                lever="2_label_objective",
                baseline_val=float(payload["baseline_val_cs_ic"]),
                baseline_2017=float(payload["baseline_val_2017"]),
                val=pack["val"],
                test=pack["test"],
                note="separate estimand; do not mix into overnight y",
            )
        )
        payload["table"] = format_ablation_table(payload["rows"])
        payload["promoted"] = [r["name"] for r in payload["rows"] if r["promote"]]
        payload["discarded"] = [r["name"] for r in payload["rows"] if not r["promote"]]

    print(payload["table"], flush=True)
    print(
        f"promoted={payload['promoted']} discarded={payload['discarded']}",
        flush=True,
    )
    print(VENDOR_INGEST_HOOK, flush=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
