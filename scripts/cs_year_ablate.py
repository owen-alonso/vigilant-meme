"""Val-gated year-stability / residual / universe ablations.

Builds last-bar rows once per residual/universe setting, then ridge variants
are seconds. Pick on locked val only; test is reported after the fact.

    python scripts/cs_year_ablate.py --data-dir /tmp/liquid_daily --universe liquid
    python scripts/cs_year_ablate.py --cache /tmp/cs_lastbars44.npz
    python scripts/cs_year_ablate.py --data-dir /tmp/liquid_daily --double-residual \\
        --cache /tmp/cs_lastbars_double.npz
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
from forecast.data import FEATURE_NAMES, build_datasets
from forecast.ridge import (
    cs_stats,
    feature_mask,
    fit_ridge_xy,
    labelled_rows,
    univariate_cs_ics,
    walk_forward_predict,
    year_cs_ics,
    year_feature_ics,
    year_stable_mask,
)

PROMOTED = dict(
    ridge=10.0,
    rank_target=True,
    feat_winsor=3.0,
    mask_mode="no_long_ts",
)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "ts_ret": ("ret_1", "ret_5", "ret_15", "ret_60", "ret_390"),
    "ohlc": ("range_hl", "body_co", "close_loc", "wick_up", "wick_dn"),
    "vol": ("vol_level", "vol_change", "volume_z", "turnover_z", "ret_vol"),
    "peer_idio": ("peer_ret_1", "mkt_ret_1", "idio_ret_1", "sector_ret_1", "idio_sector"),
    "cs_z": ("cs_rank_1", "cs_ret_1", "cs_ret_5", "cs_ret_15", "cs_ret_60", "cs_volume", "cs_vol"),
    "cs_products": (
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
    ),
    "calendar": ("traded", "staleness", "new_session", "tod_sin", "tod_cos", "tod_frac", "dow_frac"),
}


def _cfg(
    data_dir: str,
    universe: str,
    *,
    double_residual: bool,
    residualize_features: bool,
    industry_residual: bool,
) -> DataConfig:
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
        train_from="1999-01-01",
        double_residual=double_residual,
        residualize_features=residualize_features,
        industry_residual=industry_residual,
    )


def _dump(bundle: dict[str, Any], path: Path) -> None:
    payload: dict[str, Any] = {
        "feature_mean": bundle["feature_mean"],
        "feature_std": bundle["feature_std"],
        "cs_min_names": np.array([bundle.get("cs_min_names", 30)]),
        "n_trade": np.array([len(bundle.get("meta", []))]),
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
    if "n_trade" in out:
        out["n_trade"] = int(np.asarray(out["n_trade"]).reshape(-1)[0])
    return out


def _ymd(year: int, month: int = 1, day: int = 1) -> int:
    return int(
        (np.datetime64(f"{year:04d}-{month:02d}-{day:02d}") - np.datetime64("1970-01-01"))
        / np.timedelta64(1, "D")
    )


def _score(
    w: np.ndarray,
    b: float,
    cache: dict[str, Any],
    min_names: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        x = cache[f"{split}_x"].astype(np.float64)
        y = cache[f"{split}_y"].astype(np.float64)
        d = cache[f"{split}_d"].astype(np.int64)
        pred = x @ w.astype(np.float64) + b
        row[split] = cs_stats(pred, y, d, min_names=min_names)
        row[f"{split}_years"] = year_cs_ics(pred, y, d, min_names=min_names)
    return row


def _fit_promoted(
    cache: dict[str, Any],
    *,
    year_balance: bool = False,
    year_stable: str = "",
    mask_mode: str | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    min_names = int(cache["cs_min_names"])
    keep_train = cache["train_d"].astype(np.int64) >= _ymd(1999)
    x = cache["train_x"].astype(np.float64)[keep_train]
    y = cache["train_y"].astype(np.float64)[keep_train]
    d = cache["train_d"].astype(np.int64)[keep_train]
    mask = feature_mask(mask_mode or PROMOTED["mask_mode"])
    if x.shape[1] != mask.size:
        raise ValueError(
            f"cache has {x.shape[1]} features, expected {mask.size}. Rebuild the last-bar cache."
        )
    if year_stable in ("train", "train_val"):
        xv = yv = dv = None
        if year_stable == "train_val":
            xv = cache["val_x"].astype(np.float64)
            yv = cache["val_y"].astype(np.float64)
            dv = cache["val_d"].astype(np.int64)
        keep = year_stable_mask(x, y, d, min_names=min_names, x_val=xv, y_val=yv, d_val=dv)
        mask = mask & keep
    w, b, ic = fit_ridge_xy(
        x,
        y,
        d,
        ridge=PROMOTED["ridge"],
        min_names=min_names,
        cs_demean=True,
        rank_target=PROMOTED["rank_target"],
        feat_winsor=PROMOTED["feat_winsor"],
        feature_mask_bool=mask,
        year_balance=year_balance,
    )
    return w, b, ic, mask


def _group_stability(cache: dict[str, Any], min_names: int) -> list[dict[str, Any]]:
    keep = cache["train_d"].astype(np.int64) >= _ymd(1999)
    x_tr = cache["train_x"].astype(np.float64)[keep]
    y_tr = cache["train_y"].astype(np.float64)[keep]
    d_tr = cache["train_d"].astype(np.int64)[keep]
    x_va = cache["val_x"].astype(np.float64)
    y_va = cache["val_y"].astype(np.float64)
    d_va = cache["val_d"].astype(np.int64)
    by_year = year_feature_ics(x_tr, y_tr, d_tr, min_names=min_names)
    val_ics = univariate_cs_ics(x_va, y_va, d_va, min_names=min_names)
    train_ics = univariate_cs_ics(x_tr, y_tr, d_tr, min_names=min_names)
    names = list(FEATURE_NAMES)
    if x_tr.shape[1] != len(names):
        return []
    years = sorted(by_year)
    stacked = np.stack([by_year[y] for y in years], axis=0) if years else None
    rows: list[dict[str, Any]] = []
    for group, cols in FEATURE_GROUPS.items():
        idxs = [names.index(c) for c in cols if c in names]
        if not idxs:
            continue
        frac_stable = []
        for j in idxs:
            if stacked is None:
                frac_stable.append(float("nan"))
                continue
            col = stacked[:, j]
            finite = col[np.isfinite(col) & (np.abs(col) >= 0.003)]
            if finite.size < 2:
                frac_stable.append(float("nan"))
                continue
            pos = float((finite > 0).mean())
            frac_stable.append(max(pos, 1.0 - pos))
        val_agree = []
        for j in idxs:
            if np.isfinite(train_ics[j]) and np.isfinite(val_ics[j]) and abs(train_ics[j]) >= 0.003:
                val_agree.append(float(train_ics[j] * val_ics[j] > 0))
        rows.append(
            {
                "group": group,
                "n": len(idxs),
                "train_mean_ic": float(np.nanmean([train_ics[j] for j in idxs])),
                "val_mean_ic": float(np.nanmean([val_ics[j] for j in idxs])),
                "train_year_sign_stable": float(np.nanmean(frac_stable)) if frac_stable else float("nan"),
                "val_sign_agree": float(np.mean(val_agree)) if val_agree else float("nan"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Year / residual / universe CS ablations.")
    p.add_argument("--data-dir", default="")
    p.add_argument("--universe", default="liquid", choices=("", "liquid", "liquid_wide"))
    p.add_argument("--cache", default="/tmp/cs_lastbars44.npz")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--double-residual", action="store_true")
    p.add_argument("--residualize-features", action="store_true")
    p.add_argument("--industry-residual", action="store_true")
    p.add_argument("--out", default="/tmp/cs_year_ablate.json")
    args = p.parse_args(argv)

    cache_path = Path(args.cache)
    if args.rebuild or not cache_path.exists():
        if not args.data_dir:
            print("need --data-dir to build the last-bar cache", file=sys.stderr)
            return 2
        print(f"building last bars from {args.data_dir} ...", flush=True)
        bundle = build_datasets(
            _cfg(
                args.data_dir,
                args.universe,
                double_residual=args.double_residual,
                residualize_features=args.residualize_features,
                industry_residual=args.industry_residual,
            ),
            log_fn=print,
        )
        _dump(bundle, cache_path)
        print(f"wrote {cache_path}", flush=True)
    cache = _load(cache_path)
    min_names = int(cache["cs_min_names"])
    n_feat = int(cache["train_x"].shape[1])
    print(
        f"cache {cache_path} features={n_feat} min_names={min_names} "
        f"n_trade={cache.get('n_trade', '?')} "
        f"double={args.double_residual} feat_resid={args.residualize_features} "
        f"industry={args.industry_residual} universe={args.universe}",
        flush=True,
    )
    if n_feat != len(FEATURE_NAMES):
        print(
            f"refusing stale cache: {n_feat} != {len(FEATURE_NAMES)}. "
            "Pass --rebuild --data-dir ...",
            file=sys.stderr,
        )
        return 2

    groups = _group_stability(cache, min_names)
    print("feature-group stability (train years + val sign; no test):", flush=True)
    for g in groups:
        print(
            f"  {g['group']:<12} train_ic={g['train_mean_ic']:+.4f} "
            f"val_ic={g['val_mean_ic']:+.4f} "
            f"year_stable={g['train_year_sign_stable']:.2f} "
            f"val_agree={g['val_sign_agree']:.2f}",
            flush=True,
        )

    cases = [
        ("promoted_skip", dict()),
        ("year_balance", dict(year_balance=True)),
        ("year_stable_train", dict(year_stable="train")),
        ("year_stable_train_val", dict(year_stable="train_val")),
        ("year_balance_stable_train", dict(year_balance=True, year_stable="train")),
    ]
    rows: list[dict[str, Any]] = []
    print(f"{'case':<28} {'val':>8} {'val_t':>7} {'test':>8} {'test_t':>7} {'nkeep':>6}")
    baseline_val = None
    for name, kw in cases:
        w, b, ic, mask = _fit_promoted(cache, **kw)
        stats = _score(w, b, cache, min_names)
        stats.update(
            {
                "name": name,
                "kind": "frozen",
                "train_is_cs_ic": ic,
                "n_keep": int(mask.sum()),
                "year_balance": bool(kw.get("year_balance", False)),
                "year_stable": str(kw.get("year_stable", "")),
            }
        )
        if name == "promoted_skip":
            baseline_val = float(stats["val"]["cs_ic"])
        val, test = stats["val"], stats["test"]
        print(
            f"{name:<28} {val['cs_ic']:+8.4f} {val['cs_ic_tstat']:7.2f} "
            f"{test['cs_ic']:+8.4f} {test['cs_ic_tstat']:7.2f} {int(mask.sum()):6d}",
            flush=True,
        )
        rows.append(stats)

    # Expanding walk-forward is diagnostic unless it clearly wins val.
    x = np.concatenate([cache["train_x"], cache["val_x"], cache["test_x"]], axis=0).astype(np.float64)
    y = np.concatenate([cache["train_y"], cache["val_y"], cache["test_y"]], axis=0).astype(np.float64)
    d = np.concatenate([cache["train_d"], cache["val_d"], cache["test_d"]], axis=0).astype(np.int64)
    val_dates = np.unique(cache["val_d"])
    test_dates = np.unique(cache["test_d"])
    pred = walk_forward_predict(
        x,
        y,
        d,
        score_dates=np.unique(np.concatenate([val_dates, test_dates])),
        lookback_days=None,
        ridge=PROMOTED["ridge"],
        min_names=min_names,
        cs_demean=True,
        rank_target=PROMOTED["rank_target"],
        feat_winsor=PROMOTED["feat_winsor"],
        feature_mask_bool=feature_mask(PROMOTED["mask_mode"]),
        min_train_dates=60,
    )
    val_sel = np.isin(d, val_dates)
    test_sel = np.isin(d, test_dates)
    wf = {
        "name": "wf_expanding_promoted",
        "kind": "walk_forward",
        "val": cs_stats(pred[val_sel], y[val_sel], d[val_sel], min_names=min_names),
        "test": cs_stats(pred[test_sel], y[test_sel], d[test_sel], min_names=min_names),
    }
    print(
        f"{'wf_expanding_promoted':<28} {wf['val']['cs_ic']:+8.4f} {wf['val']['cs_ic_tstat']:7.2f} "
        f"{wf['test']['cs_ic']:+8.4f} {wf['test']['cs_ic_tstat']:7.2f}",
        flush=True,
    )
    rows.append(wf)

    verdicts: list[dict[str, Any]] = []
    for stats in rows:
        name = stats["name"]
        val_ic = float(stats["val"]["cs_ic"])
        if name == "promoted_skip":
            verdict = "baseline"
        elif name.startswith("wf_"):
            margin = 0.003
            verdict = (
                "promote_wf"
                if baseline_val is not None and val_ic > baseline_val + margin
                else "diagnostic_only"
            )
        else:
            verdict = (
                "promote"
                if baseline_val is not None and val_ic > float(baseline_val) + 0.002
                else "discard"
            )
        verdicts.append({"name": name, "val_cs_ic": val_ic, "verdict": verdict})
        print(f"  verdict {name}: {verdict} (val={val_ic:+.4f})", flush=True)

    payload = {
        "universe": args.universe,
        "double_residual": args.double_residual,
        "residualize_features": args.residualize_features,
        "industry_residual": args.industry_residual,
        "n_features": n_feat,
        "n_trade": cache.get("n_trade"),
        "feature_groups": groups,
        "cases": [
            {k: v for k, v in r.items() if not str(k).endswith("_years") or k in ("val_years",)}
            for r in rows
        ],
        "year_tables": {r["name"]: {"val": r.get("val_years"), "test": r.get("test_years")} for r in rows if "val_years" in r},
        "verdicts": verdicts,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
