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
from forecast.data import FEATURE_NAMES, build_datasets
from forecast.ridge import (
    augment_cs_products,
    cs_stats,
    feature_mask,
    fit_listnet_xy,
    fit_ranknet_xy,
    fit_regime_ridge,
    fit_residual_mlp,
    fit_ridge_xy,
    labelled_rows,
    predict_regime,
    predict_residual_mlp,
    stable_feature_mask,
    univariate_cs_ics,
    walk_forward_predict,
    year_cs_ics,
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


def _train_slice(
    cache: dict[str, Any], train_from_days: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_tr = cache["train_x"].astype(np.float64)
    y_tr = cache["train_y"].astype(np.float64)
    d_tr = cache["train_d"].astype(np.int64)
    if train_from_days is not None:
        keep = d_tr >= int(train_from_days)
        x_tr, y_tr, d_tr = x_tr[keep], y_tr[keep], d_tr[keep]
    return x_tr, y_tr, d_tr


def _frozen(
    cache: dict[str, Any],
    *,
    train_from_days: int | None,
    ridge: float,
    rank_target: bool,
    cs_zscore: bool,
    mask_mode: str,
    date_halflife: float,
    y_winsor: float = 0.0,
    feat_winsor: float = 0.0,
    drop_disp_q: float = 0.0,
    huber_delta: float = 0.0,
    sign_constrain: bool = False,
    drop_crashes: bool = False,
    feature_mask_bool: np.ndarray | None = None,
    year_balance: bool = False,
) -> dict[str, Any]:
    min_names = int(cache["cs_min_names"])
    x_tr, y_tr, d_tr = _train_slice(cache, train_from_days)
    mask = feature_mask(mask_mode) if feature_mask_bool is None else feature_mask_bool
    w, b, train_ic = fit_ridge_xy(
        x_tr,
        y_tr,
        d_tr,
        ridge=ridge,
        min_names=min_names,
        cs_demean=True,
        cs_zscore=cs_zscore,
        rank_target=rank_target,
        feature_mask_bool=mask,
        date_halflife=date_halflife,
        y_winsor=y_winsor,
        feat_winsor=feat_winsor,
        drop_disp_q=drop_disp_q,
        huber_delta=huber_delta,
        sign_constrain=sign_constrain,
        drop_crashes=drop_crashes,
        year_balance=year_balance,
    )
    row: dict[str, Any] = {"train_is_cs_ic": train_ic, "n_train": int(y_tr.size)}
    for split in ("val", "test"):
        x = cache[f"{split}_x"].astype(np.float64)
        y = cache[f"{split}_y"].astype(np.float64)
        d = cache[f"{split}_d"].astype(np.int64)
        pred = x @ w.astype(np.float64) + b
        row[split] = cs_stats(pred, y, d, min_names=min_names)
    row["_w"] = w
    row["_b"] = b
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CS train→test collapse ablations.")
    p.add_argument("--data-dir", default="")
    p.add_argument("--universe", default="liquid", choices=("", "liquid", "liquid_wide"))
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
        ("frozen_rank_lam10", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0)),
        ("frozen_rank_lam100", dict(train_from_days=ymd(1999), rank_target=True, ridge=100.0)),
        ("frozen_rank_huber", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, huber_delta=1.0)),
        ("frozen_rank_crashes", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, drop_crashes=True)),
        ("frozen_rank_sign", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, sign_constrain=True)),
        ("frozen_rank_disp10", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, drop_disp_q=0.10)),
        ("frozen_rank_fwinsor", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, feat_winsor=3.0)),
        ("frozen_rank_core", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, mask_mode="core")),
        ("frozen_rank_nolong", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, mask_mode="no_long_ts")),
        ("frozen_rank_noohlc", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, mask_mode="no_ohlc")),
        ("frozen_value_winsor", dict(train_from_days=ymd(1999), rank_target=False, ridge=10.0, y_winsor=3.0)),
        ("frozen_year_balance", dict(train_from_days=ymd(1999), rank_target=True, ridge=10.0, feat_winsor=3.0, mask_mode="no_long_ts", year_balance=True)),
    ]
    base = dict(ridge=1.0, rank_target=False, cs_zscore=False, mask_mode="all", date_halflife=0.0)
    print(f"{'case':<28} {'val':>8} {'val_t':>7} {'test':>8} {'test_t':>7} {'ntr':>8}")
    def _public(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if not str(k).startswith("_")}

    def _emit(stats: dict[str, Any], name: str, kind: str) -> dict[str, Any]:
        stats = dict(stats)
        stats["name"] = name
        stats["kind"] = kind
        rows.append(_public(stats))
        val, test = stats["val"], stats["test"]
        ntr = stats.get("n_train", "na")
        print(
            f"{name:<28} {val['cs_ic']:+8.4f} {val['cs_ic_tstat']:7.2f} "
            f"{test['cs_ic']:+8.4f} {test['cs_ic_tstat']:7.2f} {ntr:>8}",
            flush=True,
        )
        return stats

    for name, kw in frozen_cases:
        cfg = {**base, **kw}
        _emit(_frozen(cache, **cfg), name, "frozen")

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
            "lookback_days": lookback,
            "val": _eval(pred[val_sel], y[val_sel], d[val_sel], min_names),
            "test": _eval(pred[test_sel], y[test_sel], d[test_sel], min_names),
        }
        _emit(stats, name, "walk_forward")

    # --- round 2: ranking objectives, stable mask, regime, residual, products ---
    x_tr, y_tr, d_tr = _train_slice(cache, ymd(1999))
    uni_tr = univariate_cs_ics(x_tr, y_tr, d_tr, min_names=min_names)
    uni_va = univariate_cs_ics(
        cache["val_x"].astype(np.float64),
        cache["val_y"].astype(np.float64),
        cache["val_d"].astype(np.int64),
        min_names=min_names,
    )
    print("univariate CS IC train vs val:", flush=True)
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<14} train={uni_tr[i]:+.4f} val={uni_va[i]:+.4f}", flush=True)
    x_va = cache["val_x"].astype(np.float64)
    y_va = cache["val_y"].astype(np.float64)
    d_va = cache["val_d"].astype(np.int64)
    x_te = cache["test_x"].astype(np.float64)
    y_te = cache["test_y"].astype(np.float64)
    d_te = cache["test_d"].astype(np.int64)

    def _score_w(w: np.ndarray, b: float = 0.0) -> dict[str, Any]:
        return {
            "n_train": int(y_tr.size),
            "val": cs_stats(x_va @ w + b, y_va, d_va, min_names=min_names),
            "test": cs_stats(x_te @ w + b, y_te, d_te, min_names=min_names),
        }

    w_ln, b_ln, ic_ln = fit_listnet_xy(
        x_tr, y_tr, d_tr, ridge=10.0, min_names=min_names, rank_target=True
    )
    row = _score_w(w_ln.astype(np.float64), b_ln)
    row["train_is_cs_ic"] = ic_ln
    _emit(row, "listnet_rank_lam10", "listnet")

    w_rn, b_rn, ic_rn = fit_ranknet_xy(
        x_tr, y_tr, d_tr, ridge=10.0, min_names=min_names, rank_target=True, steps=80
    )
    row = _score_w(w_rn.astype(np.float64), b_rn)
    row["train_is_cs_ic"] = ic_rn
    _emit(row, "ranknet_rank_lam10", "ranknet")

    keep = stable_feature_mask(x_tr, y_tr, d_tr, x_va, y_va, d_va, min_names=min_names)
    row = _frozen(
        cache,
        train_from_days=ymd(1999),
        ridge=10.0,
        rank_target=True,
        cs_zscore=False,
        mask_mode="all",
        date_halflife=0.0,
        feature_mask_bool=keep,
    )
    row["n_keep"] = int(keep.sum())
    _emit(row, "frozen_rank_stable", "frozen")

    names = list(FEATURE_NAMES)
    mkt_i = names.index("mkt_ret_1") if "mkt_ret_1" in names else 0
    vol_i = names.index("vol_level") if "vol_level" in names else 0
    for tag, col in (("mktabs", mkt_i), ("vol", vol_i)):
        w_lo, w_hi, split, tr_ic = fit_regime_ridge(
            x_tr, y_tr, d_tr, x_tr[:, col], ridge=10.0, min_names=min_names, rank_target=True
        )
        pred_va = predict_regime(x_va, d_va, x_va[:, col], w_lo, w_hi, split)
        pred_te = predict_regime(x_te, d_te, x_te[:, col], w_lo, w_hi, split)
        _emit(
            {
                "train_is_cs_ic": tr_ic,
                "n_train": int(y_tr.size),
                "split": split,
                "val": cs_stats(pred_va, y_va, d_va, min_names=min_names),
                "test": cs_stats(pred_te, y_te, d_te, min_names=min_names),
            },
            f"regime_{tag}",
            "regime",
        )

    pair_names = [n for n in ("cs_ret_1", "cs_rank_1", "cs_vol", "idio_sector") if n in names]
    cols = [names.index(n) for n in pair_names]
    x_tr_i = augment_cs_products(x_tr, cols)
    x_va_i = augment_cs_products(x_va, cols)
    x_te_i = augment_cs_products(x_te, cols)
    w_i, b_i, ic_i = fit_ridge_xy(
        x_tr_i, y_tr, d_tr, ridge=10.0, min_names=min_names, rank_target=True
    )
    _emit(
        {
            "train_is_cs_ic": ic_i,
            "n_train": int(y_tr.size),
            "val": cs_stats(x_va_i @ w_i + b_i, y_va, d_va, min_names=min_names),
            "test": cs_stats(x_te_i @ w_i + b_i, y_te, d_te, min_names=min_names),
        },
        "frozen_rank_products",
        "frozen",
    )

    base_fit = _frozen(
        cache,
        train_from_days=ymd(1999),
        ridge=10.0,
        rank_target=True,
        cs_zscore=False,
        mask_mode="all",
        date_halflife=0.0,
    )
    w0 = base_fit["_w"].astype(np.float64)
    pred_tr = x_tr @ w0
    resid = y_tr - pred_tr
    w1, b1, w2 = fit_residual_mlp(x_tr, resid, d_tr, hidden=8, ridge=25.0, steps=160)
    resid_va = predict_residual_mlp(x_va, d_va, w1, b1, w2)
    resid_te = predict_residual_mlp(x_te, d_te, w1, b1, w2)
    skip_va = x_va @ w0
    skip_te = x_te @ w0
    best_a, best_val = 0.0, -1e9
    for a in (0.0, 0.15, 0.3, 0.5, 0.8, 1.0):
        st = cs_stats(skip_va + a * resid_va, y_va, d_va, min_names=min_names)
        if st["cs_ic"] > best_val:
            best_val = float(st["cs_ic"])
            best_a = a
    _emit(
        {
            "n_train": int(y_tr.size),
            "blend": best_a,
            "val": cs_stats(skip_va + best_a * resid_va, y_va, d_va, min_names=min_names),
            "test": cs_stats(skip_te + best_a * resid_te, y_te, d_te, min_names=min_names),
        },
        f"ensemble_mlp_a{best_a:.2f}",
        "ensemble",
    )

    # Single-pass leave-one-feature-out on val (not iterative greedy).
    loo_best = float(base_fit["val"]["cs_ic"])
    loo_j = -1
    loo_val = None
    loo_test = None
    for j, name in enumerate(names):
        trial = np.ones(x_tr.shape[1], dtype=bool)
        trial[j] = False
        w, b, _ = fit_ridge_xy(
            x_tr,
            y_tr,
            d_tr,
            ridge=10.0,
            min_names=min_names,
            rank_target=True,
            feature_mask_bool=trial,
        )
        st = cs_stats(x_va @ w + b, y_va, d_va, min_names=min_names)
        if st["cs_ic"] > loo_best + 1e-5:
            loo_best = float(st["cs_ic"])
            loo_j = int(j)
            loo_val = st
            loo_test = cs_stats(x_te @ w + b, y_te, d_te, min_names=min_names)
    if loo_j >= 0 and loo_val is not None and loo_test is not None:
        _emit(
            {
                "n_train": int(y_tr.size),
                "dropped": [names[loo_j]],
                "val": loo_val,
                "test": loo_test,
            },
            f"loo_drop_{names[loo_j]}",
            "frozen",
        )
    else:
        _emit(base_fit, "loo_no_drop", "frozen")

    extra_keep = np.ones(x_tr_i.shape[1] - x_tr.shape[1], dtype=bool)
    for tag, ridge, fwinsor, use_nolong in (
        ("prod_fwinsor", 10.0, 3.0, False),
        ("prod_lam100", 100.0, 0.0, False),
        ("prod_nolong", 10.0, 0.0, True),
        ("prod_fwinsor_lam100", 100.0, 3.0, False),
    ):
        mask = np.concatenate([feature_mask("no_long_ts"), extra_keep]) if use_nolong else None
        w, b, ic = fit_ridge_xy(
            x_tr_i,
            y_tr,
            d_tr,
            ridge=ridge,
            min_names=min_names,
            rank_target=True,
            feat_winsor=fwinsor,
            feature_mask_bool=mask,
        )
        _emit(
            {
                "train_is_cs_ic": ic,
                "n_train": int(y_tr.size),
                "val": cs_stats(x_va_i @ w + b, y_va, d_va, min_names=min_names),
                "test": cs_stats(x_te_i @ w + b, y_te, d_te, min_names=min_names),
            },
            tag,
            "frozen",
        )

    promoted = _frozen(
        cache,
        train_from_days=ymd(1999),
        ridge=10.0,
        rank_target=True,
        cs_zscore=False,
        mask_mode="all",
        date_halflife=0.0,
    )
    years = {
        "val": year_cs_ics(x_va @ promoted["_w"] + promoted["_b"], y_va, d_va, min_names=min_names),
        "test": year_cs_ics(x_te @ promoted["_w"] + promoted["_b"], y_te, d_te, min_names=min_names),
    }
    print("year CS IC (rank+lam10):", flush=True)
    for split in ("val", "test"):
        for row in years[split]:
            print(
                f"  {split} {int(row['year'])}: cs_ic={row['cs_ic']:+.4f} "
                f"t={row['cs_ic_tstat']:.2f} n={int(row['cs_n_dates'])}",
                flush=True,
            )
    rows.append({"name": "year_cs_ic_rank_lam10", "kind": "diagnostic", **years})

    out_path = Path(args.out)
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
