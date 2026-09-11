"""Val-gated nonstationarity attacks on the promoted CS skip.

1) Regime-conditioned last-bar ridge heads (train-only bucket edges).
2) Surgical drop of vol + cs_products (not the auto year-stable mask).
3) Readout-only trailing train window (3y/5y/7y ending at val start).
4) Equal-weight ensemble diagnostic if both skips stay val-positive.

Promote only if locked val mean CS IC beats the promoted skip by >= 0.003
AND val-2017 CS IC recovers. Test is printed after the val gate, never used
to pick a spec.

    python scripts/cs_regime_ablate.py --data-dir data --universe liquid
    python scripts/cs_regime_ablate.py --cache /tmp/cs_lastbars44.npz
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
    assign_score_buckets,
    bucket_edges_from_train,
    cs_stats,
    feature_mask,
    fit_regime_heads,
    fit_ridge_xy,
    labelled_rows,
    late_train_holdout_mask,
    predict_regime_heads,
    predict_regime_mixture,
    regime_score_map,
    trailing_window_keep,
    year_cs_ics,
)

PROMOTED = dict(
    ridge=10.0,
    rank_target=True,
    feat_winsor=3.0,
    mask_mode="no_long_ts",
)
PROMOTED_VAL_IC = 0.0290
VAL_LIFT = 0.003
VAL_2017_FLOOR = 0.015
VAL_2017_T = 1.5


def _cfg(data_dir: str, universe: str) -> DataConfig:
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


def _year_row(rows: list[dict[str, Any]], year: int) -> dict[str, float]:
    for row in rows:
        if int(row.get("year", -1)) == int(year):
            return row
    return {"year": float(year), "cs_ic": float("nan"), "cs_ic_tstat": float("nan")}


def _fit_kwargs(mask_mode: str) -> dict[str, Any]:
    return dict(
        ridge=PROMOTED["ridge"],
        rank_target=PROMOTED["rank_target"],
        feat_winsor=PROMOTED["feat_winsor"],
        feature_mask_bool=feature_mask(mask_mode),
    )


def _train_xy(cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = cache["train_d"].astype(np.int64) >= _ymd(1999)
    return (
        cache["train_x"].astype(np.float64)[keep],
        cache["train_y"].astype(np.float64)[keep],
        cache["train_d"].astype(np.int64)[keep],
    )


def _panel_xy(cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Train+val+test rows so trailing SPY vol can see history before val."""
    x = np.concatenate(
        [
            cache["train_x"].astype(np.float64),
            cache["val_x"].astype(np.float64),
            cache["test_x"].astype(np.float64),
        ],
        axis=0,
    )
    d = np.concatenate(
        [
            cache["train_d"].astype(np.int64),
            cache["val_d"].astype(np.int64),
            cache["test_d"].astype(np.int64),
        ],
        axis=0,
    )
    return x, d


def _scores_for(cache: dict[str, Any], kind: str) -> dict[int, float]:
    x, d = _panel_xy(cache)
    return regime_score_map(x, d, kind=kind)


def _score_pred(
    pred: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    min_names: int,
) -> dict[str, Any]:
    stats = cs_stats(pred, y, d, min_names=min_names)
    years = year_cs_ics(pred, y, d, min_names=min_names)
    y2017 = _year_row(years, 2017)
    stats["cs_ic_2017"] = float(y2017.get("cs_ic", float("nan")))
    stats["cs_t_2017"] = float(y2017.get("cs_ic_tstat", float("nan")))
    stats["years"] = years
    return stats


def _split_score(
    pred_fn,
    cache: dict[str, Any],
    min_names: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        x = cache[f"{split}_x"].astype(np.float64)
        y = cache[f"{split}_y"].astype(np.float64)
        d = cache[f"{split}_d"].astype(np.int64)
        row[split] = _score_pred(pred_fn(x, d), y, d, min_names)
    return row


def _verdict(val: dict[str, Any], baseline_val: float, *, kind: str) -> str:
    ic = float(val.get("cs_ic", float("nan")))
    ic2017 = float(val.get("cs_ic_2017", float("nan")))
    t2017 = float(val.get("cs_t_2017", float("nan")))
    if not np.isfinite(ic):
        return "discard"
    lift = ic - float(baseline_val)
    recovered = (
        np.isfinite(ic2017)
        and np.isfinite(t2017)
        and ic2017 >= VAL_2017_FLOOR
        and t2017 >= VAL_2017_T
    )
    if lift >= VAL_LIFT and recovered:
        return "promote"
    if kind == "ensemble":
        return "diagnostic_only"
    return "discard"


def _print_row(name: str, val: dict[str, Any], test: dict[str, Any], extra: str = "") -> None:
    print(
        f"{name:<32} val={val['cs_ic']:+.4f} t={val['cs_ic_tstat']:6.2f} "
        f"val2017={val['cs_ic_2017']:+.4f} t2017={val['cs_t_2017']:6.2f} "
        f"test={test['cs_ic']:+.4f} t={test['cs_ic_tstat']:6.2f}{extra}",
        flush=True,
    )


def _fit_promoted_w(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    min_names: int,
    mask_mode: str,
) -> tuple[np.ndarray, float]:
    w, b, ic = fit_ridge_xy(
        x, y, d, min_names=min_names, **_fit_kwargs(mask_mode)
    )
    return w.astype(np.float64), ic


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regime / drop / trailing-window CS attacks.")
    p.add_argument("--data-dir", default="")
    p.add_argument("--universe", default="liquid", choices=("", "liquid", "liquid_wide"))
    p.add_argument("--cache", default="/tmp/cs_lastbars44.npz")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--out", default="/tmp/cs_regime_ablate.json")
    args = p.parse_args(argv)

    cache_path = Path(args.cache)
    if args.rebuild or not cache_path.exists():
        if not args.data_dir:
            print("need --data-dir to build the last-bar cache", file=sys.stderr)
            return 2
        print(f"building last bars from {args.data_dir} ...", flush=True)
        bundle = build_datasets(_cfg(args.data_dir, args.universe), log_fn=print)
        _dump(bundle, cache_path)
        print(f"wrote {cache_path}", flush=True)
    cache = _load(cache_path)
    min_names = int(cache["cs_min_names"])
    n_feat = int(cache["train_x"].shape[1])
    if n_feat != len(FEATURE_NAMES):
        print(
            f"refusing stale cache: {n_feat} != {len(FEATURE_NAMES)}. "
            "Pass --rebuild --data-dir ...",
            file=sys.stderr,
        )
        return 2
    x_tr, y_tr, d_tr = _train_xy(cache)
    x_va = cache["val_x"].astype(np.float64)
    y_va = cache["val_y"].astype(np.float64)
    d_va = cache["val_d"].astype(np.int64)
    def _as_day(days: int) -> str:
        return str(np.datetime64("1970-01-01") + np.timedelta64(int(days), "D"))

    print(
        f"cache {cache_path} features={n_feat} min_names={min_names} "
        f"n_trade={cache.get('n_trade', '?')} universe={args.universe} "
        f"train={_as_day(int(d_tr.min()))}->{_as_day(int(d_tr.max()))} "
        f"val={_as_day(int(d_va.min()))}->{_as_day(int(d_va.max()))}",
        flush=True,
    )

    cases: list[dict[str, Any]] = []

    w0, ic0 = _fit_promoted_w(x_tr, y_tr, d_tr, min_names, PROMOTED["mask_mode"])
    base = _split_score(lambda x, d: x @ w0, cache, min_names)
    base.update(
        {
            "name": "promoted_skip",
            "kind": "baseline",
            "train_is_cs_ic": ic0,
            "mask_mode": PROMOTED["mask_mode"],
        }
    )
    baseline_val = float(base["val"]["cs_ic"])
    _print_row("promoted_skip", base["val"], base["test"])
    cases.append(base)

    # --- 1) regime heads: choose kind + K on late-train only, then refit ---
    fit_mask, sel_mask = late_train_holdout_mask(d_tr)
    x_fit, y_fit, d_fit = x_tr[fit_mask], y_tr[fit_mask], d_tr[fit_mask]
    x_sel, y_sel, d_sel = x_tr[sel_mask], y_tr[sel_mask], d_tr[sel_mask]
    print(
        f"late-train holdout { _as_day(int(d_sel.min())) }->{_as_day(int(d_sel.max()))} "
        f"(fit ends {_as_day(int(d_fit.max()))})",
        flush=True,
    )
    mask = feature_mask(PROMOTED["mask_mode"])
    regime_grid = [
        ("spy_vol", 2),
        ("spy_vol", 3),
        ("spy_vol", 5),
        ("cs_disp", 2),
        ("cs_disp", 3),
        ("cs_disp", 5),
    ]
    inner_rows: list[dict[str, Any]] = []
    print("regime inner-train selection (no locked val):", flush=True)
    fit_keys = set(int(v) for v in np.unique(d_fit))
    sel_keys = set(int(v) for v in np.unique(d_sel))
    for kind, k in regime_grid:
        scores_tr = regime_score_map(x_tr, d_tr, kind=kind)
        scores_fit = {dk: v for dk, v in scores_tr.items() if dk in fit_keys}
        scores_sel = {dk: v for dk, v in scores_tr.items() if dk in sel_keys}
        if len(scores_fit) < 80:
            print(f"  skip {kind}_k{k}: too few fit scores", flush=True)
            continue
        edges = bucket_edges_from_train(scores_fit, k)
        buckets_fit = assign_score_buckets(scores_fit, edges)
        buckets_sel = assign_score_buckets(scores_sel, edges)
        w_heads, _, counts = fit_regime_heads(
            x_fit,
            y_fit,
            d_fit,
            buckets_fit,
            n_buckets=k,
            min_names=min_names,
            min_dates_per_bucket=40,
            ridge=PROMOTED["ridge"],
            rank_target=PROMOTED["rank_target"],
            feat_winsor=PROMOTED["feat_winsor"],
            feature_mask_bool=mask,
        )
        if min(counts) < 40:
            print(
                f"  skip {kind}_k{k}: bucket counts {counts}",
                flush=True,
            )
            continue
        pred_hard = predict_regime_heads(x_sel, d_sel, buckets_sel, w_heads)
        hard = cs_stats(pred_hard, y_sel, d_sel, min_names=min_names)
        best_mix = None
        best_temp = None
        fit_vals = np.asarray([v for v in scores_fit.values() if np.isfinite(v)])
        scale = float(np.std(fit_vals)) if fit_vals.size else 1.0
        scale = max(scale, 1e-6)
        for temp_mult in (0.5, 1.0, 2.0):
            pred_mix = predict_regime_mixture(
                x_sel,
                d_sel,
                scores_sel,
                w_heads,
                edges,
                temperature=temp_mult * scale,
            )
            mix = cs_stats(pred_mix, y_sel, d_sel, min_names=min_names)
            if best_mix is None or float(mix["cs_ic"]) > float(best_mix["cs_ic"]):
                best_mix = mix
                best_temp = temp_mult * scale
        hard_ic = float(hard["cs_ic"])
        mix_ic = float(best_mix["cs_ic"]) if best_mix is not None else float("nan")
        use_mix = np.isfinite(mix_ic) and mix_ic > hard_ic + 0.001
        inner_ic = mix_ic if use_mix else hard_ic
        inner_rows.append(
            {
                "kind": kind,
                "n_buckets": k,
                "counts": counts,
                "inner_cs_ic": inner_ic,
                "hard_cs_ic": hard_ic,
                "mix_cs_ic": mix_ic,
                "use_mix": use_mix,
                "temperature": best_temp if use_mix else None,
            }
        )
        print(
            f"  {kind}_k{k:<1} inner={inner_ic:+.4f} hard={hard_ic:+.4f} "
            f"mix={mix_ic:+.4f} mix?={use_mix} counts={counts}",
            flush=True,
        )

    chosen = None
    if inner_rows:
        chosen = max(inner_rows, key=lambda r: (float(r["inner_cs_ic"]), -int(r["n_buckets"])))
        print(
            f"train-selected regime: {chosen['kind']} k={chosen['n_buckets']} "
            f"mix={chosen['use_mix']} inner={chosen['inner_cs_ic']:+.4f}",
            flush=True,
        )

    if chosen is not None:
        panel_scores = _scores_for(cache, chosen["kind"])
        scores_tr = {int(k): panel_scores[int(k)] for k in np.unique(d_tr) if int(k) in panel_scores}
        edges = bucket_edges_from_train(scores_tr, int(chosen["n_buckets"]))
        buckets_tr = assign_score_buckets(scores_tr, edges)
        w_heads, tr_ic, counts = fit_regime_heads(
            x_tr,
            y_tr,
            d_tr,
            buckets_tr,
            n_buckets=int(chosen["n_buckets"]),
            min_names=min_names,
            min_dates_per_bucket=60,
            feature_mask_bool=mask,
            ridge=PROMOTED["ridge"],
            rank_target=PROMOTED["rank_target"],
            feat_winsor=PROMOTED["feat_winsor"],
        )

        def _reg_pred_safe(x: np.ndarray, d: np.ndarray) -> np.ndarray:
            # Look up scores from the full panel so val/test keep train history.
            scores = {
                int(k): panel_scores[int(k)]
                for k in np.unique(d.astype(np.int64))
                if int(k) in panel_scores
            }
            if chosen["use_mix"] and chosen["temperature"] is not None:
                return predict_regime_mixture(
                    x, d, scores, w_heads, edges, temperature=float(chosen["temperature"])
                )
            return predict_regime_heads(x, d, assign_score_buckets(scores, edges), w_heads)

        name = f"regime_{chosen['kind']}_k{chosen['n_buckets']}"
        if chosen["use_mix"]:
            name += "_mix"
        row = _split_score(_reg_pred_safe, cache, min_names)
        row.update(
            {
                "name": name,
                "kind": "regime",
                "train_selected": True,
                "train_is_cs_ic": tr_ic,
                "n_buckets": int(chosen["n_buckets"]),
                "regime_kind": chosen["kind"],
                "use_mix": bool(chosen["use_mix"]),
                "temperature": chosen["temperature"],
                "bucket_counts": counts,
                "inner_cs_ic": chosen["inner_cs_ic"],
            }
        )
        _print_row(name, row["val"], row["test"], extra=f" counts={counts}")
        cases.append(row)

    # --- 2) surgical vol + cs_products drop ---
    w_drop, ic_drop = _fit_promoted_w(x_tr, y_tr, d_tr, min_names, "no_vol_products")
    drop_row = _split_score(lambda x, d: x @ w_drop, cache, min_names)
    drop_row.update(
        {
            "name": "drop_vol_cs_products",
            "kind": "surgical_drop",
            "train_is_cs_ic": ic_drop,
            "mask_mode": "no_vol_products",
            "n_keep": int(feature_mask("no_vol_products").sum()),
        }
    )
    _print_row("drop_vol_cs_products", drop_row["val"], drop_row["test"])
    cases.append(drop_row)

    # --- 3) readout-only trailing windows ending at val start ---
    val_start = int(np.min(d_va)) if d_va.size else 0
    for years in (3.0, 5.0, 7.0):
        keep = trailing_window_keep(d_tr, end_days=val_start, years=years)
        n_dates = int(np.unique(d_tr[keep]).size) if bool(keep.any()) else 0
        name = f"trailing_{int(years)}y"
        if n_dates < 60:
            print(f"{name:<32} skip: only {n_dates} train dates", flush=True)
            continue
        w_tw, ic_tw = _fit_promoted_w(
            x_tr[keep], y_tr[keep], d_tr[keep], min_names, PROMOTED["mask_mode"]
        )
        tw = _split_score(lambda x, d, w=w_tw: x @ w, cache, min_names)
        tw.update(
            {
                "name": name,
                "kind": "trailing_window",
                "train_is_cs_ic": ic_tw,
                "n_train_dates": n_dates,
                "years": years,
                "val_start": val_start,
            }
        )
        _print_row(name, tw["val"], tw["test"], extra=f" n_dates={n_dates}")
        cases.append(tw)

    # --- 4) equal-weight ensemble if both val-positive ---
    drop_val = float(drop_row["val"]["cs_ic"])
    if baseline_val > 0 and drop_val > 0:
        ens = _split_score(
            lambda x, d: 0.5 * (x @ w0) + 0.5 * (x @ w_drop),
            cache,
            min_names,
        )
        ens.update(
            {
                "name": "ensemble_promoted_novol",
                "kind": "ensemble",
            }
        )
        _print_row("ensemble_promoted_novol", ens["val"], ens["test"])
        cases.append(ens)

    print(
        f"{'case':<32} {'verdict':<18} val_lift  val2017",
        flush=True,
    )
    verdicts: list[dict[str, Any]] = []
    promoted_name = None
    for stats in cases:
        name = stats["name"]
        if name == "promoted_skip":
            verdict = "baseline"
        else:
            verdict = _verdict(stats["val"], baseline_val, kind=str(stats.get("kind", "")))
        lift = float(stats["val"]["cs_ic"]) - baseline_val
        print(
            f"  {name:<30} {verdict:<18} {lift:+.4f}  "
            f"{stats['val']['cs_ic_2017']:+.4f}",
            flush=True,
        )
        verdicts.append(
            {
                "name": name,
                "kind": stats.get("kind"),
                "val_cs_ic": float(stats["val"]["cs_ic"]),
                "val_cs_t": float(stats["val"]["cs_ic_tstat"]),
                "val_2017_cs_ic": float(stats["val"]["cs_ic_2017"]),
                "val_2017_t": float(stats["val"]["cs_t_2017"]),
                "test_cs_ic": float(stats["test"]["cs_ic"]),
                "test_cs_t": float(stats["test"]["cs_ic_tstat"]),
                "verdict": verdict,
            }
        )
        if verdict == "promote" and promoted_name is None:
            promoted_name = name

    payload = {
        "universe": args.universe,
        "n_features": n_feat,
        "n_trade": cache.get("n_trade"),
        "baseline_val_cs_ic": baseline_val,
        "promote_bar": PROMOTED_VAL_IC + VAL_LIFT,
        "val_2017_floor": VAL_2017_FLOOR,
        "regime_inner": inner_rows,
        "train_selected_regime": (
            {k: v for k, v in chosen.items() if k != "edges"} if chosen else None
        ),
        "cases": [
            {
                k: v
                for k, v in r.items()
                if k not in ("train",) or True
            }
            for r in cases
        ],
        "verdicts": verdicts,
        "promoted": promoted_name,
    }
    # Drop bulky per-year tables from the default payload except val 2017 already in stats.
    slim_cases = []
    for r in cases:
        slim = {k: v for k, v in r.items() if k not in ("train",)}
        for split in ("val", "test"):
            if split in slim and isinstance(slim[split], dict):
                slim[split] = {
                    kk: vv
                    for kk, vv in slim[split].items()
                    if kk != "years"
                }
        slim_cases.append(slim)
    payload["cases"] = slim_cases
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_path}")
    if promoted_name:
        print(f"PROMOTE {promoted_name} on locked val; report its test in the PR.")
    else:
        print(
            "NO PROMOTE: no attack lifted locked val by "
            f">={VAL_LIFT:.3f} with val-2017 recovery "
            f"(floor {VAL_2017_FLOOR:+.3f}, t>={VAL_2017_T})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
