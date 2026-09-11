"""Val-gated causal trailing skip-IC shrink, plus overnight/session labels.

Frozen promoted skip weights ``w``. At date t, shrink or flatten using mean
CS IC on dates in [t-L, t) (labels strictly before t). Choose L and the
flatten/scale mode on a late-train holdout only.

Positive scale does not change Pearson CS IC; flattening dead days (and
counting them as IC=0) is the overlay. Promote only if locked val mean CS IC
beats the promoted skip by >= 0.003 AND val-2017 recovers.

If shrink misses, optional ``--also-labels`` rebuilds last bars with overnight
or session residual labels (still causal; features at close t). That is a
different estimand — reported honestly, not mixed into the close-to-close
headline.

    python scripts/cs_shrink_ablate.py --data-dir data --universe liquid
    python scripts/cs_shrink_ablate.py --cache /tmp/cs_lastbars44.npz
    python scripts/cs_shrink_ablate.py --data-dir data --universe liquid --also-labels
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
    apply_skip_ic_shrink,
    cs_stats,
    feature_mask,
    fit_ridge_xy,
    labelled_rows,
    late_train_holdout_mask,
    trailing_skip_ic_stats,
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
LOOKBACKS = (21, 63, 126, 252, 504)
MODES = ("scale", "flatten_tstat", "flatten_weak")
LABEL_KINDS = ("overnight", "session")


def _cfg(data_dir: str, universe: str, *, label_return: str = "close") -> DataConfig:
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
        label_return=label_return,
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


def _fit_kwargs() -> dict[str, Any]:
    return dict(
        ridge=PROMOTED["ridge"],
        rank_target=PROMOTED["rank_target"],
        feat_winsor=PROMOTED["feat_winsor"],
        feature_mask_bool=feature_mask(PROMOTED["mask_mode"]),
    )


def _train_xy(cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = cache["train_d"].astype(np.int64) >= _ymd(1999)
    return (
        cache["train_x"].astype(np.float64)[keep],
        cache["train_y"].astype(np.float64)[keep],
        cache["train_d"].astype(np.int64)[keep],
    )


def _concat_panel(cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate(
        [
            cache["train_x"].astype(np.float64),
            cache["val_x"].astype(np.float64),
            cache["test_x"].astype(np.float64),
        ],
        axis=0,
    )
    y = np.concatenate(
        [
            cache["train_y"].astype(np.float64),
            cache["val_y"].astype(np.float64),
            cache["test_y"].astype(np.float64),
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
    return x, y, d


def _score_pred(
    pred: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    min_names: int,
    *,
    flat_as_zero: bool,
) -> dict[str, Any]:
    stats = cs_stats(pred, y, d, min_names=min_names, flat_as_zero=flat_as_zero)
    years = year_cs_ics(pred, y, d, min_names=min_names, flat_as_zero=flat_as_zero)
    y2017 = _year_row(years, 2017)
    stats["cs_ic_2017"] = float(y2017.get("cs_ic", float("nan")))
    stats["cs_t_2017"] = float(y2017.get("cs_ic_tstat", float("nan")))
    stats["years"] = years
    return stats


def _as_day(days: int) -> str:
    return str(np.datetime64("1970-01-01") + np.timedelta64(int(days), "D"))


def _min_obs(lookback_days: int) -> int:
    return max(10, int(lookback_days) // 4)


def _verdict(val: dict[str, Any], baseline_val: float) -> str:
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
    return "discard"


def _print_row(name: str, val: dict[str, Any], test: dict[str, Any], extra: str = "") -> None:
    n_flat = val.get("cs_n_flat", 0.0)
    print(
        f"{name:<36} val={val['cs_ic']:+.4f} t={val['cs_ic_tstat']:6.2f} "
        f"val2017={val['cs_ic_2017']:+.4f} t2017={val['cs_t_2017']:6.2f} "
        f"test={test['cs_ic']:+.4f} t={test['cs_ic_tstat']:6.2f} "
        f"flat={int(n_flat)}{extra}",
        flush=True,
    )


def _shrink_pred(
    pred: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    *,
    lookback_days: int,
    mode: str,
    train_ic: float,
    min_names: int,
) -> np.ndarray:
    keys, mu, tt, _n = trailing_skip_ic_stats(
        pred,
        y,
        d,
        lookback_days=lookback_days,
        min_names=min_names,
        min_obs=_min_obs(lookback_days),
    )
    return apply_skip_ic_shrink(
        pred,
        d,
        date_keys=keys,
        trailing_ic=mu,
        trailing_t=tt,
        train_ic=train_ic,
        mode=mode,
    )


def _split_shrunk(
    cache: dict[str, Any],
    w: np.ndarray,
    *,
    lookback_days: int,
    mode: str,
    train_ic: float,
    min_names: int,
) -> dict[str, Any]:
    x_all, y_all, d_all = _concat_panel(cache)
    raw = x_all @ w
    shrunk = _shrink_pred(
        raw,
        y_all,
        d_all,
        lookback_days=lookback_days,
        mode=mode,
        train_ic=train_ic,
        min_names=min_names,
    )
    row: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        d = cache[f"{split}_d"].astype(np.int64)
        y = cache[f"{split}_y"].astype(np.float64)
        d = cache[f"{split}_d"].astype(np.int64)
        n_tr = cache["train_x"].shape[0]
        n_va = cache["val_x"].shape[0]
        if split == "train":
            sl = slice(0, n_tr)
        elif split == "val":
            sl = slice(n_tr, n_tr + n_va)
        else:
            sl = slice(n_tr + n_va, None)
        row[split] = _score_pred(
            shrunk[sl], y, d, min_names, flat_as_zero=True
        )
    return row


def _ensure_cache(
    cache_path: Path,
    *,
    data_dir: str,
    universe: str,
    rebuild: bool,
    label_return: str = "close",
) -> dict[str, Any]:
    if rebuild or not cache_path.exists():
        if not data_dir:
            print("need --data-dir to build the last-bar cache", file=sys.stderr)
            raise SystemExit(2)
        print(
            f"building last bars from {data_dir} label_return={label_return} ...",
            flush=True,
        )
        bundle = build_datasets(
            _cfg(data_dir, universe, label_return=label_return), log_fn=print
        )
        _dump(bundle, cache_path)
        print(f"wrote {cache_path}", flush=True)
    cache = _load(cache_path)
    n_feat = int(cache["train_x"].shape[1])
    if n_feat != len(FEATURE_NAMES):
        print(
            f"refusing stale cache: {n_feat} != {len(FEATURE_NAMES)}. "
            "Pass --rebuild --data-dir ...",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return cache


def _run_promoted_skip(cache: dict[str, Any]) -> tuple[np.ndarray, float, dict[str, Any]]:
    min_names = int(cache["cs_min_names"])
    x_tr, y_tr, d_tr = _train_xy(cache)
    w, _, ic0 = fit_ridge_xy(x_tr, y_tr, d_tr, min_names=min_names, **_fit_kwargs())
    w = w.astype(np.float64)
    row: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        x = cache[f"{split}_x"].astype(np.float64)
        y = cache[f"{split}_y"].astype(np.float64)
        d = cache[f"{split}_d"].astype(np.int64)
        row[split] = _score_pred(x @ w, y, d, min_names, flat_as_zero=False)
    return w, float(ic0), row


def _run_shrink_attack(cache: dict[str, Any]) -> dict[str, Any]:
    min_names = int(cache["cs_min_names"])
    x_tr, y_tr, d_tr = _train_xy(cache)
    x_va = cache["val_x"].astype(np.float64)
    d_va = cache["val_d"].astype(np.int64)
    print(
        f"features={int(cache['train_x'].shape[1])} min_names={min_names} "
        f"n_trade={cache.get('n_trade', '?')} "
        f"train={_as_day(int(d_tr.min()))}->{_as_day(int(d_tr.max()))} "
        f"val={_as_day(int(d_va.min()))}->{_as_day(int(d_va.max()))}",
        flush=True,
    )

    w0, ic0, base_row = _run_promoted_skip(cache)
    base = dict(base_row)
    base.update(
        {
            "name": "promoted_skip",
            "kind": "baseline",
            "train_is_cs_ic": ic0,
        }
    )
    baseline_val = float(base["val"]["cs_ic"])
    _print_row("promoted_skip", base["val"], base["test"])

    fit_mask, sel_mask = late_train_holdout_mask(d_tr)
    x_fit, y_fit, d_fit = x_tr[fit_mask], y_tr[fit_mask], d_tr[fit_mask]
    y_sel, d_sel = y_tr[sel_mask], d_tr[sel_mask]
    print(
        f"late-train holdout {_as_day(int(d_sel.min()))}->{_as_day(int(d_sel.max()))} "
        f"(fit ends {_as_day(int(d_fit.max()))})",
        flush=True,
    )
    w_fit, _, ic_fit = fit_ridge_xy(
        x_fit, y_fit, d_fit, min_names=min_names, **_fit_kwargs()
    )
    w_fit = w_fit.astype(np.float64)
    pred_tr = x_tr @ w_fit
    pred_sel = pred_tr[sel_mask]
    inner_base = cs_stats(pred_sel, y_sel, d_sel, min_names=min_names, flat_as_zero=True)
    print(
        f"inner unshrunk cs_ic={float(inner_base['cs_ic']):+.4f} "
        f"t={float(inner_base['cs_ic_tstat']):.2f}",
        flush=True,
    )

    inner_rows: list[dict[str, Any]] = []
    print("shrink inner-train selection (no locked val):", flush=True)
    for L in LOOKBACKS:
        for mode in MODES:
            shrunk_tr = _shrink_pred(
                pred_tr,
                y_tr,
                d_tr,
                lookback_days=L,
                mode=mode,
                train_ic=ic_fit,
                min_names=min_names,
            )
            inner = cs_stats(
                shrunk_tr[sel_mask],
                y_sel,
                d_sel,
                min_names=min_names,
                flat_as_zero=True,
            )
            row = {
                "lookback_days": int(L),
                "mode": mode,
                "inner_cs_ic": float(inner["cs_ic"]),
                "inner_cs_t": float(inner["cs_ic_tstat"]),
                "inner_n_flat": float(inner.get("cs_n_flat", 0.0)),
            }
            inner_rows.append(row)
            print(
                f"  L={L:<4} {mode:<16} inner={row['inner_cs_ic']:+.4f} "
                f"t={row['inner_cs_t']:6.2f} flat={int(row['inner_n_flat'])}",
                flush=True,
            )

    chosen = max(inner_rows, key=lambda r: (float(r["inner_cs_ic"]), -int(r["lookback_days"])))
    print(
        f"train-selected shrink: L={chosen['lookback_days']} mode={chosen['mode']} "
        f"inner={chosen['inner_cs_ic']:+.4f} (unshrunk {float(inner_base['cs_ic']):+.4f})",
        flush=True,
    )

    shrink_row = _split_shrunk(
        cache,
        w0,
        lookback_days=int(chosen["lookback_days"]),
        mode=str(chosen["mode"]),
        train_ic=ic0,
        min_names=min_names,
    )
    name = f"shrink_L{chosen['lookback_days']}_{chosen['mode']}"
    shrink_row.update(
        {
            "name": name,
            "kind": "skip_ic_shrink",
            "train_selected": True,
            "lookback_days": int(chosen["lookback_days"]),
            "mode": chosen["mode"],
            "inner_cs_ic": chosen["inner_cs_ic"],
            "train_is_cs_ic": ic0,
        }
    )
    _print_row(name, shrink_row["val"], shrink_row["test"])

    cases = [base, shrink_row]
    print(f"{'case':<36} {'verdict':<18} val_lift  val2017", flush=True)
    verdicts: list[dict[str, Any]] = []
    promoted_name = None
    for stats in cases:
        case_name = stats["name"]
        if case_name == "promoted_skip":
            verdict = "baseline"
        else:
            verdict = _verdict(stats["val"], baseline_val)
        lift = float(stats["val"]["cs_ic"]) - baseline_val
        print(
            f"  {case_name:<34} {verdict:<18} {lift:+.4f}  "
            f"{stats['val']['cs_ic_2017']:+.4f}",
            flush=True,
        )
        verdicts.append(
            {
                "name": case_name,
                "kind": stats.get("kind"),
                "val_cs_ic": float(stats["val"]["cs_ic"]),
                "val_cs_t": float(stats["val"]["cs_ic_tstat"]),
                "val_2017_cs_ic": float(stats["val"]["cs_ic_2017"]),
                "val_2017_t": float(stats["val"]["cs_t_2017"]),
                "test_cs_ic": float(stats["test"]["cs_ic"]),
                "test_cs_t": float(stats["test"]["cs_ic_tstat"]),
                "cs_n_flat": float(stats["val"].get("cs_n_flat", 0.0)),
                "verdict": verdict,
                "lookback_days": stats.get("lookback_days"),
                "mode": stats.get("mode"),
            }
        )
        if verdict == "promote" and promoted_name is None:
            promoted_name = case_name

    slim = []
    for r in cases:
        item = {k: v for k, v in r.items() if k != "train"}
        for split in ("val", "test"):
            if split in item and isinstance(item[split], dict):
                item[split] = {
                    kk: vv for kk, vv in item[split].items() if kk != "years"
                }
        slim.append(item)
    return {
        "baseline_val_cs_ic": baseline_val,
        "inner_unshrunk_cs_ic": float(inner_base["cs_ic"]),
        "inner_rows": inner_rows,
        "train_selected": {
            "lookback_days": int(chosen["lookback_days"]),
            "mode": chosen["mode"],
            "inner_cs_ic": chosen["inner_cs_ic"],
        },
        "cases": slim,
        "verdicts": verdicts,
        "promoted": promoted_name,
    }


def _run_label_attack(
    *,
    data_dir: str,
    universe: str,
    cache_stem: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    print("label-return attacks (new estimand; not mixed into close-to-close gate):", flush=True)
    for kind in LABEL_KINDS:
        path = cache_stem.with_name(f"{cache_stem.stem}_{kind}.npz")
        cache = _ensure_cache(
            path,
            data_dir=data_dir,
            universe=universe,
            rebuild=True,
            label_return=kind,
        )
        w, ic0, split = _run_promoted_skip(cache)
        del w
        name = f"label_{kind}"
        split.update(
            {
                "name": name,
                "kind": "label_return",
                "label_return": kind,
                "train_is_cs_ic": ic0,
            }
        )
        _print_row(name, split["val"], split["test"], extra=f" y={kind}")
        rows.append(
            {
                "name": name,
                "label_return": kind,
                "train_is_cs_ic": ic0,
                "val": {k: v for k, v in split["val"].items() if k != "years"},
                "test": {k: v for k, v in split["test"].items() if k != "years"},
                "note": (
                    "different estimand than close-to-close residual; "
                    "do not compare lift vs +0.0290 as the same object"
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Causal trailing skip-IC shrink CS attack.")
    p.add_argument("--data-dir", default="")
    p.add_argument("--universe", default="liquid", choices=("", "liquid", "liquid_wide"))
    p.add_argument("--cache", default="/tmp/cs_lastbars44.npz")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument(
        "--also-labels",
        action="store_true",
        help="after shrink, rebuild overnight/session residual last bars",
    )
    p.add_argument("--out", default="/tmp/cs_shrink_ablate.json")
    args = p.parse_args(argv)

    cache_path = Path(args.cache)
    cache = _ensure_cache(
        cache_path,
        data_dir=args.data_dir,
        universe=args.universe,
        rebuild=args.rebuild,
        label_return="close",
    )
    shrink = _run_shrink_attack(cache)
    payload: dict[str, Any] = {
        "universe": args.universe,
        "n_features": int(cache["train_x"].shape[1]),
        "n_trade": cache.get("n_trade"),
        "promote_bar": PROMOTED_VAL_IC + VAL_LIFT,
        "val_2017_floor": VAL_2017_FLOOR,
        **shrink,
        "labels": [],
    }
    if args.also_labels:
        if not args.data_dir:
            print("--also-labels needs --data-dir", file=sys.stderr)
            return 2
        payload["labels"] = _run_label_attack(
            data_dir=args.data_dir,
            universe=args.universe,
            cache_stem=cache_path,
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_path}")
        if payload.get("promoted"):
            print(f"PROMOTE {payload['promoted']} on locked val; report its test in the PR.")
        else:
            print(
                "NO PROMOTE: trailing skip-IC shrink did not lift locked val by "
                f">={VAL_LIFT:.3f} with val-2017 recovery "
                f"(floor {VAL_2017_FLOOR:+.3f}, t>={VAL_2017_T})."
            )
            print(
                "Close-to-close last-bar daily residual CS did not reach 0.04–0.08. "
                "Keep IR~1 as the honest close-to-close book."
            )
            overnight = next(
                (r for r in payload.get("labels") or [] if r.get("label_return") == "overnight"),
                None,
            )
            if overnight:
                val = overnight["val"]
                test = overnight["test"]
                print(
                    "Overnight residual is a different estimand: "
                    f"val CS IC={val['cs_ic']:+.4f} t={val['cs_ic_tstat']:.2f} "
                    f"val2017={val['cs_ic_2017']:+.4f} t2017={val['cs_t_2017']:.2f} "
                    f"test={test['cs_ic']:+.4f} t={test['cs_ic_tstat']:.2f}. "
                    "Do not mix into the close-to-close headline."
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
