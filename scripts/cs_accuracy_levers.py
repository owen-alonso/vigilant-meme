"""Val-gated overnight accuracy levers (post-PR#5).

Locked calendar cuts. Promote only if locked val CS IC lifts ≥0.003 and
val-2017 is not killed. Do not retarget from test/2023. Default overnight
``y`` stays causal MOC→MOO unless a separate estimand wins the gate.

    python scripts/cs_accuracy_levers.py --synthetic
    python scripts/cs_accuracy_levers.py --data-dir data --universe liquid
    python scripts/cs_accuracy_levers.py --data-dir data --universe liquid --rebuild-residuals
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
from forecast.data import FEATURE_NAMES, build_datasets
from forecast.levers import (
    blend_readouts,
    dead_ic_blend_weights,
    sign_consistency_weights,
    sleeve_row_mask,
)
from forecast.overnight import (
    LIVE_BUNDLE,
    LIVE_LONG_ONLY_BUNDLE,
    LIVE_MICRO_BUNDLE,
    LIVE_MICRO_LONG_ONLY_BUNDLE,
    VAL_2017_KEEP,
    VAL_LIFT,
    slim_cs_stats,
)
from forecast.ridge import (
    cs_stats,
    feature_mask,
    fit_ridge_xy,
    labelled_rows,
    trailing_skip_ic_stats,
    year_cs_ics,
)
from forecast.synthetic import write_cs_overnight_universe

PROMOTED = dict(ridge=10.0, rank_target=True, feat_winsor=3.0, mask_mode="no_long_ts")


def _ymd(year: int, month: int = 1, day: int = 1) -> int:
    return int(
        (np.datetime64(f"{year:04d}-{month:02d}-{day:02d}") - np.datetime64("1970-01-01"))
        / np.timedelta64(1, "D")
    )


def _cfg(data_dir: str, universe: str, **extra: Any) -> DataConfig:
    preset = interval_data_kwargs("daily")
    kw = dict(
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
        cross_section_min_names=8 if universe in ("", "synthetic") else 30,
        allow_mixed_prices=universe in ("", "synthetic"),
        cs_zscore=True,
        universe="" if universe in ("", "synthetic") else universe,
        sector_residual=True,
        equities_only=universe not in ("", "synthetic"),
        train_from="" if universe in ("", "synthetic") else "1999-01-01",
        label_return="overnight",
    )
    kw.update(extra)
    return DataConfig(**kw)


def _dump(bundle: dict[str, Any], path: Path) -> None:
    payload: dict[str, Any] = {
        "feature_mean": bundle["feature_mean"],
        "feature_std": bundle["feature_std"],
        "cs_min_names": np.array([bundle.get("cs_min_names", 30)]),
        "n_trade": np.array([bundle.get("n_trading_names", 0)]),
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


def _train_xy(cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = cache["train_d"].astype(np.int64) >= _ymd(1999)
    if not bool(keep.any()):
        keep = np.ones(len(cache["train_d"]), dtype=bool)
    return (
        cache["train_x"].astype(np.float64)[keep],
        cache["train_y"].astype(np.float64)[keep],
        cache["train_d"].astype(np.int64)[keep],
    )


def _year_row(rows: list[dict[str, Any]], year: int) -> dict[str, float]:
    for row in rows:
        if int(row.get("year", -1)) == int(year):
            return row
    return {"year": float(year), "cs_ic": float("nan"), "cs_ic_tstat": float("nan")}


def _split_stats(cache: dict[str, Any], pred_fn) -> dict[str, Any]:
    min_names = int(cache["cs_min_names"])
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        x = cache[f"{split}_x"].astype(np.float64)
        y = cache[f"{split}_y"].astype(np.float64)
        d = cache[f"{split}_d"].astype(np.int64)
        pred = pred_fn(x, d)
        stats = cs_stats(pred, y, d, min_names=min_names)
        years = year_cs_ics(pred, y, d, min_names=min_names)
        y2017 = _year_row(years, 2017)
        stats["cs_ic_2017"] = float(y2017.get("cs_ic", float("nan")))
        stats["years"] = years
        out[split] = stats
        out[f"{split}_pred"] = pred
        out[f"{split}_y"] = y
        out[f"{split}_d"] = d
    return out


def _fit(
    cache: dict[str, Any],
    *,
    long_only_quantile: float = 0.0,
    sign_shrink: bool = False,
    mask_mode: str | None = None,
    sleeve_floor: float = 0.0,
) -> tuple[np.ndarray, float, Any]:
    x, y, d = _train_xy(cache)
    mask = feature_mask(mask_mode or PROMOTED["mask_mode"])
    col_scale = None
    if sign_shrink:
        col_scale = sign_consistency_weights(x, y, d, min_names=int(cache["cs_min_names"]))
    if sleeve_floor > 0 and "turnover_z" in FEATURE_NAMES:
        tz = x[:, list(FEATURE_NAMES).index("turnover_z")]
        keep = sleeve_row_mask(tz, d, floor=sleeve_floor)
        if bool(keep.any()):
            x, y, d = x[keep], y[keep], d[keep]
    w, b, ic = fit_ridge_xy(
        x,
        y,
        d,
        ridge=float(PROMOTED["ridge"]),
        min_names=int(cache["cs_min_names"]),
        cs_demean=True,
        rank_target=True,
        feat_winsor=float(PROMOTED["feat_winsor"]),
        feature_mask_bool=mask,
        long_only_quantile=float(long_only_quantile or 0.0),
        col_scale=col_scale,
    )
    return w, b, ic


def _gate(val_ic: float, val_2017: float, skip_val: float, skip_2017: float) -> bool:
    lift = float(val_ic) - float(skip_val)
    keep_2017 = (not np.isfinite(val_2017)) or val_2017 >= float(VAL_2017_KEEP)
    worse_2017 = (
        np.isfinite(skip_2017)
        and np.isfinite(val_2017)
        and val_2017 < float(skip_2017) - 0.01
    )
    return bool(np.isfinite(val_ic) and lift >= float(VAL_LIFT) and keep_2017 and not worse_2017)


def _wide(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = np.unique(dates)
    max_n = max(int((dates == k).sum()) for k in keys)
    p = np.full((len(keys), max_n), np.nan)
    r = np.full((len(keys), max_n), np.nan)
    index = pd.to_datetime(keys.astype("datetime64[D]"))
    for i, key in enumerate(keys):
        sel = dates == key
        n = int(sel.sum())
        p[i, :n] = pred[sel]
        r[i, :n] = y[sel]
    cols = [f"N{j}" for j in range(max_n)]
    return pd.DataFrame(p, index=index, columns=cols), pd.DataFrame(r, index=index, columns=cols)


def _feat_wide(cache: dict[str, Any], split: str, name: str) -> pd.DataFrame | None:
    if name not in FEATURE_NAMES:
        return None
    x = cache[f"{split}_x"].astype(np.float64)
    d = cache[f"{split}_d"].astype(np.int64)
    keys = np.unique(d)
    max_n = max(int((d == k).sum()) for k in keys)
    p = np.full((len(keys), max_n), np.nan)
    index = pd.to_datetime(keys.astype("datetime64[D]"))
    col = x[:, list(FEATURE_NAMES).index(name)]
    for i, key in enumerate(keys):
        sel = d == key
        n = int(sel.sum())
        p[i, :n] = col[sel]
    return pd.DataFrame(p, index=index, columns=[f"N{j}" for j in range(max_n)])


def _book(pred, y, dates, cache, *, long_only: bool, bundle: dict[str, Any], **kw) -> dict[str, Any]:
    p, r = _wide(pred, y, dates)
    tz = _feat_wide(cache, "test", "turnover_z")
    vol = _feat_wide(cache, "test", "vol_level")
    return book_pnl(
        p,
        r,
        quantile=0.2,
        weighting="quantile",
        hold_halflife=0.0,
        causal_vol=True,
        vol_target=0.15,
        lever_cap=3.0,
        min_names=8,
        holding="overnight",
        long_only=long_only,
        round_trip_bps=float(bundle.get("round_trip_bps", 20.0)),
        moc_bps=float(bundle.get("moc_bps", 0.0)),
        moo_bps=float(bundle.get("moo_bps", 0.0)),
        borrow_bps=float(bundle.get("borrow_bps", 0.0)),
        hedge_cost_bps=float(bundle.get("hedge_cost_bps", 0.0)),
        impact_vol_k=float(bundle.get("impact_vol_k", 0.0)),
        thin_mult=float(bundle.get("thin_mult", 1.0)),
        thin_pctile=float(bundle.get("thin_pctile", 0.0)),
        borrow_thin_k=float(bundle.get("borrow_thin_k", 0.0)),
        impact_adv_k=float(bundle.get("impact_adv_k", 0.0)),
        turnover_z=tz,
        vol_level=vol,
        **kw,
    )


def _row(name: str, split: dict[str, Any], skip_val: float, skip_2017: float, **extra: Any) -> dict[str, Any]:
    val = split["val"]
    test = split["test"]
    val_ic = float(val["cs_ic"])
    val_2017 = float(val.get("cs_ic_2017", float("nan")))
    ok = _gate(val_ic, val_2017, skip_val, skip_2017)
    years = {int(r["year"]): r for r in (test.get("years") or [])}
    y2023 = years.get(2023, {})
    row = {
        "name": name,
        "val_cs_ic": val_ic,
        "val_2017": val_2017,
        "val_lift": val_ic - float(skip_val),
        "test_cs_ic": float(test["cs_ic"]),
        "test_2023": float(y2023.get("cs_ic", float("nan"))),
        "promote": bool(ok),
        **extra,
    }
    print(
        f"  {name}: val={val_ic:+.4f} lift={row['val_lift']:+.4f} "
        f"val2017={val_2017:+.4f} test={row['test_cs_ic']:+.4f} "
        f"2023={row['test_2023']:+.4f} {'PROMOTE' if ok else 'no'}",
        flush=True,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Val-gated overnight accuracy levers.")
    p.add_argument("--data-dir", default="")
    p.add_argument("--universe", default="liquid", choices=("", "liquid", "liquid_wide", "synthetic"))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--cache", default="/tmp/cs_accuracy_lastbars.npz")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--rebuild-residuals", action="store_true", help="rebuild last bars with size/peer residual flags")
    p.add_argument("--out", default="/tmp/cs_accuracy_levers.json")
    args = p.parse_args(argv)

    if args.synthetic:
        args.universe = "synthetic"
        if not args.data_dir:
            args.data_dir = "/tmp/cs_accuracy_synth"
        if args.cache == "/tmp/cs_accuracy_lastbars.npz":
            args.cache = "/tmp/cs_accuracy_synth_lastbars.npz"
        write_cs_overnight_universe(args.data_dir, n_names=12, n_days=240, seed=3, rho=0.65)
        print(f"synthetic overnight universe -> {args.data_dir}", flush=True)

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.rebuild:
        print(f"cache {cache_path}", flush=True)
        cache = _load(cache_path)
    else:
        if not args.data_dir:
            raise SystemExit("need --data-dir or --synthetic")
        print(f"building last bars from {args.data_dir}", flush=True)
        bundle = build_datasets(_cfg(args.data_dir, args.universe), log_fn=print)
        _dump(bundle, cache_path)
        cache = _load(cache_path)

    w, b, train_ic = _fit(cache)
    skip = _split_stats(cache, lambda x, _d: x @ w + b)
    skip_val = float(skip["val"]["cs_ic"])
    skip_2017 = float(skip["val"]["cs_ic_2017"])
    print(f"promoted skip overnight train CS IC={train_ic:+.4f} val={skip_val:+.4f}", flush=True)

    rows: list[dict[str, Any]] = []
    print("accuracy levers (val-gate; not test/2023):", flush=True)

    w_sh, b_sh, _ = _fit(cache, sign_shrink=True)
    rows.append(
        _row(
            "sign_shrink",
            _split_stats(cache, lambda x, _d, w=w_sh, b=b_sh: x @ w + b),
            skip_val,
            skip_2017,
        )
    )

    w_lo, b_lo, _ = _fit(cache, long_only_quantile=0.2)
    rows.append(
        _row(
            "long_only_objective",
            _split_stats(cache, lambda x, _d, w=w_lo, b=b_lo: x @ w + b),
            skip_val,
            skip_2017,
        )
    )

    w_sl, b_sl, _ = _fit(cache, sleeve_floor=2.0 / 3.0)
    rows.append(
        _row(
            "liquid_sleeve_skip",
            _split_stats(cache, lambda x, _d, w=w_sl, b=b_sl: x @ w + b),
            skip_val,
            skip_2017,
        )
    )

    def _ens(x, d, w_a=w, b_a=b, w_b=w_sl, b_b=b_sl):
        return blend_readouts([x @ w_a + b_a, x @ w_b + b_b], d)

    rows.append(_row("ensemble_skip_sleeve", _split_stats(cache, _ens), skip_val, skip_2017))

    def _regime(x, d, w_a=w, b_a=b, w_b=w_sl, b_b=b_sl):
        primary = x @ w_a + b_a
        alt = x @ w_b + b_b
        y_split = cache["val_y"] if np.array_equal(d, cache["val_d"]) else cache["test_y"]
        keys, _mu, tt, _n = trailing_skip_ic_stats(
            primary, y_split.astype(np.float64), d, lookback_days=63, min_names=8, min_obs=10
        )
        tt_row = np.ones(d.shape[0], dtype=np.float64)
        if keys.size:
            loc = np.searchsorted(keys, d.astype(np.int64))
            loc = np.clip(loc, 0, keys.size - 1)
            ok = keys[loc] == d.astype(np.int64)
            tt_row[ok] = tt[loc[ok]]
        mix = dead_ic_blend_weights(tt_row, dead_t=1.0, dead_mix=0.75)
        z1 = blend_readouts([primary], d)
        z2 = blend_readouts([alt], d)
        return (1.0 - mix) * z1 + mix * z2

    rows.append(_row("regime_dead_ic_blend", _split_stats(cache, _regime), skip_val, skip_2017))

    residual_rows: list[dict[str, Any]] = []
    if args.rebuild_residuals and args.data_dir:
        for name, extra in (
            ("double_residual", dict(double_residual=True)),
            ("size_residual", dict(size_residual=True)),
            ("peer_residual", dict(peer_residual=True)),
            ("size_peer_residual", dict(size_residual=True, peer_residual=True)),
        ):
            print(f"rebuild {name} ...", flush=True)
            try:
                bundle = build_datasets(_cfg(args.data_dir, args.universe, **extra), log_fn=None)
            except Exception as exc:
                residual_rows.append({"name": name, "error": str(exc), "promote": False})
                print(f"  {name}: failed {exc}", flush=True)
                continue
            path = Path(str(args.cache) + f".{name}.npz")
            _dump(bundle, path)
            rc = _load(path)
            rw, rb, _ = _fit(rc)
            residual_rows.append(
                _row(name, _split_stats(rc, lambda x, _d, w=rw, b=rb: x @ w + b), skip_val, skip_2017)
            )

    print("live long-only books (sizing / micro; IC may be unchanged):", flush=True)
    books = {}
    for bname, bundle, lo, extra in (
        ("live_long_only", LIVE_LONG_ONLY_BUNDLE, True, {}),
        ("live_micro_long_only", LIVE_MICRO_LONG_ONLY_BUNDLE, True, {}),
        (
            "live_long_only_ic_shrink",
            LIVE_LONG_ONLY_BUNDLE,
            True,
            dict(ic_shrink_lookback=63, gap_risk_cap=0.2, gap_vol_k=0.5),
        ),
        ("live_ls", LIVE_BUNDLE, False, {}),
        ("live_micro_ls", LIVE_MICRO_BUNDLE, False, {}),
    ):
        stats = _book(
            skip["test_pred"], skip["test_y"], skip["test_d"], cache, long_only=lo, bundle=bundle, **extra
        )
        books[bname] = {
            "unlevered_net_ir": stats.get("unlevered_net_ir"),
            "levered_net_ir": stats.get("net_ir"),
            "levered_max_dd": stats.get("max_dd"),
            "mean_cs_ic": stats.get("mean_cs_ic"),
        }
        print(
            f"  {bname}: unlev IR={books[bname]['unlevered_net_ir']:+.3f} "
            f"lev IR={books[bname]['levered_net_ir']:+.3f} "
            f"maxDD={books[bname]['levered_max_dd']:+.3f}",
            flush=True,
        )

    promoted = [r["name"] for r in rows + residual_rows if r.get("promote")]
    payload = {
        "skip": {
            "train_cs_ic": float(train_ic),
            "val": slim_cs_stats(skip["val"]),
            "val_2017": skip_2017,
            "test": slim_cs_stats(skip["test"]),
        },
        "levers": rows,
        "residuals": residual_rows,
        "books": books,
        "promoted": promoted,
        "verdict": (
            f"Val-gated promotes: {promoted or 'none'}. "
            "Default overnight skip / live long-only stays unless a lever cleared ≥0.003 "
            "locked-val lift without killing val-2017. Sizing overlays do not retarget IC."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {args.out}", flush=True)
    print(payload["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
