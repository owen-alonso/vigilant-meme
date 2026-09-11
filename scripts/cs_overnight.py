"""First-class overnight gap residual protocol (parallel to the promoted skip).

Locked calendar cuts match the close-to-close book. The trading object is
``log(open_{t+1}) - log(close_t)`` residual, not close-to-close. Do not mix
the two ICs. Promote only if locked-val improves honestly and test stays strong.

    python scripts/cs_overnight.py --synthetic
    python scripts/cs_overnight.py --data-dir data --universe liquid
    python scripts/cs_overnight.py --data-dir data --universe liquid --encoder
    python scripts/cs_overnight.py --data-dir data --universe liquid --try-dynamic-a
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
from forecast.config import (
    DataConfig,
    ForecastModelConfig,
    ForecastTrainConfig,
    interval_data_kwargs,
    interval_model_kwargs,
)
from forecast.data import FEATURE_NAMES, build_datasets
from forecast.overnight import (
    OVERNIGHT_FORMULA,
    formula_log_line,
    slim_cs_stats,
)
from forecast.ridge import (
    cs_stats,
    feature_mask,
    fit_ridge_xy,
    labelled_rows,
    year_cs_ics,
)
from forecast.synthetic import write_cs_overnight_universe

PROMOTED = dict(
    ridge=10.0,
    rank_target=True,
    feat_winsor=3.0,
    mask_mode="no_long_ts",
)
VAL_LIFT = 0.003
VAL_2017_FLOOR = 0.015
VAL_2017_T = 1.5
TEST_T_FLOOR = 3.0
# Prior overnight skip print (different book than close-to-close +0.0290).
PRIOR_ON_VAL = 0.0734


def _ymd(year: int, month: int = 1, day: int = 1) -> int:
    return int(
        (np.datetime64(f"{year:04d}-{month:02d}-{day:02d}") - np.datetime64("1970-01-01"))
        / np.timedelta64(1, "D")
    )


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
        cross_section_min_names=8 if universe in ("", "synthetic") else 30,
        allow_mixed_prices=universe in ("", "synthetic"),
        cs_zscore=True,
        universe="" if universe in ("", "synthetic") else universe,
        sector_residual=True,
        equities_only=universe not in ("", "synthetic"),
        train_from="" if universe in ("", "synthetic") else "1999-01-01",
        label_return="overnight",
    )


def _dump(bundle: dict[str, Any], path: Path) -> None:
    payload: dict[str, Any] = {
        "feature_mean": bundle["feature_mean"],
        "feature_std": bundle["feature_std"],
        "cs_min_names": np.array([bundle.get("cs_min_names", 30)]),
        "n_trade": np.array([bundle.get("n_trading_names", 0)]),
        "label_return": np.array(["overnight"]),
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


def _ensure_cache(
    path: Path,
    *,
    data_dir: str,
    universe: str,
    rebuild: bool,
) -> dict[str, Any]:
    if path.exists() and not rebuild:
        print(f"cache {path}", flush=True)
        return _load(path)
    if not data_dir:
        raise SystemExit("need --data-dir or --synthetic to build overnight last bars")
    print(
        f"building overnight last bars from {data_dir}  {formula_log_line('overnight')}",
        flush=True,
    )
    bundle = build_datasets(_cfg(data_dir, universe), log_fn=print)
    _dump(bundle, path)
    print(f"wrote {path}  n_trade={bundle.get('n_trading_names')}", flush=True)
    return _load(path)


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


def _split_stats(
    cache: dict[str, Any],
    w: np.ndarray,
    b: float,
) -> dict[str, Any]:
    min_names = int(cache["cs_min_names"])
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        x = cache[f"{split}_x"].astype(np.float64)
        y = cache[f"{split}_y"].astype(np.float64)
        d = cache[f"{split}_d"].astype(np.int64)
        pred = x @ w + b
        stats = cs_stats(pred, y, d, min_names=min_names)
        years = year_cs_ics(pred, y, d, min_names=min_names)
        y2017 = _year_row(years, 2017)
        stats["cs_ic_2017"] = float(y2017.get("cs_ic", float("nan")))
        stats["cs_t_2017"] = float(y2017.get("cs_ic_tstat", float("nan")))
        stats["years"] = years
        out[split] = stats
        out[f"{split}_pred"] = pred
        out[f"{split}_y"] = y
        out[f"{split}_d"] = d
    return out


def _winsor_cols(x: np.ndarray, k: float, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if k <= 0:
        return out
    for j in np.flatnonzero(np.asarray(mask, dtype=bool)):
        col = out[:, int(j)]
        if col.size < 3:
            continue
        s = float(col.std())
        if s < 1e-8:
            continue
        mu = float(col.mean())
        out[:, int(j)] = np.clip(col, mu - k * s, mu + k * s)
    return out


def _fit_lastbar_residual(
    cache: dict[str, Any],
    w: np.ndarray,
    b: float,
    *,
    hidden: int = 32,
    steps: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Tiny last-bar MLP residual on frozen skip. Same X as the overnight ridge.

    Sequence Mamba is a different (CUDA) ablation. This answers whether extra
    last-bar capacity lifts locked overnight val. Do not retarget from test.
    """
    import torch
    import torch.nn as nn

    mask = feature_mask(PROMOTED["mask_mode"])
    x_tr, y_tr, _d_tr = _train_xy(cache)
    x_tr = _winsor_cols(x_tr, float(PROMOTED["feat_winsor"]), mask)
    w64 = np.asarray(w, dtype=np.float64).reshape(-1)
    skip_tr = x_tr @ w64 + float(b)
    resid = y_tr - skip_tr
    keep = mask.astype(bool)
    xt = torch.from_numpy(x_tr[:, keep].astype(np.float32))
    yt = torch.from_numpy(resid.astype(np.float32))
    torch.manual_seed(int(seed))
    net = nn.Sequential(
        nn.Linear(int(keep.sum()), int(hidden)),
        nn.Tanh(),
        nn.Linear(int(hidden), 1),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
    min_names = int(cache["cs_min_names"])
    x_va = cache["val_x"].astype(np.float64)
    y_va = cache["val_y"].astype(np.float64)
    d_va = cache["val_d"].astype(np.int64)
    x_te = cache["test_x"].astype(np.float64)
    y_te = cache["test_y"].astype(np.float64)
    d_te = cache["test_d"].astype(np.int64)

    def _pred(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            extra = net(torch.from_numpy(x_np[:, keep].astype(np.float32))).squeeze(-1)
        return x_np @ w64 + float(b) + extra.numpy().astype(np.float64)

    net.eval()
    start_val = float(cs_stats(_pred(x_va), y_va, d_va, min_names=min_names)["cs_ic"])
    best_val = start_val if np.isfinite(start_val) else -1e9
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    n = int(xt.shape[0])
    batch = min(4096, max(256, n))
    for step in range(int(steps)):
        net.train()
        idx = torch.randint(0, n, (batch,))
        pred = net(xt[idx]).squeeze(-1)
        loss = torch.nn.functional.huber_loss(pred, yt[idx], delta=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0 or step + 1 == int(steps):
            net.eval()
            val_ic = float(cs_stats(_pred(x_va), y_va, d_va, min_names=min_names)["cs_ic"])
            if np.isfinite(val_ic) and val_ic > best_val:
                best_val = val_ic
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    val_stats = cs_stats(_pred(x_va), y_va, d_va, min_names=min_names)
    test_stats = cs_stats(_pred(x_te), y_te, d_te, min_names=min_names)
    return {
        "best_val_ic": float(val_stats.get("cs_ic", float("nan"))),
        "val": slim_cs_stats(val_stats),
        "test": slim_cs_stats(test_stats),
        "hidden": int(hidden),
        "steps": int(steps),
        "kind": "lastbar_mlp_residual",
    }


def _fit_promoted(cache: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    x, y, d = _train_xy(cache)
    mask = feature_mask(PROMOTED["mask_mode"])
    if x.shape[1] != mask.size:
        raise ValueError(
            f"cache has {x.shape[1]} features, expected {mask.size}. Rebuild the cache."
        )
    w, b, ic = fit_ridge_xy(
        x,
        y,
        d,
        ridge=PROMOTED["ridge"],
        min_names=int(cache["cs_min_names"]),
        cs_demean=True,
        rank_target=PROMOTED["rank_target"],
        feat_winsor=PROMOTED["feat_winsor"],
        feature_mask_bool=mask,
    )
    return w, b, ic


def _print_split(name: str, split: dict[str, Any]) -> None:
    val = split["val"]
    test = split["test"]
    print(
        f"{name}: val CS IC={val['cs_ic']:+.4f} t={val['cs_ic_tstat']:.2f} "
        f"n={int(val['cs_n_dates'])}  val2017={val['cs_ic_2017']:+.4f} "
        f"t2017={val['cs_t_2017']:.2f}  "
        f"test CS IC={test['cs_ic']:+.4f} t={test['cs_ic_tstat']:.2f} "
        f"n={int(test['cs_n_dates'])}",
        flush=True,
    )
    for row in val.get("years") or []:
        print(
            f"  val {int(row['year'])}: cs_ic={row['cs_ic']:+.4f} "
            f"t={row['cs_ic_tstat']:.2f} n={int(row['cs_n_dates'])}",
            flush=True,
        )
    for row in test.get("years") or []:
        print(
            f"  test {int(row['year'])}: cs_ic={row['cs_ic']:+.4f} "
            f"t={row['cs_ic_tstat']:.2f} n={int(row['cs_n_dates'])}",
            flush=True,
        )


def _wide(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last-bar vectors -> wide date x dummy-name frames for book_pnl."""
    keys = np.unique(dates)
    # Variable breadth: pad with NaN. Names are anonymous slots per date.
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


def _stress_grid(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> list[dict[str, Any]]:
    p, r = _wide(pred, y, dates)
    min_names = 8
    cases = [
        dict(name="overnight_10bp", round_trip_bps=10.0),
        dict(name="overnight_20bp", round_trip_bps=20.0),
        dict(name="overnight_40bp", round_trip_bps=40.0),
        dict(name="auction_10_plus_rt10", round_trip_bps=10.0, open_auction_bps=10.0),
        dict(name="borrow_5_plus_rt10", round_trip_bps=10.0, borrow_bps=5.0),
        dict(
            name="live_friction_bundle",
            round_trip_bps=20.0,
            open_auction_bps=10.0,
            borrow_bps=5.0,
            hedge_cost_bps=10.0,
        ),
        dict(name="long_only_10bp", round_trip_bps=10.0, long_only=True),
        dict(name="vol_target_1_toy", round_trip_bps=10.0, vol_target=1.0),
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        name = case.pop("name")
        stats = book_pnl(
            p,
            r,
            quantile=0.2,
            weighting="quantile",
            hold_halflife=0.0,
            causal_vol=True,
            vol_target=float(case.get("vol_target", 0.15)),
            lever_cap=3.0,
            min_names=min_names,
            holding="overnight",
            round_trip_bps=float(case.get("round_trip_bps", 10.0)),
            open_auction_bps=float(case.get("open_auction_bps", 0.0)),
            borrow_bps=float(case.get("borrow_bps", 0.0)),
            hedge_cost_bps=float(case.get("hedge_cost_bps", 0.0)),
            long_only=bool(case.get("long_only", False)),
        )
        row = {
            "name": name,
            "unlevered_net_ir": stats.get("unlevered_net_ir"),
            "levered_net_ir": stats.get("net_ir"),
            "unlevered_max_dd": stats.get("unlevered_max_dd"),
            "levered_max_dd": stats.get("max_dd"),
            "mean_turnover": stats.get("mean_turnover"),
            "mean_cs_ic": stats.get("mean_cs_ic"),
            "vol_target": stats.get("vol_target"),
            "long_only": stats.get("long_only"),
            "round_trip_bps": stats.get("round_trip_bps"),
            "open_auction_bps": stats.get("open_auction_bps"),
            "borrow_bps": stats.get("borrow_bps"),
            "hedge_cost_bps": stats.get("hedge_cost_bps"),
            "note": (
                "vol_target=1 is a toy; do not headline"
                if name == "vol_target_1_toy"
                else "overnight flatten (MOC->MOO); paper IR omits locate/auction"
            ),
        }
        rows.append(row)
        print(
            f"  stress {name}: unlev net IR={row['unlevered_net_ir']:+.3f} "
            f"lev net IR={row['levered_net_ir']:+.3f} "
            f"lev maxDD={row['levered_max_dd']:+.3f} "
            f"turn={row['mean_turnover']:.3f}",
            flush=True,
        )
        case["name"] = name  # restore if reused
    return rows


def _train_encoder(
    data_dir: str,
    universe: str,
    ckpt_dir: Path,
    *,
    dynamic_weights: bool,
    max_steps: int,
    skip_only: bool,
) -> dict[str, Any]:
    from forecast.training import train
    import torch

    ssm = interval_model_kwargs("daily")
    data_cfg = _cfg(data_dir, universe)
    # Last-bar CS skip does not need 128-bar windows. Keep the overnight
    # protocol seq_len so the encoder ablation is comparable and runnable.
    if universe not in ("", "synthetic"):
        data_cfg.cross_section_min_names = 30
        data_cfg.equities_only = True
        data_cfg.train_from = "1999-01-01"
        data_cfg.allow_mixed_prices = False
    model_cfg = ForecastModelConfig(
        n_features=len(FEATURE_NAMES),
        d_model=int(ssm.get("d_model", 32)),
        n_layer=1,
        d_state=int(ssm.get("d_state", 8)),
        expand=2,
        dropout=0.0,
        linear_skip=True,
        dynamic_weights=dynamic_weights,
        dt_min=ssm["dt_min"],
        dt_max=ssm["dt_max"],
    )
    train_cfg = ForecastTrainConfig(
        batch_size=8,
        epochs=1,
        max_steps=None if skip_only else int(max_steps),
        lr=1e-3,
        checkpoint_dir=str(ckpt_dir),
        skip_only=skip_only,
        ridge_skip=10.0,
        freeze_skip=True,
        ridge_cs_demean=True,
        ridge_rank_target=True,
        ridge_feat_winsor=3.0,
        ridge_features="no_long_ts",
        precision="fp32",
        early_stop_evals=6,
        eval_interval=max(10, int(max_steps) // 2) if not skip_only else 250,
        ic_loss_weight=2.0,
        rank_loss_weight=1.0,
        eval_train_split=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train(data_cfg, model_cfg, train_cfg, device=device, log_fn=print)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Overnight gap residual CS protocol.")
    p.add_argument("--data-dir", default="")
    p.add_argument(
        "--universe",
        default="liquid",
        choices=("", "liquid", "liquid_wide", "synthetic"),
    )
    p.add_argument("--synthetic", action="store_true", help="planted overnight CS universe")
    p.add_argument("--cache", default="/tmp/cs_overnight_lastbars.npz")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--out", default="/tmp/cs_overnight.json")
    p.add_argument(
        "--encoder",
        action="store_true",
        help="tiny frozen-skip Mamba residual on overnight y (val-gate vs skip)",
    )
    p.add_argument(
        "--try-dynamic-a",
        action="store_true",
        help="Dynamic A tiny encoder; promote only if locked val AND test lift vs skip",
    )
    p.add_argument("--encoder-steps", type=int, default=80)
    p.add_argument("--checkpoint-dir", default="checkpoints/forecast_overnight")
    p.add_argument(
        "--no-lastbar-residual",
        action="store_true",
        help="skip the cheap last-bar MLP residual (default: run it from the cache)",
    )
    args = p.parse_args(argv)

    if args.synthetic:
        args.universe = "synthetic"
        if not args.data_dir:
            args.data_dir = "/tmp/cs_overnight_synth"
        write_cs_overnight_universe(args.data_dir, n_names=12, n_days=220, seed=1, rho=0.65)
        print(f"synthetic overnight universe -> {args.data_dir}", flush=True)

    print(OVERNIGHT_FORMULA, flush=True)
    cache = _ensure_cache(
        Path(args.cache),
        data_dir=args.data_dir,
        universe=args.universe,
        rebuild=args.rebuild or args.synthetic,
    )
    w, b, train_ic = _fit_promoted(cache)
    split = _split_stats(cache, w, b)
    print(f"promoted skip overnight train CS IC={train_ic:+.4f}", flush=True)
    _print_split("skip-only overnight", split)

    val = split["val"]
    test = split["test"]
    skip_ok = (
        np.isfinite(val["cs_ic"])
        and float(val["cs_ic"]) >= 0.02
        and (not np.isfinite(val["cs_t_2017"]) or float(val["cs_t_2017"]) >= 0 or args.synthetic)
        and np.isfinite(test["cs_ic"])
        and float(test["cs_ic_tstat"]) >= TEST_T_FLOOR
    )
    payload: dict[str, Any] = {
        "formula": OVERNIGHT_FORMULA,
        "universe": args.universe,
        "n_features": int(cache["train_x"].shape[1]),
        "prior_overnight_val_cs_ic": PRIOR_ON_VAL,
        "promoted_skip": PROMOTED,
        "skip": {
            "train_cs_ic": float(train_ic),
            "val": slim_cs_stats(val)
            | {
                "cs_ic_2017": val["cs_ic_2017"],
                "cs_t_2017": val["cs_t_2017"],
            },
            "test": slim_cs_stats(test),
            "val_years": val.get("years"),
            "test_years": test.get("years"),
        },
        "stress": [],
        "lastbar_residual": None,
        "encoder": None,
        "dynamic_a": None,
        "promoted": None,
        "verdict": "",
    }
    if not args.no_lastbar_residual:
        print("tiny last-bar MLP residual (frozen skip, same overnight y) ...", flush=True)
        residual = _fit_lastbar_residual(cache, w, b)
        residual_val = float(residual.get("best_val_ic", float("nan")))
        residual_test = float((residual.get("test") or {}).get("cs_ic", float("nan")))
        payload["lastbar_residual"] = residual
        print(
            f"lastbar residual val CS IC={residual_val:+.4f}  "
            f"skip val={val['cs_ic']:+.4f}  test={residual_test:+.4f}",
            flush=True,
        )
        if np.isfinite(residual_val) and residual_val >= float(val["cs_ic"]) + VAL_LIFT:
            print("lastbar residual lifted locked overnight val.", flush=True)
        else:
            print(
                "NO PROMOTE lastbar residual: did not lift locked overnight val by "
                f"{VAL_LIFT:.3f}. Sequence Mamba is not the default next step.",
                flush=True,
            )
    print("overnight holding-period stress (locked test last bars):", flush=True)
    payload["stress"] = _stress_grid(split["test_pred"], split["test_y"], split["test_d"])

    if args.encoder or args.try_dynamic_a:
        if not args.data_dir:
            print("--encoder needs --data-dir / --synthetic", file=sys.stderr)
            return 2

    if args.encoder:
        enc_dir = Path(args.checkpoint_dir) / "encoder"
        print("tiny residual encoder (frozen skip, d_model=32 n_layer=1) ...", flush=True)
        enc = _train_encoder(
            args.data_dir,
            args.universe,
            enc_dir,
            dynamic_weights=False,
            max_steps=int(args.encoder_steps),
            skip_only=False,
        )
        payload["encoder"] = {
            "best_val_ic": enc.get("best_val_ic"),
            "test": slim_cs_stats(enc.get("test") or {}),
            "skip_only_val": slim_cs_stats(enc.get("skip_only_val") or {}),
            "device": enc.get("device"),
        }
        enc_val = float(enc.get("best_val_ic", float("nan")))
        print(
            f"encoder val CS IC={enc_val:+.4f}  skip val={val['cs_ic']:+.4f}  "
            f"test={float((enc.get('test') or {}).get('cs_ic', float('nan'))):+.4f}",
            flush=True,
        )
        if np.isfinite(enc_val) and enc_val >= float(val["cs_ic"]) + VAL_LIFT:
            print("encoder lifted locked val vs overnight skip.", flush=True)
        else:
            print("NO PROMOTE encoder: did not lift locked overnight val.", flush=True)

    if args.try_dynamic_a:
        dyn_dir = Path(args.checkpoint_dir) / "dynamic_a"
        print("Dynamic A tiny encoder (val-gate; do not retarget from test) ...", flush=True)
        dyn = _train_encoder(
            args.data_dir,
            args.universe,
            dyn_dir,
            dynamic_weights=True,
            max_steps=int(args.encoder_steps),
            skip_only=False,
        )
        dyn_val = float(dyn.get("best_val_ic", float("nan")))
        dyn_test = float((dyn.get("test") or {}).get("cs_ic", float("nan")))
        payload["dynamic_a"] = {
            "best_val_ic": dyn_val,
            "test": slim_cs_stats(dyn.get("test") or {}),
            "device": dyn.get("device"),
        }
        val_lift = np.isfinite(dyn_val) and dyn_val >= float(val["cs_ic"]) + VAL_LIFT
        test_lift = np.isfinite(dyn_test) and dyn_test >= float(test["cs_ic"]) + VAL_LIFT
        if val_lift and test_lift:
            print(
                f"Dynamic A lifted val ({dyn_val:+.4f}) and test ({dyn_test:+.4f}) vs skip. "
                "Still report as an ablation, not a bigger-Mamba default.",
                flush=True,
            )
        else:
            print(
                f"NO PROMOTE Dynamic A: val={dyn_val:+.4f} test={dyn_test:+.4f} "
                f"vs skip val={val['cs_ic']:+.4f} test={test['cs_ic']:+.4f}.",
                flush=True,
            )

    live = next((r for r in payload["stress"] if r["name"] == "live_friction_bundle"), None)
    paper = next((r for r in payload["stress"] if r["name"] == "overnight_10bp"), None)
    if skip_ok:
        payload["promoted"] = "overnight_skip"
        payload["verdict"] = (
            "Overnight skip is the overnight book (not the close-to-close headline). "
            "Paper 10bp flatten IR is not live P&L; auction/borrow/hedge overlay cut it."
        )
    else:
        payload["verdict"] = (
            "Overnight skip did not stay strong under the locked protocol. "
            "Do not promote. Next highest-EV estimand is not bigger Mamba: "
            "try a tradeable next-open *limit* fill (open+N minutes) or a weekly residual "
            "if overnight collapses under auction/borrow; keep close-to-close IR~1 as the "
            "honest close-to-close book."
        )
    if live and paper:
        payload["friction_note"] = (
            f"paper 10bp unlev net IR={paper['unlevered_net_ir']:+.3f}; "
            f"live-friction bundle unlev net IR={live['unlevered_net_ir']:+.3f} "
            f"(20bp RT + 10bp auction + 5bp borrow + 10bp hedge). "
            "Shorting overnight needs a locate; long-only is the no-borrow path."
        )
        print(payload["friction_note"], flush=True)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_path}", flush=True)
    print(payload["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
