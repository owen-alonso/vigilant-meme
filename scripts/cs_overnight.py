"""First-class overnight gap residual protocol (parallel to the promoted skip).

Locked calendar cuts match the close-to-close book. The trading object is
``log(open_{t+1}) - log(close_t)`` residual, not close-to-close. Do not mix
the two ICs. Promote only if locked-val improves honestly and test stays strong.

    python scripts/cs_overnight.py --synthetic
    python scripts/cs_overnight.py --data-dir data --universe liquid
    python scripts/cs_overnight.py --data-dir data --universe liquid --try-fill 15
    python scripts/cs_overnight.py --data-dir data --universe liquid --encoder
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
    FILL_FORMULA,
    LIVE_BUNDLE,
    LIVE_FLAT_BUNDLE,
    LIVE_LOCATE_BUNDLE,
    LIVE_LONG_ONLY_BUNDLE,
    HARSH_BUNDLE,
    EX_POST_GAP_BUNDLE,
    FILL_LIVE_BUNDLE,
    OVERNIGHT_FORMULA,
    PAPER_BUNDLE,
    VAL_2017_KEEP,
    VAL_LIFT,
    formula_log_line,
    slim_cs_stats,
)
from forecast.ridge import (
    cs_stats,
    feature_mask,
    fit_ridge_xy,
    labelled_rows,
    year_cs_ics,
    year_feature_ics,
    year_stable_mask,
)
from forecast.synthetic import write_cs_overnight_universe

PROMOTED = dict(
    ridge=10.0,
    rank_target=True,
    feat_winsor=3.0,
    mask_mode="no_long_ts",
)
VAL_2017_T = 1.5
TEST_T_FLOOR = 3.0
# Prior overnight skip print (different book than close-to-close +0.0290).
PRIOR_ON_VAL = 0.0734


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
    interval: str = "daily",
) -> DataConfig:
    preset = interval_data_kwargs(interval)
    seq_len = 32 if interval == "daily" else int(preset["seq_len"])
    return DataConfig(
        data_dir=data_dir,
        interval=interval,
        horizon=1,
        seq_len=seq_len,
        stride=1,
        min_context=8 if interval == "daily" else int(preset["min_context"]),
        warmup_bars=16 if interval == "daily" else int(preset["warmup_bars"]),
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
        label_return=label_return,
        fill_minutes=int(fill_minutes or 0),
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
    label_return: str = "overnight",
    fill_minutes: int = 0,
    interval: str = "daily",
) -> dict[str, Any]:
    if path.exists() and not rebuild:
        print(f"cache {path}", flush=True)
        return _load(path)
    if not data_dir:
        raise SystemExit("need --data-dir or --synthetic to build overnight last bars")
    cfg = _cfg(
        data_dir,
        universe,
        label_return=label_return,
        fill_minutes=fill_minutes,
        interval=interval,
    )
    print(
        f"building last bars from {data_dir}  {formula_log_line(label_return, fill_minutes=fill_minutes)}",
        flush=True,
    )
    bundle = build_datasets(cfg, log_fn=print)
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


def _fit_skip(
    cache: dict[str, Any],
    *,
    ridge: float | None = None,
    rank_target: bool | None = None,
    feat_winsor: float | None = None,
    mask_mode: str | None = None,
    date_halflife: float = 0.0,
    drop_disp_q: float = 0.0,
    train_from_year: int | None = None,
    year_stable: bool = False,
) -> tuple[np.ndarray, float, float]:
    x, y, d = _train_xy(cache)
    if train_from_year is not None:
        keep = d >= _ymd(int(train_from_year))
        if bool(keep.any()):
            x, y, d = x[keep], y[keep], d[keep]
    mask = feature_mask(mask_mode or PROMOTED["mask_mode"])
    if year_stable:
        extra = year_stable_mask(x, y, d, min_names=int(cache["cs_min_names"]))
        mask = mask & extra
    if x.shape[1] != mask.size:
        raise ValueError(
            f"cache has {x.shape[1]} features, expected {mask.size}. Rebuild the cache."
        )
    w, b, ic = fit_ridge_xy(
        x,
        y,
        d,
        ridge=float(PROMOTED["ridge"] if ridge is None else ridge),
        min_names=int(cache["cs_min_names"]),
        cs_demean=True,
        rank_target=PROMOTED["rank_target"] if rank_target is None else bool(rank_target),
        feat_winsor=float(PROMOTED["feat_winsor"] if feat_winsor is None else feat_winsor),
        feature_mask_bool=mask,
        date_halflife=float(date_halflife),
        drop_disp_q=float(drop_disp_q),
    )
    return w, b, ic


def _fit_promoted(cache: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    return _fit_skip(cache)


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


def _wide_vec(values: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    keys = np.unique(dates)
    max_n = max(int((dates == k).sum()) for k in keys)
    p = np.full((len(keys), max_n), np.nan)
    index = pd.to_datetime(keys.astype("datetime64[D]"))
    for i, key in enumerate(keys):
        sel = dates == key
        n = int(sel.sum())
        p[i, :n] = values[sel]
    cols = [f"N{j}" for j in range(max_n)]
    return pd.DataFrame(p, index=index, columns=cols)


def _wide(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last-bar vectors -> wide date x dummy-name frames for book_pnl."""
    return _wide_vec(pred, dates), _wide_vec(y, dates)


def _feat_wide(cache: dict[str, Any], split: str, name: str) -> pd.DataFrame:
    names = list(FEATURE_NAMES)
    if name not in names:
        raise KeyError(name)
    x = cache[f"{split}_x"].astype(np.float64)
    d = cache[f"{split}_d"].astype(np.int64)
    return _wide_vec(x[:, names.index(name)], d)


def _book_row(stats: dict[str, Any], name: str, note: str) -> dict[str, Any]:
    return {
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
        "moc_bps": stats.get("moc_bps"),
        "moo_bps": stats.get("moo_bps"),
        "borrow_bps": stats.get("borrow_bps"),
        "hedge_cost_bps": stats.get("hedge_cost_bps"),
        "impact_vol_k": stats.get("impact_vol_k"),
        "thin_mult": stats.get("thin_mult"),
        "thin_pctile": stats.get("thin_pctile"),
        "locate_pctile": stats.get("locate_pctile"),
        "ex_post_gap_k": stats.get("ex_post_gap_k"),
        "mean_long_nav": stats.get("mean_long_nav"),
        "mean_short_nav": stats.get("mean_short_nav"),
        "mean_shorts_blocked": stats.get("mean_shorts_blocked"),
        "mean_cost_unlev": stats.get("mean_cost_unlev"),
        "cost_parts": stats.get("cost_parts"),
        "capacity_note": stats.get("capacity_note"),
        "note": note,
    }


def _run_book(
    pred: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    long_only: bool = False,
    holding: str = "overnight",
    turnover_z: pd.DataFrame | None = None,
    vol_level: pd.DataFrame | None = None,
    vol_target: float = 0.15,
    adv_floor_pctile: float = 0.0,
) -> dict[str, Any]:
    return book_pnl(
        pred,
        realized,
        quantile=0.2,
        weighting="quantile",
        hold_halflife=0.0,
        causal_vol=True,
        vol_target=float(vol_target),
        lever_cap=3.0,
        min_names=8,
        holding=holding,
        long_only=long_only,
        round_trip_bps=float(bundle.get("round_trip_bps", 10.0)),
        open_auction_bps=float(bundle.get("open_auction_bps", 0.0)),
        borrow_bps=float(bundle.get("borrow_bps", 0.0)),
        hedge_cost_bps=float(bundle.get("hedge_cost_bps", 0.0)),
        moc_bps=float(bundle.get("moc_bps", 0.0)),
        moo_bps=float(bundle.get("moo_bps", 0.0)),
        session_exit_bps=float(bundle.get("session_exit_bps", 0.0)),
        impact_vol_k=float(bundle.get("impact_vol_k", 0.0)),
        thin_mult=float(bundle.get("thin_mult", 1.0)),
        thin_pctile=float(bundle.get("thin_pctile", 0.0)),
        locate_pctile=float(bundle.get("locate_pctile", 0.0)),
        ex_post_gap_k=float(bundle.get("ex_post_gap_k", 0.0)),
        turnover_z=turnover_z,
        vol_level=vol_level,
        adv_floor_pctile=float(adv_floor_pctile),
    )


def _stress_grid(
    pred: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    turnover_z: pd.DataFrame | None = None,
    vol_level: pd.DataFrame | None = None,
    holding: str = "overnight",
) -> list[dict[str, Any]]:
    p, r = _wide(pred, y, dates)
    cases: list[tuple[str, dict[str, Any], bool, str]] = [
        ("overnight_10bp", PAPER_BUNDLE, False, "paper flatten; understates auction/locate"),
        ("overnight_20bp", {**PAPER_BUNDLE, "round_trip_bps": 20.0}, False, "flat 20bp RT"),
        ("overnight_40bp", {**PAPER_BUNDLE, "round_trip_bps": 40.0}, False, "flat 40bp RT"),
        (
            "live_flat",
            LIVE_FLAT_BUNDLE,
            False,
            "old overlay: 20bp RT + 10bp exit-half auction + 5 borrow + 10 hedge",
        ),
        (
            "live",
            LIVE_BUNDLE,
            False,
            "name-level MOC/MOO + thin/vol impact; shorts unconstrained",
        ),
        (
            "live_locate",
            LIVE_LOCATE_BUNDLE,
            False,
            "live + cannot short bottom 30% CS turnover_z",
        ),
        (
            "live_long_only",
            LIVE_LONG_ONLY_BUNDLE,
            True,
            "live costs, no shorts, no locate, borrow=0; ETF hedge overlay remains",
        ),
        (
            "harsh_auction",
            HARSH_BUNDLE,
            False,
            "ugly MOO / HTB / impact stress. If this dies, say so.",
        ),
        (
            "live_ex_post_gap",
            EX_POST_GAP_BUNDLE,
            False,
            "SENSITIVITY: extra k*|realized gap|*|w| (not the default live book)",
        ),
        (
            "long_only_10bp",
            PAPER_BUNDLE,
            True,
            "paper long-only; no locate",
        ),
        (
            "vol_target_1_toy",
            PAPER_BUNDLE,
            False,
            "vol_target=1 is a toy; do not headline",
        ),
        (
            "live_liquid_sleeve",
            LIVE_BUNDLE,
            False,
            "live costs, top CS turnover_z tercile only (val-gated sleeve, same skip w)",
            2.0 / 3.0,
        ),
        (
            "live_liquid_sleeve_long_only",
            LIVE_LONG_ONLY_BUNDLE,
            True,
            "liquid sleeve + long-only (no locate)",
            2.0 / 3.0,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        name, bundle, long_only, note = case[0], case[1], case[2], case[3]
        floor = float(case[4]) if len(case) > 4 else 0.0
        stats = _run_book(
            p,
            r,
            bundle=bundle,
            long_only=long_only,
            holding=holding,
            turnover_z=turnover_z,
            vol_level=vol_level,
            vol_target=1.0 if name == "vol_target_1_toy" else 0.15,
            adv_floor_pctile=floor,
        )
        row = _book_row(stats, name, note)
        row["adv_floor_pctile"] = floor
        rows.append(row)
        print(
            f"  stress {name}: unlev net IR={row['unlevered_net_ir']:+.3f} "
            f"lev net IR={row['levered_net_ir']:+.3f} "
            f"lev maxDD={row['levered_max_dd']:+.3f} "
            f"turn={row['mean_turnover']:.3f} "
            f"long={row['mean_long_nav']:.2f} short={row['mean_short_nav']:.2f}",
            flush=True,
        )
    return rows


def _ic_by_tercile(
    pred: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    z: np.ndarray,
    *,
    min_names: int,
) -> list[dict[str, float]]:
    """CS IC inside low/mid/high terciles of a known-at-t feature (e.g. turnover_z)."""
    rows: list[dict[str, float]] = []
    ics = [[], [], []]
    for key in np.unique(dates):
        sel = dates == key
        if int(sel.sum()) < int(min_names):
            continue
        zz = z[sel]
        pp = pred[sel]
        yy = y[sel]
        finite = np.isfinite(zz)
        if int(finite.sum()) < int(min_names):
            continue
        cuts = np.nanpercentile(zz[finite], [100.0 / 3.0, 200.0 / 3.0])
        buckets = [
            finite & (zz <= cuts[0]),
            finite & (zz > cuts[0]) & (zz <= cuts[1]),
            finite & (zz > cuts[1]),
        ]
        from forecast.training import _pearson

        for i, mask in enumerate(buckets):
            if int(mask.sum()) < 3:
                continue
            val = _pearson(pp[mask], yy[mask])
            if np.isfinite(val):
                ics[i].append(float(val))
    labels = ("low", "mid", "high")
    for i, lab in enumerate(labels):
        arr = np.asarray(ics[i], dtype=np.float64)
        rows.append(
            {
                "tercile": lab,
                "cs_ic": float(arr.mean()) if arr.size else float("nan"),
                "n_dates": float(arr.size),
            }
        )
    return rows


def _sleeve_min_names(min_names: int, floor: float) -> int:
    """Protocol min_names scaled to the remaining CS after the turnover floor.

    A 30-name liquid tape with floor=2/3 keeps ~1/3 of names. Evaluating that
    sleeve at min_names=30 drops every date and prints a fake NaN IC.
    """
    remain = max(0.05, 1.0 - float(floor))
    return max(8, int(round(float(min_names) * remain)))


def _sleeve_cs(
    pred: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    turnover_z: np.ndarray,
    *,
    floor: float,
    min_names: int,
) -> dict[str, float]:
    """CS IC on names at/above the within-date turnover_z percentile (known at t)."""
    keep = np.zeros(pred.shape[0], dtype=bool)
    p = float(floor)
    for key in np.unique(dates):
        sel = dates == key
        row = turnover_z[sel]
        finite = np.isfinite(row)
        if int(finite.sum()) < 5:
            continue
        cut = float(np.nanpercentile(row[finite], 100.0 * p))
        local = np.zeros(int(sel.sum()), dtype=bool)
        local[finite] = row[finite] >= cut
        keep[sel] = local
    if not bool(keep.any()):
        return {"cs_ic": float("nan"), "cs_ic_tstat": float("nan"), "cs_n_dates": 0.0}
    eval_min = _sleeve_min_names(int(min_names), p)
    return slim_cs_stats(cs_stats(pred[keep], y[keep], dates[keep], min_names=eval_min))


def _diagnose_years(
    cache: dict[str, Any],
    w: np.ndarray,
    b: float,
    split: dict[str, Any],
) -> dict[str, Any]:
    """2023 / year-stability diagnostics. Reported after the fact; not a promote knob."""
    min_names = int(cache["cs_min_names"])
    names = list(FEATURE_NAMES)
    out: dict[str, Any] = {"val_years": split["val"].get("years"), "test_years": split["test"].get("years")}
    test_years = {int(r["year"]): r for r in (split["test"].get("years") or [])}
    y2023 = test_years.get(2023, {})
    out["test_2023"] = {
        "cs_ic": float(y2023.get("cs_ic", float("nan"))),
        "cs_ic_tstat": float(y2023.get("cs_ic_tstat", float("nan"))),
        "cs_n_dates": float(y2023.get("cs_n_dates", float("nan"))),
        "dead": bool(
            np.isfinite(y2023.get("cs_ic", float("nan")))
            and abs(float(y2023.get("cs_ic_tstat", 0) or 0)) < 1.5
        ),
    }
    x_va = cache["val_x"].astype(np.float64)
    y_va = cache["val_y"].astype(np.float64)
    d_va = cache["val_d"].astype(np.int64)
    x_te = cache["test_x"].astype(np.float64)
    y_te = cache["test_y"].astype(np.float64)
    d_te = cache["test_d"].astype(np.int64)
    pred_te = split["test_pred"]
    val_uni = year_feature_ics(x_va, y_va, d_va, min_names=min_names)
    te_yr = year_feature_ics(x_te, y_te, d_te, min_names=min_names)
    if 2023 in te_yr:
        u23 = te_yr[2023]
        # mean val uni IC across val years
        if val_uni:
            stacked = np.stack(list(val_uni.values()), axis=0)
            finite = np.isfinite(stacked)
            counts = finite.sum(axis=0)
            sums = np.where(finite, stacked, 0.0).sum(axis=0)
            val_mean = np.full(stacked.shape[1], np.nan, dtype=np.float64)
            ok = counts > 0
            val_mean[ok] = sums[ok] / counts[ok]
        else:
            val_mean = np.full(u23.shape, np.nan)
        flips = []
        for j, name in enumerate(names):
            a = float(val_mean[j]) if j < val_mean.size else float("nan")
            b_ic = float(u23[j]) if j < u23.size else float("nan")
            if np.isfinite(a) and np.isfinite(b_ic) and a * b_ic < 0 and abs(a) >= 0.01:
                flips.append({"feature": name, "val_cs_ic": a, "y2023_cs_ic": b_ic})
        flips.sort(key=lambda r: abs(r["val_cs_ic"]), reverse=True)
        out["feature_sign_flips_2023_vs_val"] = flips[:12]
        print("  2023 feature sign flips vs val (known-at-t columns):", flush=True)
        if not flips:
            print("    none with |val CS IC|>=0.01", flush=True)
        for row in flips[:8]:
            print(
                f"    {row['feature']}: val={row['val_cs_ic']:+.4f}  2023={row['y2023_cs_ic']:+.4f}",
                flush=True,
            )
    if "turnover_z" in names:
        tz = x_te[:, names.index("turnover_z")]
        yr = (np.datetime64("1970-01-01") + d_te.astype("timedelta64[D]")).astype("datetime64[Y]").astype(int) + 1970
        mask_23 = yr == 2023
        if bool(mask_23.any()):
            terc = _ic_by_tercile(
                pred_te[mask_23], y_te[mask_23], d_te[mask_23], tz[mask_23], min_names=max(5, min_names // 4)
            )
            out["test_2023_ic_by_turnover_tercile"] = terc
            print("  2023 CS IC by turnover_z tercile (low=thin):", flush=True)
            for row in terc:
                print(
                    f"    {row['tercile']}: cs_ic={row['cs_ic']:+.4f} n={int(row['n_dates'])}",
                    flush=True,
                )
        terc_val = _ic_by_tercile(
            split["val_pred"], y_va, d_va, x_va[:, names.index("turnover_z")], min_names=max(5, min_names // 4)
        )
        out["val_ic_by_turnover_tercile"] = terc_val
    print(
        f"  test 2023 CS IC={out['test_2023']['cs_ic']:+.4f} "
        f"t={out['test_2023']['cs_ic_tstat']:.2f} "
        f"dead={out['test_2023']['dead']}",
        flush=True,
    )
    return out


def _stability_ablate(
    cache: dict[str, Any],
    skip_val: float,
    skip_val_2017: float,
) -> dict[str, Any]:
    """Causal skip variants. Promote only on locked val; do not retarget 2023/test."""
    candidates = [
        ("recency_2y", dict(date_halflife=504.0)),
        ("recency_5y", dict(date_halflife=1260.0)),
        ("train_from_2009", dict(train_from_year=2009)),
        ("train_from_2012", dict(train_from_year=2012)),
        ("no_ohlc", dict(mask_mode="no_ohlc")),
        ("core_cs", dict(mask_mode="core")),
        ("drop_disp_5", dict(drop_disp_q=0.05)),
        ("feat_winsor_5", dict(feat_winsor=5.0)),
        ("year_stable_train", dict(year_stable=True)),
    ]
    rows: list[dict[str, Any]] = []
    promoted: str | None = None
    best_val = float(skip_val)
    print("overnight year-stability ablations (val-gate; not test/2023):", flush=True)
    for name, kwargs in candidates:
        w, b, train_ic = _fit_skip(cache, **kwargs)
        split = _split_stats(cache, w, b)
        val = split["val"]
        test = split["test"]
        val_ic = float(val["cs_ic"])
        val_2017 = float(val["cs_ic_2017"])
        test_years = {int(r["year"]): r for r in (test.get("years") or [])}
        y2023 = test_years.get(2023, {})
        lift = val_ic - float(skip_val)
        keep_2017 = (not np.isfinite(val_2017)) or val_2017 >= float(VAL_2017_KEEP)
        ok = (
            np.isfinite(val_ic)
            and lift >= float(VAL_LIFT)
            and keep_2017
            and not (np.isfinite(skip_val_2017) and np.isfinite(val_2017) and val_2017 < float(skip_val_2017) - 0.01)
        )
        row = {
            "name": name,
            "train_cs_ic": float(train_ic),
            "val_cs_ic": val_ic,
            "val_t": float(val["cs_ic_tstat"]),
            "val_2017": val_2017,
            "test_cs_ic": float(test["cs_ic"]),
            "test_t": float(test["cs_ic_tstat"]),
            "test_2023": float(y2023.get("cs_ic", float("nan"))),
            "test_2023_t": float(y2023.get("cs_ic_tstat", float("nan"))),
            "val_lift": float(lift),
            "promote": bool(ok),
            "kwargs": {k: (v if not isinstance(v, (np.generic,)) else float(v)) for k, v in kwargs.items()},
        }
        rows.append(row)
        print(
            f"  {name}: val={val_ic:+.4f} lift={lift:+.4f} val2017={val_2017:+.4f} "
            f"test={float(test['cs_ic']):+.4f} 2023={row['test_2023']:+.4f} "
            f"{'PROMOTE' if ok else 'no'}",
            flush=True,
        )
        if ok and val_ic > best_val:
            best_val = val_ic
            promoted = name
    return {"candidates": rows, "promoted": promoted}


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
    p.add_argument(
        "--no-stability",
        action="store_true",
        help="skip causal year-stability skip variants (val-gated; default: run)",
    )
    p.add_argument(
        "--try-fill",
        type=int,
        default=0,
        metavar="N",
        help="val-gate open+N minute fill as a *separate* estimand (does not replace overnight y)",
    )
    p.add_argument(
        "--try-weekly",
        action="store_true",
        help="val-gate weekly close-to-close residual as a fallback estimand (needs --data-dir)",
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
        "year_diagnosis": None,
        "stability": None,
        "liquid_sleeve": None,
        "fill": None,
        "weekly": None,
        "promoted": None,
        "verdict": "",
        "next_estimand": None,
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
    try:
        tz = _feat_wide(cache, "test", "turnover_z")
        vol = _feat_wide(cache, "test", "vol_level")
    except Exception:
        tz, vol = None, None
    payload["stress"] = _stress_grid(
        split["test_pred"],
        split["test_y"],
        split["test_d"],
        turnover_z=tz,
        vol_level=vol,
        holding="overnight",
    )
    print("overnight year diagnosis (report only; do not retarget from 2023):", flush=True)
    payload["year_diagnosis"] = _diagnose_years(cache, w, b, split)
    names = list(FEATURE_NAMES)
    tz_i = names.index("turnover_z") if "turnover_z" in names else None
    if tz_i is not None:
        floor = 2.0 / 3.0
        min_names = int(cache["cs_min_names"])
        sleeve_val = _sleeve_cs(
            split["val_pred"], split["val_y"], split["val_d"],
            cache["val_x"].astype(np.float64)[:, tz_i],
            floor=floor, min_names=min_names,
        )
        yr = (np.datetime64("1970-01-01") + split["val_d"].astype("timedelta64[D]")).astype("datetime64[Y]").astype(int) + 1970
        m2017 = yr == 2017
        sleeve_2017 = _sleeve_cs(
            split["val_pred"][m2017], split["val_y"][m2017], split["val_d"][m2017],
            cache["val_x"].astype(np.float64)[m2017, tz_i],
            floor=floor, min_names=min_names,
        ) if bool(m2017.any()) else {"cs_ic": float("nan")}
        sleeve_test = _sleeve_cs(
            split["test_pred"], split["test_y"], split["test_d"],
            cache["test_x"].astype(np.float64)[:, tz_i],
            floor=floor, min_names=min_names,
        )
        yr_te = (np.datetime64("1970-01-01") + split["test_d"].astype("timedelta64[D]")).astype("datetime64[Y]").astype(int) + 1970
        m23 = yr_te == 2023
        sleeve_2023 = _sleeve_cs(
            split["test_pred"][m23], split["test_y"][m23], split["test_d"][m23],
            cache["test_x"].astype(np.float64)[m23, tz_i],
            floor=floor, min_names=min_names,
        ) if bool(m23.any()) else {"cs_ic": float("nan")}
        lift = float(sleeve_val.get("cs_ic", float("nan"))) - float(val["cs_ic"])
        keep_2017 = (
            not np.isfinite(sleeve_2017.get("cs_ic", float("nan")))
            or float(sleeve_2017["cs_ic"]) >= float(VAL_2017_KEEP)
        )
        promote_sleeve = np.isfinite(lift) and lift >= float(VAL_LIFT) and keep_2017
        payload["liquid_sleeve"] = {
            "adv_floor_pctile": floor,
            "cs_min_names": _sleeve_min_names(min_names, floor),
            "val": sleeve_val,
            "val_2017": sleeve_2017,
            "test": sleeve_test,
            "test_2023": sleeve_2023,
            "val_lift": float(lift),
            "promote": bool(promote_sleeve),
            "note": (
                "same overnight skip w; trade only top CS turnover_z tercile. "
                "Known at t. Not a new model. Live long-only sleeve can have "
                "worse causal-vol DD than the full long-only book (lumpier)."
            ),
        }
        print(
            f"liquid sleeve (top turnover tercile): val={float(sleeve_val.get('cs_ic', float('nan'))):+.4f} "
            f"lift={lift:+.4f} val2017={float(sleeve_2017.get('cs_ic', float('nan'))):+.4f} "
            f"test={float(sleeve_test.get('cs_ic', float('nan'))):+.4f} "
            f"2023={float(sleeve_2023.get('cs_ic', float('nan'))):+.4f} "
            f"{'PROMOTE live sleeve' if promote_sleeve else 'no'}",
            flush=True,
        )
    if not args.no_stability:
        payload["stability"] = _stability_ablate(
            cache, float(val["cs_ic"]), float(val["cs_ic_2017"])
        )

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

    live = next((r for r in payload["stress"] if r["name"] == "live"), None)
    live_flat = next((r for r in payload["stress"] if r["name"] == "live_flat"), None)
    live_lo = next((r for r in payload["stress"] if r["name"] == "live_long_only"), None)
    live_loc = next((r for r in payload["stress"] if r["name"] == "live_locate"), None)
    harsh = next((r for r in payload["stress"] if r["name"] == "harsh_auction"), None)
    paper = next((r for r in payload["stress"] if r["name"] == "overnight_10bp"), None)
    sleeve_ls = next((r for r in payload["stress"] if r["name"] == "live_liquid_sleeve"), None)
    sleeve_lo = next(
        (r for r in payload["stress"] if r["name"] == "live_liquid_sleeve_long_only"), None
    )

    if args.try_fill and args.data_dir:
        n_fill = int(args.try_fill)
        fill_cache_path = Path(str(args.cache) + f".fill{n_fill}.npz")
        print(FILL_FORMULA, flush=True)
        fill_cache = _ensure_cache(
            fill_cache_path,
            data_dir=args.data_dir,
            universe=args.universe,
            rebuild=True,
            label_return="open_fill",
            fill_minutes=n_fill,
        )
        fw, fb, ftrain = _fit_promoted(fill_cache)
        fsplit = _split_stats(fill_cache, fw, fb)
        fval = float(fsplit["val"]["cs_ic"])
        f2017 = float(fsplit["val"]["cs_ic_2017"])
        ftest = float(fsplit["test"]["cs_ic"])
        promote_fill = (
            np.isfinite(fval)
            and fval >= float(val["cs_ic"]) + VAL_LIFT
            and (not np.isfinite(f2017) or f2017 >= float(VAL_2017_KEEP) or args.synthetic)
        )
        payload["fill"] = {
            "minutes": n_fill,
            "alpha": n_fill / 390.0,
            "train_cs_ic": float(ftrain),
            "val": slim_cs_stats(fsplit["val"]) | {"cs_ic_2017": f2017},
            "test": slim_cs_stats(fsplit["test"]),
            "promote": bool(promote_fill),
            "note": (
                "separate estimand; default overnight y unchanged"
                if not promote_fill
                else "val-gate passed; still report separately from MOC→MOO overnight"
            ),
        }
        print(
            f"open+{n_fill}m fill: val={fval:+.4f} val2017={f2017:+.4f} "
            f"test={ftest:+.4f} vs overnight val={float(val['cs_ic']):+.4f} "
            f"{'PROMOTE-as-separate' if promote_fill else 'no promote'}",
            flush=True,
        )
        try:
            ftz = _feat_wide(fill_cache, "test", "turnover_z")
            fvol = _feat_wide(fill_cache, "test", "vol_level")
        except Exception:
            ftz, fvol = None, None
        fp, fr = _wide(fsplit["test_pred"], fsplit["test_y"], fsplit["test_d"])
        fill_book = _run_book(
            fp, fr, bundle=FILL_LIVE_BUNDLE, holding="open_fill", turnover_z=ftz, vol_level=fvol
        )
        payload["fill"]["live_book"] = _book_row(
            fill_book, f"fill{n_fill}_live", "MOC + continuous open+N exit; not MOO"
        )
        print(
            f"  fill live unlev net IR={payload['fill']['live_book']['unlevered_net_ir']:+.3f}",
            flush=True,
        )

    if args.try_weekly and args.data_dir:
        wk_cache = Path(str(args.cache) + ".weekly.npz")
        print("weekly close-to-close residual fallback (val-gate; not bigger Mamba)", flush=True)
        weekly_cache = _ensure_cache(
            wk_cache,
            data_dir=args.data_dir,
            universe=args.universe,
            rebuild=True,
            label_return="close",
            interval="weekly",
        )
        ww, wb, wtrain = _fit_promoted(weekly_cache)
        wsplit = _split_stats(weekly_cache, ww, wb)
        wval = float(wsplit["val"]["cs_ic"])
        payload["weekly"] = {
            "train_cs_ic": float(wtrain),
            "val": slim_cs_stats(wsplit["val"]) | {"cs_ic_2017": float(wsplit["val"]["cs_ic_2017"])},
            "test": slim_cs_stats(wsplit["test"]),
            "note": "weekly residual is a different estimand; not mixed into overnight y",
        }
        print(
            f"weekly residual: val={wval:+.4f} test={float(wsplit['test']['cs_ic']):+.4f}",
            flush=True,
        )

    harsh_ir = float((harsh or {}).get("unlevered_net_ir") or float("nan"))
    live_ir = float((live or {}).get("unlevered_net_ir") or float("nan"))
    overnight_dies = np.isfinite(harsh_ir) and harsh_ir < 0.3
    fill_ok = bool((payload.get("fill") or {}).get("promote"))
    if overnight_dies and fill_ok:
        payload["next_estimand"] = "open_fill"
    elif overnight_dies:
        payload["next_estimand"] = "weekly_residual"
    else:
        payload["next_estimand"] = "overnight_live"

    stab_name = (payload.get("stability") or {}).get("promoted")
    if skip_ok:
        payload["promoted"] = f"overnight_skip+{stab_name}" if stab_name else "overnight_skip"
        payload["verdict"] = (
            "Overnight skip remains the overnight book (not the close-to-close headline). "
            "Paper 10bp flatten IR is not live P&L. Headline live (name-level auction) "
            "and live_long_only / live_locate; do not mix ICs. "
            + (
                f"Val-gated skip variant {stab_name} lifted locked val without killing val-2017."
                if stab_name
                else "No year-stability skip variant cleared the locked-val gate."
            )
        )
    else:
        payload["verdict"] = (
            "Overnight skip did not stay strong under the locked protocol. "
            "Do not promote. Next highest-EV estimand is not bigger Mamba."
        )
    if overnight_dies:
        payload["verdict"] += (
            " Harsh auction realism killed overnight net IR; "
            f"next estimand={payload['next_estimand']} "
            "(open+N fill if it won val, else weekly residual). Close-to-close IR~1 stays "
            "the honest close-to-close book."
        )
    notes = []
    if paper and live_flat:
        notes.append(
            f"paper 10bp unlev net IR={paper['unlevered_net_ir']:+.3f}; "
            f"live_flat overlay={live_flat['unlevered_net_ir']:+.3f}"
        )
    if live:
        notes.append(f"live auction unlev net IR={live['unlevered_net_ir']:+.3f}")
    if live_loc:
        notes.append(
            f"live+locate unlev net IR={live_loc['unlevered_net_ir']:+.3f} "
            f"(mean shorts blocked/date={live_loc.get('mean_shorts_blocked')})"
        )
    if live_lo:
        notes.append(
            f"live long-only unlev net IR={live_lo['unlevered_net_ir']:+.3f} "
            f"maxDD={live_lo['levered_max_dd']:+.3f} (no locate)"
        )
    if harsh:
        notes.append(f"harsh auction unlev net IR={harsh['unlevered_net_ir']:+.3f}")
    if sleeve_ls:
        notes.append(
            f"live liquid sleeve unlev net IR={sleeve_ls['unlevered_net_ir']:+.3f} "
            f"lev maxDD={sleeve_ls['levered_max_dd']:+.3f}"
        )
    if sleeve_lo:
        notes.append(
            f"live liquid sleeve long-only unlev net IR={sleeve_lo['unlevered_net_ir']:+.3f} "
            f"lev maxDD={sleeve_lo['levered_max_dd']:+.3f}"
        )
    sleeve_meta = payload.get("liquid_sleeve") or {}
    if sleeve_meta:
        notes.append(
            "liquid sleeve CS "
            f"val={float(sleeve_meta.get('val', {}).get('cs_ic', float('nan'))):+.4f} "
            f"lift={float(sleeve_meta.get('val_lift', float('nan'))):+.4f} "
            f"{'PROMOTE sleeve filter' if sleeve_meta.get('promote') else 'no sleeve promote'}"
        )
    payload["friction_note"] = "; ".join(notes)
    if payload["friction_note"]:
        print(payload["friction_note"], flush=True)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_path}", flush=True)
    print(payload["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
