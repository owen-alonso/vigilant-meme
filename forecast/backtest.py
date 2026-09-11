"""Last-bar quantile long-short book on a locked test window.

Usage:
    python -m forecast.backtest --checkpoint checkpoints/forecast_ridge/best.pt
    python -m forecast.backtest --checkpoint ... --quantile 0.2 --cost-bps 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from forecast.checkpoint import load_forecaster
from forecast.config import DataConfig
from forecast.generate import (
    forecast_panel,
    load_forecast_panels,
    parse_symbols,
    resolve_data_files,
    symbol_from_path,
)
from forecast.training import _pearson, _spearman
from mamba_lm.paths import anchor_to_repo, resolve_path


def test_start_from_state(state: dict[str, Any]) -> pd.Timestamp | None:
    ends = []
    for row in state.get("symbols") or []:
        raw = row.get("val_end")
        if raw:
            ends.append(pd.Timestamp(raw))
    if not ends:
        return None
    return max(ends)


def score_last_bars(
    model: torch.nn.Module,
    panels: dict[str, pd.DataFrame],
    state: dict[str, Any],
    device: torch.device,
    *,
    context: int,
    start: pd.Timestamp | None,
    batch_size: int = 64,
    exclude: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide pred_norm and realized residual (same units as the training label)."""
    skip = exclude or frozenset()
    pred_parts: list[pd.Series] = []
    y_parts: list[pd.Series] = []
    for symbol, panel in panels.items():
        if symbol in skip or len(panel) < context:
            continue
        if start is not None:
            stamps = pd.to_datetime(panel["datetime"])
            mask = stamps >= start
            idx = np.flatnonzero(mask.to_numpy())
            idx = idx[idx >= context - 1]
        else:
            idx = np.arange(context - 1, len(panel), dtype=np.int64)
        if idx.size == 0:
            continue
        scored = forecast_panel(
            model,
            panel,
            state,
            device,
            context=context,
            positions=idx.astype(np.int64),
            batch_size=batch_size,
            include_uncertainty=False,
        )
        when = pd.to_datetime(scored["datetime"]).dt.tz_localize(None).dt.normalize()
        pred_parts.append(
            pd.Series(scored["pred_norm"].to_numpy(dtype=np.float64), index=when, name=symbol)
        )
        if "realized_bps" in scored.columns and "scale" in scored.columns:
            realized = scored["realized_bps"].to_numpy(dtype=np.float64) / 1e4
            scale = scored["scale"].to_numpy(dtype=np.float64)
            y_norm = np.divide(
                realized, scale, out=np.full_like(realized, np.nan), where=scale > 0
            )
        else:
            y_norm = np.full(len(scored), np.nan)
        y_parts.append(pd.Series(y_norm, index=when, name=symbol))
    if not pred_parts:
        empty = pd.DataFrame()
        return empty, empty
    pred = pd.concat(pred_parts, axis=1).sort_index()
    realized = pd.concat(y_parts, axis=1).reindex_like(pred)
    pred = pred.groupby(level=0).last()
    realized = realized.groupby(level=0).last()
    return pred, realized


def cs_ic_by_date(
    pred: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    min_names: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for ts in pred.index:
        p = pred.loc[ts]
        y = realized.loc[ts]
        pair = pd.concat([p, y], axis=1, keys=["p", "y"]).dropna()
        if len(pair) < min_names:
            continue
        rows.append(
            {
                "datetime": ts,
                "n": float(len(pair)),
                "ic": _pearson(pair["p"].to_numpy(), pair["y"].to_numpy()),
                "ic_spearman": _spearman(pair["p"].to_numpy(), pair["y"].to_numpy()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["datetime", "n", "ic", "ic_spearman"])
    return pd.DataFrame(rows).set_index("datetime")


def quantile_weights(
    scores: pd.Series,
    *,
    quantile: float,
    long_only: bool = False,
) -> pd.Series:
    """Long top / short bottom (dollar-neutral) or long-only top quantile."""
    s = scores.dropna()
    w = pd.Series(0.0, index=s.index, dtype=np.float64)
    n = int(s.size)
    if n < 2:
        return w
    q = min(0.49, max(0.05, float(quantile)))
    k = max(1, int(math.floor(n * q)))
    order = s.sort_values()
    if long_only:
        if n < 5:
            w.loc[order.index[-1]] = 1.0
            return w
        long = order.index[-k:]
        w.loc[long] = 1.0 / k
        return w
    if n < 5:
        w.loc[order.index[0]] = -0.5
        w.loc[order.index[-1]] = 0.5
        return w
    short = order.index[:k]
    long = order.index[-k:]
    w.loc[short] = -0.5 / k
    w.loc[long] = 0.5 / k
    return w


def book_pnl(
    pred: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    quantile: float = 0.2,
    round_trip_bps: float = 10.0,
    vol_target: float = 1.0,
    periods_per_year: float = 252.0,
    long_only: bool = False,
    min_names: int = 8,
) -> dict[str, Any]:
    """Vol-scaled quantile long-short (or long-only). Costs from one-way turnover * half round-trip."""
    dates = pred.index.intersection(realized.index)
    weights: list[pd.Series] = []
    gross: list[float] = []
    for ts in dates:
        pair_all = pd.concat(
            [pred.loc[ts], realized.loc[ts]], axis=1, keys=["p", "r"]
        ).dropna()
        if len(pair_all) < int(min_names):
            continue
        w = quantile_weights(
            pair_all["p"], quantile=quantile, long_only=long_only
        )
        r = pair_all["r"].reindex(w.index)
        pair = pd.concat([w, r], axis=1, keys=["w", "r"]).dropna()
        if len(pair) < 2:
            continue
        pnl = float((pair["w"] * pair["r"]).sum())
        if not np.isfinite(pnl):
            continue
        weights.append(w.rename(ts))
        gross.append(pnl)
    if len(gross) < 5:
        return {
            "n_dates": float(len(gross)),
            "gross_ir": float("nan"),
            "net_ir": float("nan"),
            "hit_rate": float("nan"),
            "max_dd": float("nan"),
            "mean_cs_ic": float("nan"),
            "mean_cs_ic_spearman": float("nan"),
        }
    w_panel = pd.concat(weights, axis=1).T.fillna(0.0)
    w_panel.index = pd.to_datetime(w_panel.index)
    prev = w_panel.shift(1).fillna(0.0)
    turnover = 0.5 * (w_panel - prev).abs().sum(axis=1)
    gross_s = pd.Series(gross, index=w_panel.index, dtype=np.float64)
    vol = float(gross_s.std(ddof=0))
    lever = 1.0
    if vol > 1e-12 and vol_target > 0:
        lever = float(vol_target) / (vol * math.sqrt(periods_per_year))
    cost = (float(round_trip_bps) * 1e-4) * turnover * lever
    net_s = lever * gross_s - cost
    equity = net_s.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    ir = lambda x: (
        float(x.mean() / x.std(ddof=0) * math.sqrt(periods_per_year))
        if float(x.std(ddof=0)) > 1e-12
        else float("nan")
    )
    ics = cs_ic_by_date(pred.loc[dates], realized.loc[dates])
    return {
        "n_dates": float(len(net_s)),
        "n_names": float(pred.shape[1]),
        "lever": lever,
        "gross_ir": ir(lever * gross_s),
        "net_ir": ir(net_s),
        "hit_rate": float((net_s > 0).mean()),
        "max_dd": float(dd.min()) if len(dd) else float("nan"),
        "mean_turnover": float(turnover.mean()),
        "mean_cost": float(cost.mean()),
        "round_trip_bps": float(round_trip_bps),
        "quantile": float(quantile),
        "vol_target": float(vol_target),
        "long_only": bool(long_only),
        "min_names": float(min_names),
        "mean_cs_ic": float(ics["ic"].mean()) if len(ics) else float("nan"),
        "mean_cs_ic_spearman": (
            float(ics["ic_spearman"].mean()) if len(ics) else float("nan")
        ),
        "cs_ic_tstat": (
            float(ics["ic"].mean() / (ics["ic"].std(ddof=1) / math.sqrt(len(ics))))
            if len(ics) > 2 and float(ics["ic"].std(ddof=1) or 0) > 1e-12
            else float("nan")
        ),
        "net": net_s,
        "gross": lever * gross_s,
        "cs_ic": ics,
        "weights": w_panel,
    }


def format_report(stats: dict[str, Any], *, checkpoint: Path, test_start: Any) -> str:
    lines = [
        "=" * 72,
        "  LAST-BAR QUANTILE LONG-SHORT",
        "=" * 72,
        f"  Checkpoint  {checkpoint}",
        f"  Test from   {test_start}",
        f"  Names       {int(stats.get('n_names', 0))}   dates {int(stats.get('n_dates', 0))}",
        f"  Quantile    {stats.get('quantile', float('nan')):.2f}  "
        f"round-trip {stats.get('round_trip_bps', float('nan')):.1f} bp"
        f"{'  LONG-ONLY' if stats.get('long_only') else ''}",
        f"  Lever       {stats.get('lever', float('nan')):.3f}  (vol target "
        f"{stats.get('vol_target', float('nan')):.2f} annual)",
        "",
        f"  mean CS IC (Pearson)   {stats.get('mean_cs_ic', float('nan')):+.4f}  "
        f"t={stats.get('cs_ic_tstat', float('nan')):.2f}",
        f"  mean CS IC (Spearman)  {stats.get('mean_cs_ic_spearman', float('nan')):+.4f}",
        f"  gross IR               {stats.get('gross_ir', float('nan')):+.3f}",
        f"  net IR                 {stats.get('net_ir', float('nan')):+.3f}",
        f"  hit rate               {stats.get('hit_rate', float('nan')):.3f}",
        f"  max DD (vol units)     {stats.get('max_dd', float('nan')):+.3f}",
        f"  mean turnover (1-way)  {stats.get('mean_turnover', float('nan')):.3f}",
        "=" * 72,
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quantile long-short backtest of last-bar scores.")
    p.add_argument("--checkpoint", default="checkpoints/forecast/best.pt")
    p.add_argument("--data", default=None)
    p.add_argument("--symbols", default=None)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--cost-bps", type=float, default=10.0, help="round-trip cost in basis points")
    p.add_argument("--vol-target", type=float, default=1.0)
    p.add_argument("--long-only", action="store_true", help="long the top quantile only (no short leg)")
    p.add_argument(
        "--min-names",
        type=int,
        default=None,
        help="skip dates with fewer names than this (default: checkpoint cross_section_min_names)",
    )
    p.add_argument(
        "--all-names",
        action="store_true",
        help="score every parquet in data/, not only names listed in the checkpoint",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--json", default=None, help="write scalar stats to this path")
    p.add_argument("--cs-csv", default=None, help="write per-date CS IC series")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2
    model, state = load_forecaster(ckpt_path, device)
    data_cfg = DataConfig.from_dict(state["data_config"])
    try:
        files = resolve_data_files(
            args.data, data_cfg.data_dir, parse_symbols(args.symbols), interval=data_cfg.interval
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.data:
        data_path = resolve_path(args.data)
        universe_dir = data_path if data_path.is_dir() else data_path.parent
    else:
        universe_dir = data_cfg.data_dir
    panels = load_forecast_panels(files, data_cfg, universe_dir=universe_dir)
    bench = str(data_cfg.benchmark_symbol or "").upper()
    trained = {
        str(row.get("symbol", "")).upper()
        for row in (state.get("symbols") or [])
        if row.get("symbol")
    }
    if trained and not args.symbols and not args.all_names:
        files = [
            f
            for f in files
            if symbol_from_path(f) in trained or symbol_from_path(f) == bench
        ]
    min_names = (
        int(args.min_names)
        if args.min_names is not None
        else int(data_cfg.cross_section_min_names)
    )
    start = test_start_from_state(state)
    pred, realized = score_last_bars(
        model,
        panels,
        state,
        device,
        context=data_cfg.seq_len,
        start=start,
        batch_size=args.batch_size,
        exclude=frozenset({bench} if bench else ()),
    )
    stats = book_pnl(
        pred,
        realized,
        quantile=args.quantile,
        round_trip_bps=args.cost_bps,
        vol_target=args.vol_target,
        periods_per_year=252.0 if data_cfg.is_daily() else (52.0 if data_cfg.interval == "weekly" else 12.0),
        long_only=args.long_only,
        min_names=min_names,
    )
    print(format_report(stats, checkpoint=ckpt_path, test_start=start))
    if args.json:
        out = {k: v for k, v in stats.items() if k not in {"net", "gross", "cs_ic", "weights"}}
        path = anchor_to_repo(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {path}", file=sys.stderr)
    if args.cs_csv:
        ics = stats.get("cs_ic")
        if isinstance(ics, pd.DataFrame) and len(ics):
            path = anchor_to_repo(args.cs_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            ics.to_csv(path, index_label="datetime")
            print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
