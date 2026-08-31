"""Generate next-hour return forecasts from a trained checkpoint.

Usage:
    python -m forecast.generate --checkpoint checkpoints/forecast/best.pt
    python -m forecast.generate --checkpoint checkpoints/forecast/best.pt --last 10
    python -m forecast.generate --checkpoint checkpoints/forecast/best.pt \\
        --symbols AAPL,MSFT --csv pred_moves.csv

By default every parquet in the checkpoint's data_dir is scored. The printed
table has one column per ticker; cells are predicted next-hour moves in bp.

The checkpoint carries its own DataConfig and feature normalization, so the
features built here are identical to the ones the model was trained on.

The model predicts a volatility-normalized return; this script multiplies it
back by the volatility known at the forecast bar and reports basis points
(1 bp = 0.01%).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from forecast.checkpoint import load_forecaster, uncertainty_is_trained
from forecast.config import DataConfig
from forecast.data import (
    FEATURE_NAMES,
    build_panel,
    discover_symbol_files,
    symbol_from_path,
)
from forecast.model import ReturnForecaster
from mamba_lm.paths import anchor_to_repo, resolve_path


@torch.no_grad()
def forecast_panel(
    model: ReturnForecaster,
    panel: pd.DataFrame,
    state: dict[str, Any],
    device: torch.device,
    *,
    context: int,
    positions: np.ndarray,
    batch_size: int = 32,
    include_uncertainty: bool = False,
) -> pd.DataFrame:
    """Forecast at each bar index in ``positions``."""
    feature_names = state.get("feature_names", list(FEATURE_NAMES))
    mean_vec = np.asarray(state["feature_mean"], dtype=np.float32)
    std_vec = np.asarray(state["feature_std"], dtype=np.float32)
    raw = panel[feature_names].to_numpy(dtype=np.float32)
    normalized = (raw - mean_vec) / std_vec

    preds: list[float] = []
    sigmas: list[float] = []
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        windows = np.stack(
            [normalized[i - context + 1 : i + 1] for i in chunk], axis=0
        )
        x = torch.from_numpy(windows).to(device)
        mean, log_sigma = model(x)
        preds.extend(mean[:, -1].float().cpu().numpy().tolist())
        if include_uncertainty:
            sigmas.extend(torch.exp(log_sigma[:, -1]).float().cpu().numpy().tolist())

    rows = panel.iloc[positions]
    scale = rows["scale"].to_numpy(dtype=np.float64)
    pred_norm = np.asarray(preds, dtype=np.float64)
    pred_log_return = pred_norm * scale
    close = rows["close"].to_numpy(dtype=np.float64)

    out = pd.DataFrame(
        {
            "datetime": rows["datetime"].to_numpy(),
            "close": close,
            "traded": rows["traded"].to_numpy().astype(bool),
            "pred_norm": pred_norm,
            "scale": scale,
            "pred_return_bps": pred_log_return * 1e4,
            "pred_price_1h": close * np.exp(pred_log_return),
        }
    )
    if include_uncertainty and sigmas:
        out["pred_uncertainty_bps"] = np.asarray(sigmas, dtype=np.float64) * scale * 1e4
    if "target_raw" in panel.columns:
        realized = rows["target_raw"].to_numpy(dtype=np.float64) * 1e4
        has_outcome = rows["valid"].to_numpy(dtype=bool)
        out["realized_bps"] = np.where(has_outcome, realized, np.nan)
        if "horizon_traded" in panel.columns:
            out["horizon_traded"] = rows["horizon_traded"].to_numpy() > 0
    return out.reset_index(drop=True)


def choose_positions(
    panel: pd.DataFrame,
    *,
    context: int,
    last: int,
    asof: str | None,
) -> np.ndarray:
    """Bar indices to forecast at: either a single as-of bar or the newest N."""
    if asof is not None:
        stamp = pd.Timestamp(asof)
        matches = np.flatnonzero(panel["datetime"].to_numpy() <= np.datetime64(stamp))
        if matches.size == 0:
            raise ValueError(f"no bars at or before {stamp}")
        end = int(matches[-1])
        if end < context - 1:
            raise ValueError(
                f"only {end + 1} bars before {stamp}; need {context} for context"
            )
        return np.array([end], dtype=np.int64)

    end = len(panel) - 1
    first = max(context - 1, end - last + 1)
    if first > end:
        raise ValueError(f"need at least {context} bars, panel has {len(panel)}")
    return np.arange(first, end + 1, dtype=np.int64)


def parse_symbols(raw: str | None) -> list[str] | None:
    if raw is None or not str(raw).strip():
        return None
    return [part.strip().upper() for part in str(raw).split(",") if part.strip()]


def resolve_data_files(
    data_arg: str | None,
    data_dir: str | Path,
    symbols: list[str] | None,
) -> list[Path]:
    """Parquet files to score: one file, a directory, or the checkpoint data_dir."""
    if data_arg:
        path = resolve_path(data_arg)
        if path.is_dir():
            files = discover_symbol_files(path)
        else:
            files = [path]
    else:
        files = discover_symbol_files(data_dir)

    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"data file not found: {missing[0]}")

    if symbols is not None:
        wanted = set(symbols)
        files = [f for f in files if symbol_from_path(f) in wanted]
        have = {symbol_from_path(f) for f in files}
        unknown = sorted(wanted - have)
        if unknown:
            raise FileNotFoundError(
                "no parquet for symbol(s): " + ", ".join(unknown)
            )
    if not files:
        raise FileNotFoundError("no symbol parquet files to forecast")
    return files


def predicted_move_wide(by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Datetime index, one column per ticker, values = predicted move in bp."""
    parts: list[pd.Series] = []
    for sym in sorted(by_symbol):
        df = by_symbol[sym]
        parts.append(
            pd.Series(
                df["pred_return_bps"].to_numpy(dtype=np.float64),
                index=pd.to_datetime(df["datetime"]),
                name=sym,
            )
        )
    return pd.concat(parts, axis=1, sort=False).sort_index()


def latest_snapshot(by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows are fields; each ticker is its own column (latest bar per file)."""
    columns: dict[str, dict[str, str]] = {}
    for sym in sorted(by_symbol):
        row = by_symbol[sym].iloc[-1]
        entry = {
            "as of": _fmt_when(row["datetime"]),
            "print": "yes" if bool(row["traded"]) else "no",
            "last $": _fmt_px(float(row["close"])),
            "pred (bp)": f"{float(row['pred_return_bps']):+.1f}",
            "pred $ in 1h": _fmt_px(float(row["pred_price_1h"])),
        }
        if "realized_bps" in by_symbol[sym].columns:
            realized = row["realized_bps"]
            entry["realized (bp)"] = (
                f"{float(realized):+.1f}" if pd.notna(realized) else "-"
            )
        columns[sym] = entry
    return pd.DataFrame(columns)


def _fmt_px(value: float) -> str:
    return f"{value:,.4f}"


def _fmt_when(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def _format_wide_bp(wide: pd.DataFrame) -> str:
    display = wide.copy()
    display.index = pd.to_datetime(display.index).strftime("%Y-%m-%d %H:%M")
    display.index.name = "when"
    shown = display.apply(
        lambda col: col.map(lambda v: f"{v:+.1f}" if pd.notna(v) else "-")
    )
    return shown.to_string()


def format_stock_column_report(
    by_symbol: dict[str, pd.DataFrame],
    *,
    trained_on: str,
    checkpoint: Path,
    data_cfg: DataConfig,
    context: int,
    device: torch.device,
    notes: list[str],
) -> str:
    """Header plus a table whose columns are tickers (predicted move in bp)."""
    lines: list[str] = []
    names = ", ".join(sorted(by_symbol))
    lines.append("=" * 72)
    lines.append("  NEXT-HOUR RETURN FORECAST")
    lines.append("=" * 72)
    lines.append(f"  Horizon     {data_cfg.horizon} minutes ahead  (same session)")
    lines.append(f"  Context     last {context} minute bars")
    lines.append(f"  Stocks      {names}")
    lines.append(f"  Checkpoint  {checkpoint}")
    lines.append(f"  Trained on  {trained_on or 'unknown'}")
    lines.append(f"  Device      {device}")
    lines.append("")
    lines.append("  Units: bp = basis points = 0.01%.  +10 bp means the model")
    lines.append("  expects the price about 0.10% higher in one hour.")
    lines.append("  Each ticker is its own column. pred (bp) is the predicted move.")
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"  Note: {note}")

    if not by_symbol:
        lines.append("")
        lines.append("  No forecast rows.")
        lines.append("=" * 72)
        return "\n".join(lines)

    snapshot = latest_snapshot(by_symbol)
    lines.append("")
    lines.append("-" * 72)
    lines.append("  LATEST  (one column per stock)")
    lines.append("-" * 72)
    for row in snapshot.to_string().splitlines():
        lines.append("  " + row)

    wide = predicted_move_wide(by_symbol)
    if len(wide) > 1:
        lines.append("")
        lines.append("-" * 72)
        lines.append("  PREDICTED MOVE (bp)  |  one column per stock")
        lines.append("-" * 72)
        for row in _format_wide_bp(wide).splitlines():
            lines.append("  " + row)

    lines.append("=" * 72)
    return "\n".join(lines)


def _collect_shared_notes(
    state: dict[str, Any],
    symbols: list[str],
    *,
    context: int,
    min_context: int,
    include_unc: bool,
) -> list[str]:
    notes: list[str] = []
    if context < min_context:
        notes.append(
            f"context={context} is shorter than training min_context="
            f"{min_context}; the last bar of each window was never supervised."
        )
    if not include_unc:
        notes.append(
            "No trained uncertainty: the loss was not gaussian NLL. "
            "Only the predicted mean is shown."
        )
    trained_on = [m["symbol"] for m in state.get("symbols", [])]
    oos = [s for s in symbols if s not in trained_on]
    if oos and trained_on:
        notes.append(
            "Out-of-sample symbol(s): "
            + ", ".join(oos)
            + " (not in the training set)."
        )
    for m in state.get("symbols", []):
        train_mix = m.get("train_source_mix") or {}
        test_mix = m.get("test_source_mix") or {}
        if train_mix and test_mix:
            train_top = max(train_mix, key=train_mix.get)
            test_top = max(test_mix, key=test_mix.get)
            if train_top != test_top:
                notes.append(
                    f"Train vendor was mostly {train_top}, test mostly {test_top}. "
                    "Do not read a backtest IC as the same experiment."
                )
                break
    return notes


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Forecast next-hour equity returns.")
    p.add_argument("--checkpoint", default="checkpoints/forecast/best.pt")
    p.add_argument(
        "--data",
        default=None,
        help="one parquet, or a directory of parquets; default is every file "
        "in the checkpoint's data_dir",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="comma-separated tickers to include, e.g. AAPL,MSFT",
    )
    p.add_argument(
        "--context",
        type=int,
        default=None,
        help="bars of history per forecast (default: the training seq_len)",
    )
    p.add_argument("--last", type=int, default=1, help="forecast the newest N bars")
    p.add_argument("--asof", default=None, help="forecast a single bar, e.g. '2026-08-21 14:30'")
    p.add_argument("--csv", default=None, help="write the predicted-move table (stocks as columns)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"checkpoint not found: {ckpt_path}\nTrain one first: python -m forecast.training"
        )

    model, state = load_forecaster(ckpt_path, device)
    data_cfg = DataConfig.from_dict(state["data_config"])
    context = args.context or data_cfg.seq_len
    include_unc = uncertainty_is_trained(model, state)

    try:
        files = resolve_data_files(
            args.data, data_cfg.data_dir, parse_symbols(args.symbols)
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    by_symbol: dict[str, pd.DataFrame] = {}
    skip_notes: list[str] = []
    for path in files:
        symbol = symbol_from_path(path)
        print(f"forecasting {symbol} ...", file=sys.stderr)
        try:
            panel = build_panel(path, data_cfg)
            positions = choose_positions(
                panel, context=context, last=args.last, asof=args.asof
            )
            result = forecast_panel(
                model,
                panel,
                state,
                device,
                context=context,
                positions=positions,
                batch_size=args.batch_size,
                include_uncertainty=include_unc,
            )
        except (ValueError, SystemExit) as exc:
            skip_notes.append(f"skipped {symbol}: {exc}")
            continue
        by_symbol[symbol] = result

    if not by_symbol:
        raise SystemExit("no symbols produced a forecast\n" + "\n".join(skip_notes))

    trained_on = ", ".join(m["symbol"] for m in state.get("symbols", []))
    notes = _collect_shared_notes(
        state,
        sorted(by_symbol),
        context=context,
        min_context=data_cfg.min_context,
        include_unc=include_unc,
    )
    notes.extend(skip_notes)

    print(
        format_stock_column_report(
            by_symbol,
            trained_on=trained_on,
            checkpoint=ckpt_path,
            data_cfg=data_cfg,
            context=context,
            device=device,
            notes=notes,
        )
    )

    if args.csv:
        wide = predicted_move_wide(by_symbol)
        csv_path = anchor_to_repo(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(csv_path, index_label="datetime")
        print(f"\nWrote CSV: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
