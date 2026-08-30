"""Generate next-hour return forecasts from a trained checkpoint.

Usage:
    python -m forecast.generate --checkpoint checkpoints/forecast/best.pt
    python -m forecast.generate --checkpoint checkpoints/forecast/best.pt \\
        --data data/AAPL_clean_1min.parquet --last 20 --csv aapl_forecast.csv

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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast.config import DataConfig, ForecastModelConfig
from forecast.data import FEATURE_NAMES, REPO_ROOT, build_panel, discover_symbol_files
from forecast.model import ReturnForecaster


def load_forecaster(
    checkpoint: str | Path, device: torch.device
) -> tuple[ReturnForecaster, dict[str, Any]]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ForecastModelConfig.from_dict(state["model_config"])
    model = ReturnForecaster(model_cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def uncertainty_is_trained(model: ReturnForecaster, state: dict[str, Any]) -> bool:
    """Only the gaussian NLL trains log_sigma; otherwise the column is noise."""
    train_cfg = state.get("train_config") or {}
    return bool(model.config.heteroscedastic) and train_cfg.get("loss") == "gaussian"


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
            raise SystemExit(f"no bars at or before {stamp}")
        end = int(matches[-1])
        if end < context - 1:
            raise SystemExit(
                f"only {end + 1} bars before {stamp}; need {context} for context"
            )
        return np.array([end], dtype=np.int64)

    end = len(panel) - 1
    first = max(context - 1, end - last + 1)
    if first > end:
        raise SystemExit(f"need at least {context} bars, panel has {len(panel)}")
    return np.arange(first, end + 1, dtype=np.int64)


def _fmt_px(value: float) -> str:
    return f"{value:,.4f}"


def _fmt_bps(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f} bp"


def _fmt_when(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def format_forecast_report(
    result: pd.DataFrame,
    *,
    symbol: str,
    trained_on: str,
    checkpoint: Path,
    data_path: Path,
    data_cfg: DataConfig,
    context: int,
    device: torch.device,
    include_uncertainty: bool,
    notes: list[str],
) -> str:
    """Human-readable terminal report. 1 bp = 0.01%."""
    lines: list[str] = []
    horizon_min = data_cfg.horizon
    lines.append("=" * 72)
    lines.append(f"  NEXT-HOUR RETURN FORECAST  |  {symbol}")
    lines.append("=" * 72)
    lines.append(f"  Horizon     {horizon_min} minutes ahead  (same session)")
    lines.append(f"  Context     last {context} minute bars")
    lines.append(f"  Data        {data_path}")
    lines.append(f"  Checkpoint  {checkpoint}")
    lines.append(f"  Trained on  {trained_on or 'unknown'}")
    lines.append(f"  Device      {device}")
    lines.append("")
    lines.append("  Units: bp = basis points = 0.01%.  +10 bp means the model")
    lines.append("  expects the price about 0.10% higher in one hour.")
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"  Note: {note}")

    if result.empty:
        lines.append("")
        lines.append("  No forecast rows.")
        lines.append("=" * 72)
        return "\n".join(lines)

    latest = result.iloc[-1]
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"  LATEST  as of {_fmt_when(latest['datetime'])}")
    lines.append("-" * 72)
    traded = "real print" if bool(latest["traded"]) else "untraded / filled slot"
    lines.append(f"  Last price          {_fmt_px(float(latest['close']))}   ({traded})")
    lines.append(
        f"  Predicted move      {_fmt_bps(float(latest['pred_return_bps']))}"
        f"   ->  {_fmt_px(float(latest['pred_price_1h']))} in 1 hour"
    )
    if include_uncertainty and "pred_uncertainty_bps" in result.columns:
        unc = float(latest["pred_uncertainty_bps"])
        lines.append(f"  Residual uncertainty +/-{unc:.1f} bp  (model-predicted, 1 sigma)")
    if "realized_bps" in result.columns:
        realized = latest["realized_bps"]
        if pd.notna(realized):
            lines.append(f"  Realized (since)    {_fmt_bps(float(realized))}")
        else:
            lines.append("  Realized (since)    not yet known (horizon still open)")

    table = result.copy()
    table["when"] = pd.to_datetime(table["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
    table["print"] = np.where(table["traded"], "yes", "no")
    table["pred_bp"] = table["pred_return_bps"].map(lambda v: f"{v:+.1f}")
    table["pred_px"] = table["pred_price_1h"].map(lambda v: f"{v:,.4f}")
    table["last_px"] = table["close"].map(lambda v: f"{v:,.4f}")
    show = ["when", "print", "last_px", "pred_bp", "pred_px"]
    headers = {
        "when": "when",
        "print": "print",
        "last_px": "last $",
        "pred_bp": "pred (bp)",
        "pred_px": "pred $ in 1h",
    }
    if include_uncertainty and "pred_uncertainty_bps" in table.columns:
        table["unc_bp"] = table["pred_uncertainty_bps"].map(lambda v: f"+/-{v:.1f}")
        show.append("unc_bp")
        headers["unc_bp"] = "uncert. (bp)"
    if "realized_bps" in table.columns:
        table["real_bp"] = table["realized_bps"].map(
            lambda v: f"{v:+.1f}" if pd.notna(v) else "-"
        )
        show.append("real_bp")
        headers["real_bp"] = "realized (bp)"

    lines.append("")
    lines.append("-" * 72)
    title = "  ALL REQUESTED BARS" if len(result) > 1 else "  DETAIL"
    lines.append(title)
    lines.append("-" * 72)
    display = table[show].rename(columns=headers)
    body = display.to_string(index=False)
    for row in body.splitlines():
        lines.append("  " + row)

    scored = result["realized_bps"].dropna() if "realized_bps" in result.columns else pd.Series(dtype=float)
    if len(scored) >= 2:
        pred = result.loc[scored.index, "pred_return_bps"].to_numpy()
        real = scored.to_numpy()
        ic = float(np.corrcoef(pred, real)[0, 1]) if np.std(pred) > 0 and np.std(real) > 0 else float("nan")
        lines.append("")
        lines.append(
            f"  On these {len(scored)} bars that already have an outcome: "
            f"IC = {ic:+.3f}"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Forecast next-hour equity returns.")
    p.add_argument("--checkpoint", default="checkpoints/forecast/best.pt")
    p.add_argument(
        "--data",
        default=None,
        help="parquet of 1-minute OHLCV bars; defaults to the first file in the "
        "checkpoint's data_dir",
    )
    p.add_argument(
        "--context",
        type=int,
        default=None,
        help="bars of history per forecast (default: the training seq_len)",
    )
    p.add_argument("--last", type=int, default=1, help="forecast the newest N bars")
    p.add_argument("--asof", default=None, help="forecast a single bar, e.g. '2026-08-21 14:30'")
    p.add_argument("--csv", default=None, help="write results to this path")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(
            f"checkpoint not found: {ckpt_path}\nTrain one first: python -m forecast.training"
        )

    model, state = load_forecaster(ckpt_path, device)
    data_cfg = DataConfig.from_dict(state["data_config"])
    context = args.context or data_cfg.seq_len
    include_unc = uncertainty_is_trained(model, state)

    notes: list[str] = []
    if context < data_cfg.min_context:
        notes.append(
            f"context={context} is shorter than training min_context="
            f"{data_cfg.min_context}; the last bar of each window was never supervised."
        )
    if not include_unc:
        notes.append(
            "No trained uncertainty: the loss was not gaussian NLL. "
            "Only the predicted mean is shown."
        )

    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute() and not data_path.exists():
            data_path = REPO_ROOT / data_path
    else:
        data_path = discover_symbol_files(data_cfg.data_dir)[0]
    if not data_path.exists():
        raise SystemExit(f"data file not found: {data_path}")
    panel = build_panel(data_path, data_cfg)
    symbol = str(panel["symbol"].iloc[0])

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

    trained_on = ", ".join(m["symbol"] for m in state.get("symbols", []))
    if symbol not in trained_on:
        notes.append(
            f"{symbol} was not in the training set; this is an out-of-sample symbol."
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

    print(
        format_forecast_report(
            result,
            symbol=symbol,
            trained_on=trained_on,
            checkpoint=ckpt_path,
            data_path=data_path,
            data_cfg=data_cfg,
            context=context,
            device=device,
            include_uncertainty=include_unc,
            notes=notes,
        )
    )

    if args.csv:
        result.to_csv(args.csv, index=False)
        print(f"\nWrote CSV: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
