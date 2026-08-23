"""Generate next-hour return forecasts from a trained checkpoint.

Usage:
    # Latest forecast for the symbol the model was trained on
    python -m forecast.generate --checkpoint checkpoints/forecast/best.pt

    # Any other equity, same bar schema
    python -m forecast.generate --checkpoint checkpoints/forecast/best.pt \
        --data data/AAPL_clean_1min.parquet --last 20 --csv aapl_forecast.csv

The checkpoint carries its own ``DataConfig`` and feature normalization, so the
features built here are identical to the ones the model was trained on.

The model predicts a volatility-normalized return; this script multiplies it
back by the volatility known at the forecast bar and reports basis points.
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
    # Run as a plain script (`python forecast/generate.py`, or from an IDE):
    # put the repo root on sys.path so the package imports below resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast.config import DataConfig, ForecastModelConfig
from forecast.data import FEATURE_NAMES, build_panel, discover_symbol_files
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
) -> pd.DataFrame:
    """Forecast at each bar index in ``positions``.

    Each forecast runs the model over the ``context`` bars ending at that index
    and reads the final output. The backbone is causal, so this is exactly the
    computation the model saw during training.
    """
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
        sigmas.extend(torch.exp(log_sigma[:, -1]).float().cpu().numpy().tolist())

    rows = panel.iloc[positions]
    scale = rows["scale"].to_numpy(dtype=np.float64)
    pred_norm = np.asarray(preds, dtype=np.float64)
    pred_log_return = pred_norm * scale

    out = pd.DataFrame(
        {
            "datetime": rows["datetime"].to_numpy(),
            "close": rows["close"].to_numpy(),
            "traded": rows["traded"].to_numpy().astype(bool),
            "pred_norm": pred_norm,
            "scale": scale,
            "pred_return_bps": pred_log_return * 1e4,
            "pred_price_1h": rows["close"].to_numpy() * np.exp(pred_log_return),
            "pred_uncertainty_bps": np.asarray(sigmas, dtype=np.float64) * scale * 1e4,
        }
    )
    # Realized outcome, present only for bars old enough to have one.
    if "target_raw" in panel.columns:
        realized = rows["target_raw"].to_numpy(dtype=np.float64) * 1e4
        has_outcome = rows["valid"].to_numpy(dtype=bool)
        out["realized_bps"] = np.where(has_outcome, realized, np.nan)
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
    if not ckpt_path.exists():
        raise SystemExit(
            f"checkpoint not found: {ckpt_path}\nTrain one first: python -m forecast.training"
        )

    model, state = load_forecaster(ckpt_path, device)
    data_cfg = DataConfig.from_dict(state["data_config"])
    context = args.context or data_cfg.seq_len

    data_path = Path(args.data) if args.data else discover_symbol_files(data_cfg.data_dir)[0]
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
    )

    trained_on = ", ".join(m["symbol"] for m in state.get("symbols", []))
    print(f"symbol={symbol}  bars={len(panel)}  context={context}  device={device}")
    print(f"checkpoint={ckpt_path}  trained_on={trained_on or 'unknown'}")
    if symbol not in trained_on:
        print(f"note: {symbol} was not in the training set; this is an out-of-sample symbol")
    print(f"horizon={data_cfg.horizon} bars ahead (~{data_cfg.horizon} minutes)\n")

    display = result.copy()
    display["datetime"] = display["datetime"].astype(str)
    for col in ("close", "pred_price_1h"):
        display[col] = display[col].round(4)
    for col in ("pred_norm", "pred_return_bps", "pred_uncertainty_bps", "realized_bps"):
        if col in display:
            display[col] = display[col].round(3)
    print(
        display.drop(columns=["scale"]).to_string(index=False, na_rep="-")
    )

    if args.csv:
        result.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
