"""End-to-end forecast train -> generate integration."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecast.config import (
    DataConfig,
    ForecastModelConfig,
    ForecastTrainConfig,
    validate_loss_head,
)
from forecast.generate import main as generate_main
from forecast.training import train


def _write_symbol_parquet(data_dir: Path, symbol: str, n_sessions: int = 6) -> None:
    rows: list[pd.DataFrame] = []
    day = 0
    while len(rows) < n_sessions:
        base = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
        day += 1
        if base.dayofweek >= 5:
            continue
        n = 390
        close = 10.0 + np.linspace(0, 0.5, n)
        dt = pd.date_range(base.replace(hour=9, minute=30), periods=n, freq="min")
        rows.append(
            pd.DataFrame(
                {
                    "datetime": dt,
                    "Open": close,
                    "High": close + 0.01,
                    "Low": close - 0.01,
                    "Close": close,
                    "Volume": np.full(n, 1000.0),
                }
            )
        )
    pd.concat(rows).to_parquet(data_dir / f"{symbol}_clean_1min.parquet")


def test_default_forecast_configs_are_valid():
    validate_loss_head(ForecastModelConfig(), ForecastTrainConfig())


def test_train_then_generate(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_symbol_parquet(data_dir, "AAA")

    ckpt_dir = tmp_path / "ckpt"
    data_cfg = DataConfig(
        data_dir=str(data_dir),
        horizon=60,
        seq_len=64,
        stride=32,
        min_context=16,
        min_session_bars=30,
    )
    model_cfg = ForecastModelConfig(n_features=18, d_model=32, n_layer=2, d_state=8)
    train_cfg = ForecastTrainConfig(
        batch_size=4,
        max_steps=12,
        eval_interval=6,
        log_interval=4,
        checkpoint_dir=str(ckpt_dir),
        precision="fp32",
    )
    train(
        data_cfg,
        model_cfg,
        train_cfg,
        device=torch.device("cpu"),
        log_fn=None,
    )

    ckpt = ckpt_dir / "best.pt"
    assert ckpt.exists()

    # In-process generate (covers sys.stderr logging path).
    generate_main(["--checkpoint", str(ckpt), "--cpu", "--last", "2"])

    # Subprocess entry point should also work after packaging.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "forecast.generate",
            "--checkpoint",
            str(ckpt),
            "--cpu",
            "--last",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NEXT-HOUR RETURN FORECAST" in proc.stdout
