"""End-to-end forecast train -> generate integration."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

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


def _write_symbol_parquet(data_dir: Path, symbol: str, n_sessions: int = 120) -> None:
    rows: list[pd.DataFrame] = []
    day = 0
    while len(rows) < n_sessions:
        base = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
        day += 1
        if base.dayofweek >= 5:
            continue
        close = 10.0 + 0.01 * len(rows)
        rows.append(
            pd.DataFrame(
                {
                    "datetime": [base],
                    "Open": [close],
                    "High": [close + 0.01],
                    "Low": [close - 0.01],
                    "Close": [close],
                    "Volume": [1000.0],
                }
            )
        )
    pd.concat(rows).to_parquet(data_dir / f"{symbol}_daily.parquet")


def test_default_forecast_configs_are_valid():
    validate_loss_head(ForecastModelConfig(), ForecastTrainConfig())


def test_train_then_generate(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_symbol_parquet(data_dir, "AAA")

    ckpt_dir = tmp_path / "ckpt"
    data_cfg = DataConfig(
        data_dir=str(data_dir),
        interval="daily",
        horizon=1,
        seq_len=16,
        stride=8,
        min_context=4,
        warmup_bars=8,
        vol_halflife=5,
        z_window=10,
        z_min_periods=5,
        min_session_bars=1,
    )
    model_cfg = ForecastModelConfig(n_features=18, d_model=32, n_layer=2, d_state=8)
    train_cfg = ForecastTrainConfig(
        batch_size=1,
        max_steps=12,
        eval_interval=6,
        log_interval=4,
        checkpoint_dir=str(ckpt_dir),
        ic_loss_weight=0.0,
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
    assert "NEXT-DAY RETURN FORECAST" in proc.stdout


def test_train_saves_last_on_keyboard_interrupt(tmp_path: Path, monkeypatch):
    from forecast.training import cycle_loader

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_symbol_parquet(data_dir, "AAA")
    ckpt_dir = tmp_path / "ckpt"

    real_cycle = cycle_loader

    def interrupting_cycle(loader):
        inner = real_cycle(loader)
        yield next(inner)
        raise KeyboardInterrupt

    monkeypatch.setattr("forecast.training.cycle_loader", interrupting_cycle)
    summary = train(
        DataConfig(
            data_dir=str(data_dir),
            interval="daily",
            horizon=1,
            seq_len=16,
            stride=8,
            min_context=4,
            warmup_bars=8,
            vol_halflife=5,
            z_window=10,
            z_min_periods=5,
            min_session_bars=1,
        ),
        ForecastModelConfig(n_features=18, d_model=32, n_layer=2, d_state=8),
        ForecastTrainConfig(
            batch_size=1,
            max_steps=12,
            eval_interval=6,
            log_interval=4,
            checkpoint_dir=str(ckpt_dir),
            ic_loss_weight=0.0,
            precision="fp32",
        ),
        device=torch.device("cpu"),
        log_fn=None,
    )
    assert summary["interrupted"] is True
    assert (ckpt_dir / "last.pt").exists()
    assert summary["last_step"] >= 1
