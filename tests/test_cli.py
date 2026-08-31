"""CLI and entry-point regression tests."""

from __future__ import annotations

import subprocess
import sys

from mamba_lm.cli import _model_cfg
from mamba_lm.config import MambaConfig
from forecast.checkpoint import uncertainty_is_trained
from forecast.config import ForecastModelConfig
from forecast.model import ReturnForecaster
import argparse


def test_baseline_model_cfg_keeps_dynamic_a_default():
    args = argparse.Namespace(
        d_model=32,
        n_layer=1,
        d_state=8,
        expand=2,
        dynamic_weights=False,
        dynamic_strength=0.1,
    )
    cfg = _model_cfg(args)
    assert cfg.dynamic_weights is False
    assert cfg.dynamic_A is True


def test_mamba_lm_console_script_importable():
    from mamba_lm.cli import main

    assert callable(main)


def test_uncertainty_is_trained_for_huber_aux():
    model = ReturnForecaster(ForecastModelConfig(n_features=18, heteroscedastic=True))
    state = {"train_config": {"loss": "huber", "sigma_aux_weight": 0.5}}
    assert uncertainty_is_trained(model, state)


def test_mamba_lm_console_script_runs_report():
    proc = subprocess.run(
        [sys.executable, "-m", "mamba_lm.cli", "report", "--d-model", "32", "--n-layer", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Mamba parameters:" in proc.stdout
