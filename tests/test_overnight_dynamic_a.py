"""Dynamic A hypernet health, forecast-side gradient flow, VAL-only gates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from forecast.config import ForecastModelConfig
from forecast.dynamic_health import (
    controller_param_grad_norm,
    format_dynamic_health_ascii,
    interpret_scale_reports,
)
from forecast.model import ReturnForecaster
from forecast.overnight_dynamic import (
    align_encoder_pred,
    apply_residual_pred,
    blend_residual,
    decide_dynamic_a_cs_promote,
    decide_dynamic_a_overnight_promote,
    format_dynamic_a_block,
)


def _ascii_ok(text: str) -> None:
    text.encode("cp1252")
    text.encode("ascii")


def test_format_dynamic_health_ascii_is_cp1252_safe():
    health = interpret_scale_reports(
        [
            {
                "layer": 0,
                "dynamic_A_mean": 1.0123,
                "dynamic_A_std": 0.0345,
                "dynamic_A_min": 0.91,
                "dynamic_A_max": 1.08,
            }
        ],
        controller_grad_norm=1.2e-3,
        after_training=True,
    )
    text = format_dynamic_health_ascii(health)
    assert "+/-" in text
    assert "live" in text
    _ascii_ok(text)
    _ascii_ok(format_dynamic_health_ascii({"active": False}))


def test_format_dynamic_a_block_is_cp1252_safe():
    payload = {
        "dynamic_a": {
            "alpha": 0.5,
            "max_steps": 8,
            "n_aligned_val": 12,
            "promotion": decide_dynamic_a_overnight_promote(
                val_on={
                    "sleeve_up_pct": 61.0,
                    "sleeve_coverage": 0.10,
                    "dir_pct": 55.0,
                },
                val_off={
                    "sleeve_up_pct": 57.0,
                    "sleeve_coverage": 0.10,
                    "dir_pct": 54.0,
                },
                val_skip={
                    "sleeve_up_pct": 56.0,
                    "sleeve_coverage": 0.10,
                    "dir_pct": 54.0,
                },
            ),
            "on": {
                "val_sleeve_up": 61.0,
                "val_sleeve_cover": 0.10,
                "val_dir": 55.0,
                "val_cs_ic": 0.04,
                "test_sleeve_up": 56.7,
                "test_sleeve_cover": 0.07,
                "test_dir": 51.1,
                "controller_present": True,
                "controller_grad_step0": 0.01,
                "controller_grad_final": 0.02,
                "health_init": interpret_scale_reports(
                    [
                        {
                            "layer": 0,
                            "dynamic_A_mean": 1.0,
                            "dynamic_A_std": 0.0,
                            "dynamic_A_min": 1.0,
                            "dynamic_A_max": 1.0,
                        }
                    ]
                ),
                "health_final": interpret_scale_reports(
                    [
                        {
                            "layer": 0,
                            "dynamic_A_mean": 1.02,
                            "dynamic_A_std": 0.03,
                            "dynamic_A_min": 0.94,
                            "dynamic_A_max": 1.09,
                        }
                    ],
                    after_training=True,
                ),
            },
            "off": {
                "val_sleeve_up": 57.0,
                "val_sleeve_cover": 0.10,
                "val_dir": 54.0,
            },
            "skip": {
                "val_sleeve_up": 56.0,
                "val_sleeve_cover": 0.10,
                "val_dir": 54.0,
                "val_cs_ic": 0.03,
                "test_sleeve_up": 56.0,
                "test_dir": 51.1,
            },
        }
    }
    text = format_dynamic_a_block(payload)
    assert "PROMOTE DYNAMIC A OVERNIGHT? YES" in text
    assert "reached_60=True" in text
    _ascii_ok(text)


def test_init_scale_is_identity_then_live_flag_needs_std():
    init = interpret_scale_reports(
        [
            {
                "layer": 0,
                "dynamic_A_mean": 1.0,
                "dynamic_A_std": 0.0,
                "dynamic_A_min": 1.0,
                "dynamic_A_max": 1.0,
            }
        ]
    )
    assert init["init_identity"] is True
    assert init["live"] is False
    assert init["collapsed"] is False
    trained_dead = interpret_scale_reports(
        [
            {
                "layer": 0,
                "dynamic_A_mean": 1.0,
                "dynamic_A_std": 0.0,
                "dynamic_A_min": 1.0,
                "dynamic_A_max": 1.0,
            }
        ],
        after_training=True,
    )
    assert trained_dead["collapsed"] is True
    live = interpret_scale_reports(
        [
            {
                "layer": 0,
                "dynamic_A_mean": 1.02,
                "dynamic_A_std": 0.02,
                "dynamic_A_min": 0.95,
                "dynamic_A_max": 1.08,
            }
        ],
        after_training=True,
    )
    assert live["live"] is True
    assert live["collapsed"] is False


def test_forecast_dynamic_a_wakes_controller_grads():
    cfg = ForecastModelConfig(
        n_features=8,
        d_model=32,
        n_layer=1,
        d_state=8,
        expand=2,
        dynamic_weights=True,
        dynamic_A=True,
        linear_skip=True,
    )
    model = ReturnForecaster(cfg)
    assert model.layers[0].mixer.controller is not None
    assert float(model.head.weight.detach().abs().sum()) > 0
    assert float(model.layers[0].mixer.out_proj.weight.detach().abs().sum()) > 0
    x = torch.randn(2, 12, 8)
    y = torch.randn(2, 12)
    mean, _log_sigma = model(x)
    health0 = model.collect_dynamic_health()
    assert health0["active"] is True
    assert abs(health0["mean"] - 1.0) < 0.15
    loss = (mean - y).pow(2).mean()
    loss.backward()
    g = controller_param_grad_norm(model)
    assert math.isfinite(g) and g > 0.0


def test_zero_head_and_out_proj_block_controller_grad():
    cfg = ForecastModelConfig(
        n_features=8,
        d_model=32,
        n_layer=1,
        d_state=8,
        expand=2,
        dynamic_weights=True,
        linear_skip=True,
    )
    model = ReturnForecaster(cfg)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    nn.init.zeros_(model.layers[0].mixer.out_proj.weight)
    x = torch.randn(2, 12, 8)
    y = torch.randn(2, 12)
    mean, _ = model(x)
    (mean - y).pow(2).mean().backward()
    g = controller_param_grad_norm(model)
    assert (not math.isfinite(g)) or g < 1e-12


def test_dynamic_weights_off_has_no_controller():
    cfg = ForecastModelConfig(
        n_features=8, d_model=32, n_layer=1, d_state=8, dynamic_weights=False
    )
    model = ReturnForecaster(cfg)
    assert model.layers[0].mixer.controller is None
    x = torch.randn(1, 8, 8)
    model(x)
    health = model.collect_dynamic_health()
    assert health["active"] is False
    _ascii_ok(format_dynamic_health_ascii(health))


def test_cs_promote_is_val_only_and_ignores_test():
    # TEST is better; VAL is not -- must not promote.
    no = decide_dynamic_a_cs_promote(
        dyn_val=0.02,
        skip_val=0.03,
        dyn_test=0.20,
        skip_test=0.03,
        val_lift=0.005,
    )
    assert no["promote"] is False
    assert no["gated_on"] == "val"
    _ascii_ok(no["reason"])
    yes = decide_dynamic_a_cs_promote(
        dyn_val=0.08,
        skip_val=0.03,
        dyn_test=-0.10,
        skip_test=0.03,
        val_lift=0.005,
    )
    assert yes["promote"] is True
    assert yes["test_lift"] < 0


def test_overnight_promote_requires_cover_and_ignores_test_keys():
    # Tiny cover must not promote even if up-rate is 90%.
    thin = decide_dynamic_a_overnight_promote(
        val_on={"sleeve_up_pct": 90.0, "sleeve_coverage": 0.01, "dir_pct": 51.2},
        val_off={"sleeve_up_pct": 50.0, "sleeve_coverage": 0.01, "dir_pct": 51.0},
        val_skip={"sleeve_up_pct": 54.0, "sleeve_coverage": 0.10, "dir_pct": 51.0},
    )
    assert thin["promote_dynamic_a"] is False
    assert thin["cover_ok"] is False
    _ascii_ok(thin["reason"])
    ok = decide_dynamic_a_overnight_promote(
        val_on={"sleeve_up_pct": 61.0, "sleeve_coverage": 0.08, "dir_pct": 52.0},
        val_off={"sleeve_up_pct": 57.0, "sleeve_coverage": 0.08, "dir_pct": 51.0},
        val_skip={"sleeve_up_pct": 56.0, "sleeve_coverage": 0.10, "dir_pct": 51.0},
    )
    assert ok["promote_dynamic_a"] is True
    assert ok["reached_60"] is True
    assert ok["gated_on"] == "val"


def test_blend_and_align_helpers():
    skip = np.array([1.0, 2.0, 3.0])
    enc = np.array([3.0, 2.0, 1.0])
    assert np.allclose(blend_residual(skip, enc, 0.0), skip)
    assert np.allclose(blend_residual(skip, enc, 1.0), enc)
    assert np.allclose(blend_residual(skip, enc, 0.5), np.array([2.0, 2.0, 2.0]))
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": [1, 2],
            "pred": [0.1, -0.2],
            "scale": [0.01, 0.02],
            "close": [10.0, 20.0],
            "y": [0.0, 0.0],
            "r_on": [0.0, 0.0],
        }
    )
    enc_df = pd.DataFrame({"symbol": ["A"], "date": [1], "enc_pred": [0.5]})
    aligned, n_hit = align_encoder_pred(frame, enc_df)
    assert n_hit == 1
    assert aligned[0] == 0.5
    assert aligned[1] == -0.2
    out = apply_residual_pred(frame, aligned)
    assert np.isfinite(out["pred_r"].to_numpy()).all()
    assert np.isfinite(out["implied_open"].to_numpy()).all()


def test_tiny_encoder_ablation_records_live_or_grad(tmp_path):
    """Short planted-synth train: ON has a controller and step-0 grads."""
    from forecast.overnight_dynamic import train_tiny_overnight_encoder
    from forecast.synthetic import write_cs_overnight_universe

    data_dir = tmp_path / "dyna"
    write_cs_overnight_universe(data_dir, n_names=8, n_days=90, seed=1, rho=0.65)
    on = train_tiny_overnight_encoder(
        str(data_dir),
        "synthetic",
        dynamic_weights=True,
        max_steps=6,
        ckpt_dir=tmp_path / "on",
        log_fn=None,
    )
    off = train_tiny_overnight_encoder(
        str(data_dir),
        "synthetic",
        dynamic_weights=False,
        max_steps=6,
        ckpt_dir=tmp_path / "off",
        log_fn=None,
    )
    assert on["controller_present"] is True
    assert off["controller_present"] is False
    assert math.isfinite(float(on["controller_grad_step0"]))
    assert float(on["controller_grad_step0"]) > 0.0
    init = on["health_init"]
    assert init["active"] is True
    assert abs(float(init["mean"]) - 1.0) < 0.2
    final = on["health_final"]
    assert final["active"] is True
    _ascii_ok(format_dynamic_health_ascii(init))
    _ascii_ok(format_dynamic_health_ascii(final))
    assert not on["preds"]["val"].empty
    # Off path must still produce last-bar preds (ablation is runnable).
    assert not off["preds"]["val"].empty
