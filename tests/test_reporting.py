"""Parameter reporting and future-flag guards."""

from __future__ import annotations

import pytest

from mamba_lm.config import MambaConfig
from mamba_lm.model import MambaLM
from mamba_lm.reporting import format_parameter_report, parameter_report
from tests.helpers import dynamic_config, tiny_config


def test_dynamic_overhead_is_reported_and_small():
    base = MambaLM(tiny_config(dynamic_weights=False))
    dyn = MambaLM(dynamic_config())
    base_r = parameter_report(base)
    dyn_r = parameter_report(dyn)

    assert base_r["dynamic_controller_parameters"] == 0
    assert dyn_r["dynamic_controller_parameters"] > 0
    assert dyn_r["mamba_parameters"] == base_r["mamba_parameters"]
    assert dyn_r["total_parameters"] == dyn_r["mamba_parameters"] + dyn_r["dynamic_controller_parameters"]
    assert dyn_r["dynamic_overhead_pct"] < 50.0

    text = format_parameter_report(dyn_r)
    assert "Mamba parameters:" in text
    assert "Dynamic controller parameters:" in text
    assert "Dynamic overhead:" in text
    assert "Total model parameters:" in text


def test_unimplemented_flags_raise():
    with pytest.raises(NotImplementedError):
        MambaConfig(d_model=32, n_layer=1, dynamic_weights=True, dynamic_B=True)
    with pytest.raises(NotImplementedError):
        MambaConfig(d_model=32, n_layer=1, dynamic_weights=True, dynamic_dt=True)
    with pytest.raises(NotImplementedError):
        MambaConfig(
            d_model=32,
            n_layer=1,
            dynamic_weights=True,
            dynamic_parameterization="low_rank",
        )
