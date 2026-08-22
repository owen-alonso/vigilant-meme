"""Parameter-count reporting that does not hide Dynamic A overhead."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _is_dynamic_param(name: str) -> bool:
    parts = name.split(".")
    return "controller" in parts or "modulator" in parts


def parameter_report(model: nn.Module) -> dict[str, Any]:
    """Split trainable parameters into baseline Mamba vs dynamic controller."""
    total = 0
    dynamic = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        total += n
        if _is_dynamic_param(name):
            dynamic += n
    base = total - dynamic
    overhead_pct = (100.0 * dynamic / base) if base else 0.0
    return {
        "mamba_parameters": base,
        "dynamic_controller_parameters": dynamic,
        "dynamic_overhead_pct": overhead_pct,
        "total_parameters": total,
    }


def format_parameter_report(report: dict[str, Any]) -> str:
    return (
        f"Mamba parameters: {report['mamba_parameters']:,}\n"
        f"Dynamic controller parameters: {report['dynamic_controller_parameters']:,}\n"
        f"Dynamic overhead: {report['dynamic_overhead_pct']:.4f}%\n"
        f"Total model parameters: {report['total_parameters']:,}"
    )
