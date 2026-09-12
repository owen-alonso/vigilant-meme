"""ASCII-safe Dynamic A controller health (Windows cp1252-safe).

The LM logger still uses a plus/minus glyph in ``format_dynamic_diagnostics``.
Forecast / overnight scripts print through this module so a cp1252 console
cannot crash on scale stats.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

COLLAPSE_STD = 1e-4
COLLAPSE_MEAN_TOL = 1e-3
SATURATE_STD = 0.02
SATURATE_NEAR = 0.05


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def controller_param_grad_norm(model: Any) -> float:
    """L2 norm of Dynamic A controller grads. NaN if none recorded."""
    total = 0.0
    n_grads = 0
    layers = getattr(model, "layers", None)
    if layers is None:
        return float("nan")
    for layer in layers:
        mixer = getattr(layer, "mixer", None)
        controller = getattr(mixer, "controller", None) if mixer is not None else None
        if controller is None:
            continue
        for param in controller.parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            total += float(grad.detach().float().pow(2).sum().item())
            n_grads += 1
    if n_grads <= 0:
        return float("nan")
    return float(math.sqrt(total))


def controller_is_present(model: Any) -> bool:
    layers = getattr(model, "layers", None)
    if layers is None:
        return False
    for layer in layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is not None and getattr(mixer, "controller", None) is not None:
            return True
    return False


def interpret_scale_reports(
    reports: Sequence[Mapping[str, Any]] | None,
    *,
    strength: float = 0.1,
    scale_eps: float = 1e-4,
    controller_grad_norm: float | None = None,
    after_training: bool = False,
) -> dict[str, Any]:
    """Classify Dynamic A scales: init-identity, live, collapsed, or saturated."""
    rows = [dict(r) for r in (reports or [])]
    if not rows:
        return {
            "active": False,
            "live": False,
            "collapsed": True,
            "init_identity": False,
            "saturated": False,
            "reason": "no Dynamic A scale recorded (controller off or no forward)",
            "layers": [],
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_layers": 0.0,
            "controller_grad_norm": (
                float("nan")
                if controller_grad_norm is None
                else _as_float(controller_grad_norm)
            ),
            "clamp_min": float(scale_eps),
            "clamp_max": float(1.0 + float(strength)),
        }
    means = [_as_float(r.get("dynamic_A_mean")) for r in rows]
    stds = [_as_float(r.get("dynamic_A_std")) for r in rows]
    mins = [_as_float(r.get("dynamic_A_min")) for r in rows]
    maxs = [_as_float(r.get("dynamic_A_max")) for r in rows]
    mean = float(sum(means) / len(means)) if means else float("nan")
    std = float(max(stds)) if stds else float("nan")
    mn = float(min(mins)) if mins else float("nan")
    mx = float(max(maxs)) if maxs else float("nan")
    clamp_min = float(scale_eps)
    clamp_max = float(1.0 + float(strength))
    init_identity = bool(
        math.isfinite(std)
        and std < COLLAPSE_STD
        and math.isfinite(mean)
        and abs(mean - 1.0) < COLLAPSE_MEAN_TOL
    )
    live = bool(math.isfinite(std) and std >= COLLAPSE_STD)
    collapsed = bool(init_identity and after_training)
    saturated = bool(
        math.isfinite(std)
        and std < SATURATE_STD
        and math.isfinite(mean)
        and (
            abs(mean - clamp_max) < SATURATE_NEAR
            or abs(mean - clamp_min) < SATURATE_NEAR
        )
    )
    if not rows:
        reason = "no Dynamic A scale recorded"
    elif live and not saturated:
        reason = "live: token/bar-dependent A scales (std>0, not clamped)"
    elif saturated:
        reason = "saturated: scales stuck near clamp"
    elif init_identity and after_training:
        reason = "collapsed: s~1 with near-zero std after training"
    elif init_identity:
        reason = "init identity: s=1 (zero controller out_proj); expected at step 0"
    else:
        reason = "Dynamic A scales recorded"
    return {
        "active": True,
        "live": live and not saturated,
        "collapsed": collapsed,
        "init_identity": init_identity,
        "saturated": saturated,
        "reason": reason,
        "layers": rows,
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "n_layers": float(len(rows)),
        "controller_grad_norm": (
            float("nan")
            if controller_grad_norm is None
            else _as_float(controller_grad_norm)
        ),
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
    }


def format_dynamic_health_ascii(health: Mapping[str, Any] | None) -> str:
    """Compact one-line health. ASCII only (no plus/minus glyph, no arrows)."""
    h = dict(health or {})
    if not h:
        return "A_scale=off"
    if not bool(h.get("active")):
        return "A_scale=off"
    grad = _as_float(h.get("controller_grad_norm"))
    grad_s = "nan" if not math.isfinite(grad) else f"{grad:.3e}"
    flags = []
    if bool(h.get("live")):
        flags.append("live")
    if bool(h.get("init_identity")):
        flags.append("init_s1")
    if bool(h.get("collapsed")):
        flags.append("collapsed")
    if bool(h.get("saturated")):
        flags.append("saturated")
    flag_s = ",".join(flags) if flags else "recorded"
    parts = []
    for row in h.get("layers") or []:
        parts.append(
            f"L{int(_as_float(row.get('layer'), 0.0))}="
            f"{_as_float(row.get('dynamic_A_mean')):.4f}"
            f"+/-{_as_float(row.get('dynamic_A_std')):.4f}"
            f"[{_as_float(row.get('dynamic_A_min')):.3f},"
            f"{_as_float(row.get('dynamic_A_max')):.3f}]"
        )
    body = ", ".join(parts) if parts else (
        f"mean={_as_float(h.get('mean')):.4f} "
        f"std={_as_float(h.get('std')):.4f} "
        f"[{_as_float(h.get('min')):.3f},{_as_float(h.get('max')):.3f}]"
    )
    return f"A_scale[{body}] {flag_s} grad={grad_s}"
