"""Historic pretrain -> recent fine-tune date-cut contract.

The desktop file is ``data/_panel_cache/pretrain_finetune_cuts.json``.
This module parses that JSON (or the checked-in schema at
``forecast/pretrain_finetune_cuts.json``) and applies the windows without
rewriting residual labels.

Session-rank recency (``time_upweight_recent``) is a *sample-weight* schedule
on fine-tune train only. ``ridge_date_halflife`` stays the calendar-day hook;
``session_halflife=126`` is the 126-session half-life the cuts file asks for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from forecast.config import DataConfig, ForecastTrainConfig
from mamba_lm.paths import REPO_ROOT, resolve_path


CUTS_SCHEMA = "pretrain_finetune_cuts/v1"
CUTS_FILENAME = "pretrain_finetune_cuts.json"
IN_REPO_CUTS = REPO_ROOT / "forecast" / CUTS_FILENAME
DESKTOP_CUTS_REL = Path("data") / "_panel_cache" / CUTS_FILENAME

PhaseName = Literal["pretrain", "finetune"]


class CutsError(ValueError):
    """Invalid cuts JSON or phase window."""


def parse_iso_date(value: Any, *, field: str) -> str:
    """Require ``YYYY-MM-DD``. Returns the normalized date string."""
    raw = str(value or "").strip()
    if not raw:
        raise CutsError(f"{field} is required (YYYY-MM-DD)")
    try:
        day = np.datetime64(raw, "D")
    except ValueError as exc:
        raise CutsError(f"{field}={raw!r} is not YYYY-MM-DD") from exc
    out = str(day)
    if len(out) != 10 or out[4] != "-" or out[7] != "-":
        raise CutsError(f"{field}={raw!r} is not YYYY-MM-DD")
    return out


def exclusive_end(*, test_end: Any = "", test_through: Any = "", field: str) -> str:
    """Exclusive end date. ``test_through`` is inclusive (contract wording)."""
    end_raw = str(test_end or "").strip()
    through_raw = str(test_through or "").strip()
    if end_raw and through_raw:
        end = parse_iso_date(end_raw, field=f"{field}.test_end")
        through = parse_iso_date(through_raw, field=f"{field}.test_through")
        implied = str(np.datetime64(through, "D") + np.timedelta64(1, "D"))
        if end != implied:
            raise CutsError(
                f"{field}: test_end={end} disagrees with test_through={through} "
                f"(expected exclusive {implied})"
            )
        return end
    if end_raw:
        return parse_iso_date(end_raw, field=f"{field}.test_end")
    if through_raw:
        through = parse_iso_date(through_raw, field=f"{field}.test_through")
        return str(np.datetime64(through, "D") + np.timedelta64(1, "D"))
    raise CutsError(f"{field} needs test_end or test_through")


def session_rank_weights(
    dates: np.ndarray,
    halflife_sessions: float,
) -> np.ndarray:
    """Exponential recency by unique-session rank. Does not touch labels.

    ``w = 0.5 ** ((rank_max - rank[date]) / halflife)``. Unique dates in
    ``dates`` are the sessions (weekends already absent on a trading calendar).
    The latest session has weight 1. A session ``halflife`` ranks earlier has
    weight 0.5.
    """
    keys = np.asarray(dates, dtype=np.int64)
    if keys.size == 0:
        return np.ones((0,), dtype=np.float64)
    hl = float(halflife_sessions)
    if hl <= 0:
        return np.ones(keys.shape[0], dtype=np.float64)
    uniq = np.unique(keys)
    rank = {int(k): i for i, k in enumerate(uniq)}
    r = np.fromiter((rank[int(k)] for k in keys), dtype=np.float64, count=keys.size)
    return 0.5 ** ((float(r.max()) - r) / hl)


@dataclass(frozen=True)
class PhaseCuts:
    name: PhaseName
    start: str
    train_end: str
    val_end: str
    test_end: str
    test_through: str
    time_upweight_recent: bool
    halflife_sessions: float
    early_stop_sessions: int | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "train_end": self.train_end,
            "val_end": self.val_end,
            "test_end": self.test_end,
            "test_through": self.test_through,
            "time_upweight_recent": self.time_upweight_recent,
            "halflife_sessions": self.halflife_sessions,
            "early_stop_sessions": self.early_stop_sessions,
            "note": self.note,
        }


@dataclass(frozen=True)
class CutsFile:
    schema: str
    pretrain: PhaseCuts
    finetune: PhaseCuts
    val_gate: dict[str, Any]
    label_return: str
    universe: str
    skip_only: bool
    num_workers: int
    live_locate_defaults: dict[str, Any]
    dynamic_weights: dict[str, Any]
    source: str = ""

    def phase(self, name: str) -> PhaseCuts:
        key = str(name or "").strip().lower()
        if key == "pretrain":
            return self.pretrain
        if key == "finetune":
            return self.finetune
        raise CutsError(f"phase must be pretrain or finetune, got {name!r}")


def _phase_from_payload(name: PhaseName, raw: dict[str, Any], *, default_hl: float) -> PhaseCuts:
    if not isinstance(raw, dict):
        raise CutsError(f"{name} block must be an object")
    start = parse_iso_date(raw.get("start"), field=f"{name}.start")
    train_end = parse_iso_date(raw.get("train_end"), field=f"{name}.train_end")
    val_end = parse_iso_date(raw.get("val_end"), field=f"{name}.val_end")
    test_end = exclusive_end(
        test_end=raw.get("test_end", ""),
        test_through=raw.get("test_through", ""),
        field=name,
    )
    test_through = str(np.datetime64(test_end, "D") - np.timedelta64(1, "D"))
    if np.datetime64(train_end, "D") <= np.datetime64(start, "D"):
        raise CutsError(f"{name}: train_end {train_end} must be after start {start}")
    if np.datetime64(val_end, "D") <= np.datetime64(train_end, "D"):
        raise CutsError(f"{name}: val_end {val_end} must be after train_end {train_end}")
    if np.datetime64(test_end, "D") < np.datetime64(val_end, "D"):
        raise CutsError(f"{name}: test_end {test_end} must be >= val_end {val_end}")
    upweight = bool(raw.get("time_upweight_recent", name == "finetune"))
    hl = float(raw.get("halflife_sessions", default_hl if upweight else 0.0) or 0.0)
    if upweight and hl <= 0:
        raise CutsError(f"{name}: time_upweight_recent requires halflife_sessions > 0")
    early = raw.get("early_stop_sessions")
    return PhaseCuts(
        name=name,
        start=start,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        test_through=test_through,
        time_upweight_recent=upweight,
        halflife_sessions=hl if upweight else 0.0,
        early_stop_sessions=int(early) if early not in (None, "") else None,
        note=str(raw.get("note") or ""),
    )


def validate_cuts_pair(pretrain: PhaseCuts, finetune: PhaseCuts) -> None:
    if np.datetime64(pretrain.test_end, "D") > np.datetime64(finetune.start, "D"):
        raise CutsError(
            f"pretrain.test_end {pretrain.test_end} must be <= finetune.start "
            f"{finetune.start} (pretrain ends before FT)"
        )
    if finetune.time_upweight_recent is False:
        raise CutsError("finetune must apply time_upweight_recent on train")
    if pretrain.time_upweight_recent:
        raise CutsError("pretrain is frozen; do not time-upweight historic train")


def cuts_from_mapping(payload: dict[str, Any], *, source: str = "") -> CutsFile:
    if not isinstance(payload, dict):
        raise CutsError("cuts JSON must be an object")
    schema = str(payload.get("schema") or CUTS_SCHEMA)
    dyn = payload.get("dynamic_weights") or {}
    if not isinstance(dyn, dict):
        raise CutsError("dynamic_weights must be an object")
    if dyn and str(dyn.get("kind") or "time_upweight_recent") != "time_upweight_recent":
        raise CutsError(
            f"dynamic_weights.kind must be time_upweight_recent, got {dyn.get('kind')!r}"
        )
    if dyn.get("rewrite_labels", False):
        raise CutsError("dynamic_weights must not rewrite residual labels")
    apply_to = str(dyn.get("apply_to") or "finetune_train")
    if apply_to not in ("finetune_train", "finetune", ""):
        raise CutsError("time_upweight_recent applies to finetune_train only")
    default_hl = float(dyn.get("halflife_sessions") or 126.0)
    pretrain = _phase_from_payload("pretrain", payload.get("pretrain") or {}, default_hl=default_hl)
    finetune = _phase_from_payload("finetune", payload.get("finetune") or {}, default_hl=default_hl)
    validate_cuts_pair(pretrain, finetune)
    gate = payload.get("val_gate") or {}
    if not isinstance(gate, dict):
        raise CutsError("val_gate must be an object")
    locate = payload.get("live_locate_defaults") or {
        "quantile": 0.2,
        "locate_haircut": 0.5,
        "max_short_gross": 0.5,
    }
    return CutsFile(
        schema=schema,
        pretrain=pretrain,
        finetune=finetune,
        val_gate=dict(gate),
        label_return=str(payload.get("label_return") or "overnight"),
        universe=str(payload.get("universe") or "liquid"),
        skip_only=bool(payload.get("skip_only", True)),
        num_workers=int(payload.get("num_workers", 0) or 0),
        live_locate_defaults=dict(locate),
        dynamic_weights=dict(dyn) if dyn else {
            "kind": "time_upweight_recent",
            "apply_to": "finetune_train",
            "halflife_sessions": default_hl,
            "rewrite_labels": False,
        },
        source=source,
    )


def load_cuts(path: str | Path) -> CutsFile:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise CutsError(f"cuts file not found: {path}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CutsError(f"invalid JSON in {resolved}: {exc}") from exc
    return cuts_from_mapping(payload, source=str(resolved))


def resolve_cuts_path(
    explicit: str | Path | None = None,
    *,
    data_dir: str | Path = "data",
) -> Path:
    """Prefer an explicit path, then the desktop panel-cache copy, then in-repo."""
    if explicit:
        candidate = resolve_path(explicit)
        if candidate.is_file():
            return candidate
        raise CutsError(f"cuts file not found: {explicit}")
    desktop = resolve_path(Path(data_dir) / "_panel_cache" / CUTS_FILENAME)
    if desktop.is_file():
        return desktop
    if IN_REPO_CUTS.is_file():
        return IN_REPO_CUTS
    raise CutsError(
        f"no cuts file at {desktop} or {IN_REPO_CUTS}; "
        "pass --cuts-json or copy forecast/pretrain_finetune_cuts.json "
        "to data/_panel_cache/"
    )


def default_cuts() -> CutsFile:
    return load_cuts(IN_REPO_CUTS)


def apply_phase_to_data_cfg(data_cfg: DataConfig, phase: PhaseCuts) -> DataConfig:
    """Pin chronological windows. Residual-label fields are left unchanged."""
    return replace(
        data_cfg,
        train_from=phase.start,
        train_end=phase.train_end,
        val_end=phase.val_end,
        test_end=phase.test_end,
    )


def apply_phase_to_train_cfg(train_cfg: ForecastTrainConfig, phase: PhaseCuts) -> ForecastTrainConfig:
    """Fine-tune train gets session-rank recency; pretrain stays uniform.

    ``ridge_date_halflife`` is the existing calendar-day hook. The cuts file
    asks for 126 *sessions*, so FT sets ``time_upweight_*`` / session
    half-life and leaves the calendar-day hook at 0 unless the caller set it.
    """
    up = bool(phase.time_upweight_recent)
    hl = float(phase.halflife_sessions) if up else 0.0
    return replace(
        train_cfg,
        time_upweight_recent=up,
        time_upweight_halflife_sessions=hl,
        forecast_phase=phase.name,
    )


def apply_phase(
    data_cfg: DataConfig,
    train_cfg: ForecastTrainConfig,
    cuts: CutsFile,
    phase_name: str,
) -> tuple[DataConfig, ForecastTrainConfig, PhaseCuts]:
    phase = cuts.phase(phase_name)
    data_out = apply_phase_to_data_cfg(data_cfg, phase)
    if cuts.label_return and not str(getattr(data_cfg, "label_return", "") or "").strip():
        data_out = replace(data_out, label_return=cuts.label_return)
    train_out = apply_phase_to_train_cfg(train_cfg, phase)
    if cuts.skip_only:
        train_out = replace(train_out, skip_only=True)
    if int(cuts.num_workers) == 0:
        train_out = replace(train_out, num_workers=0)
    return data_out, train_out, phase


def val_gate_commands(
    *,
    checkpoint_dir: str = "checkpoints/forecast_ridge_overnight_finetune",
    data_dir: str = "data",
    universe: str = "liquid",
    baseline_ir: float = 5.58,
) -> str:
    """Desktop Yahoo liquid commands to VAL-gate FT on cost-aware live_locate IR."""
    ckpt = checkpoint_dir.rstrip("/")
    return (
        "# VAL-gate the FT checkpoint on cost-aware IR (live_locate), not hit-rate.\n"
        "# Promote only if FT VAL unlevered net IR beats the prior overnight "
        f"baseline (~+{baseline_ir:.2f} liquid live_locate).\n"
        "# TEST is report-only. Do not reopen overnight-up 60% or expand Mamba.\n"
        "# Keep live_locate q20 / haircut 0.50 / short 0.50 unless VAL IR promotes.\n"
        f"python scripts/overnight_shorting.py --data-dir {data_dir} --universe {universe} \\\n"
        f"    --json {ckpt}/shorting.json\n"
        f"python -m forecast.backtest --checkpoint {ckpt}/best.pt \\\n"
        "    --holding overnight --live-costs\n"
        "# Look at VAL live_locate unlevered_net_ir (DATE GATES / books.live_locate).\n"
        "# Do not flip LS vs long-only off TEST. Reported IR is provisional.\n"
    )
