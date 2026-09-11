"""Locked-TEST overnight skip: direction accuracy and next-open price error.

    python scripts/overnight_accuracy.py --synthetic
    python scripts/overnight_accuracy.py --data-dir data --universe liquid
    python scripts/overnight_accuracy.py --data-dir data --universe liquid \\
        --json checkpoints/forecast_ridge_overnight/accuracy.json \\
        --calibrate-json checkpoints/forecast_ridge_overnight/overnight_calibrate.json

Fits the PR #5 overnight skip on TRAIN. Scores residual*sigma on locked TEST
(the PR #8 baseline). Additional readouts (affine, drift-veto vs always-up,
piecewise/bin calibration, weekday intercepts, TS overnight ridge, sign ridge,
ADV sleeve, confidence slices, long-only book up-rate) are fit on TRAIN and
gated on locked VAL — never on TEST. Direction skill is excess vs the
unconditional overnight-up rate (~54.3% on liquid TEST), not vs a coin flip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from forecast.accuracy import evaluate_overnight_accuracy, format_accuracy_report
from forecast.synthetic import write_cs_overnight_universe


def _json_ready(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="TRUE/FALSE overnight accuracy on the locked TEST window, "
        "with train-only readouts gated on locked VAL."
    )
    p.add_argument("--data-dir", default="")
    p.add_argument(
        "--universe",
        default="liquid",
        choices=("", "liquid", "liquid_wide", "synthetic"),
    )
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--json",
        default="",
        help="optional path to write the full metric payload",
    )
    p.add_argument(
        "--calibrate-json",
        default="",
        help="optional path for the VAL-gated affine {a,b} used by generate.py",
    )
    p.add_argument(
        "--no-ablate",
        action="store_true",
        help="PR #8 residual*sigma TEST only (skip train/val readouts)",
    )
    args = p.parse_args(argv)
    if args.synthetic:
        args.universe = "synthetic"
        if not args.data_dir:
            args.data_dir = "/tmp/cs_overnight_accuracy_synth"
        write_cs_overnight_universe(
            args.data_dir, n_names=12, n_days=220, seed=1, rho=0.65
        )
        print(f"synthetic overnight universe -> {args.data_dir}", flush=True)
    if not args.data_dir:
        print("need --data-dir or --synthetic", file=sys.stderr)
        return 2
    payload = evaluate_overnight_accuracy(
        args.data_dir,
        args.universe,
        log_fn=print,
        ablate=not args.no_ablate,
    )
    print(format_accuracy_report(payload), flush=True)
    if args.json:
        dest = Path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
        print(f"wrote {dest}", flush=True)
    if args.calibrate_json:
        dest = Path(args.calibrate_json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_json_ready(payload.get("calibrate") or {}), indent=2) + "\n")
        print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
