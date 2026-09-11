"""VAL-gated overnight long-short vs long-only (TEST report-only).

    python scripts/overnight_shorting.py --synthetic
    python scripts/overnight_shorting.py --data-dir data --universe liquid
    python scripts/overnight_shorting.py --data-dir data --universe liquid \\
        --json checkpoints/forecast_ridge_overnight/shorting.json

Fits the PR #5 overnight skip on TRAIN. Locked VAL decides whether honest
``live_locate`` LS beats ``live_long_only``. Locked TEST is printed and
never used as a gate. Cloud VM has no Yahoo liquid tape — use --synthetic
here; run the liquid command on desktop.
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

from forecast.shorting import evaluate_overnight_shorting, format_shorting_report
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
        description="Overnight long-top / short-bottom vs long-only. "
        "Promote on locked VAL; report locked TEST."
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
        "--vol-target",
        type=float,
        default=0.15,
        help="annualized vol target for the levered path (default 15%%)",
    )
    args = p.parse_args(argv)
    if args.synthetic:
        args.universe = "synthetic"
        if not args.data_dir:
            args.data_dir = "/tmp/cs_overnight_shorting_synth"
        write_cs_overnight_universe(
            args.data_dir,
            n_names=12,
            n_days=220,
            seed=1,
            rho=0.65,
            include_sectors=True,
        )
        print(f"synthetic overnight universe -> {args.data_dir}", flush=True)
    if not args.data_dir:
        print("need --data-dir or --synthetic", file=sys.stderr)
        return 2
    payload = evaluate_overnight_shorting(
        args.data_dir,
        args.universe,
        log_fn=print,
        vol_target=float(args.vol_target),
    )
    print(format_shorting_report(payload), flush=True)
    if args.json:
        dest = Path(args.json)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
        print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
