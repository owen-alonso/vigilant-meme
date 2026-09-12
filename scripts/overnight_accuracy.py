"""Locked-TEST overnight skip: direction accuracy and next-open price error.

    python scripts/overnight_accuracy.py --synthetic
    python scripts/overnight_accuracy.py --synthetic --try-dynamic-a
    python scripts/overnight_accuracy.py --data-dir data --universe liquid
    python scripts/overnight_accuracy.py --data-dir data --universe liquid \\
        --json checkpoints/forecast_ridge_overnight/accuracy.json \\
        --calibrate-json checkpoints/forecast_ridge_overnight/overnight_calibrate.json

Fits the PR #5 overnight skip on TRAIN. Scores residual*sigma on locked TEST
(the PR #8 baseline). Additional readouts (affine, drift-veto vs always-up,
piecewise/bin calibration, weekday intercepts, TS overnight ridge, sign ridge,
ADV sleeve, confidence slices, conditional high-|pred| left-tail/confidence
blend, book-aligned top-q residual overnight-up, long-only book up-rate) are
fit on TRAIN and gated on locked VAL — never on TEST. Direction skill is
excess vs the unconditional overnight-up rate (~54.3% on liquid TEST), not
vs a coin flip. Book-aligned sleeve up-rate is the primary accuracy object
vs that floor; pooled TS direction stays report-only. IDEA F gates that
sleeve as an optional live_long_only path on VAL unlev net IR / max DD.
IDEA G fits residual→overnight MAE maps on the sector-overnight skip
(affine_l1 / piecewise_l1 / huber_affine / bin_calibrate) and promotes a
new MAE default only on a clear locked-VAL % MAE margin.
IDEA H scores within-date sign(pred − CS median) vs sign(r_on − CS median)
on the sector-overnight skip (hit vs 50%) and the absolute overnight-up of
the long half (pred > CS median) vs the uncond floor / top-20%. Optional
TRAIN |pred−median| floor. Live q20 unchanged.
IDEA I stacks H's long-half with E's TRAIN top-q / |pred| floor (light
re-grid on TRAIN relative hit). Absolute up vs E; live IR vs q20 (F gate).
IDEA J mirrors E on the short side: overnight down-rate of TRAIN bottom-q
/ |pred| names vs the down-floor and vs bottom-20%. Optional live_locate
LS vs long-only q20. Live q20 unchanged on hit-rate-only.
IDEA K builds a symmetric long-E / short-J LS (paper zero-cost + live_locate)
and promotes the live path only if VAL IR ≥ q20+0.05 and DD is not worse
by >0.05.
IDEA L is a two-stage TRAIN-only next-open MAE map (residual*sigma → gap,
then gap → next-open $/% with close_t) plus a residual+DOW+vol ridge.
Promote a new MAE default only if VAL % MAE beats residual×σ / zero-move /
train-median by ≥0.5 bp and is not worse than the current default.
IDEA M is sparse MAE: TRAIN affine_l1/huber on residual*sigma->r_on, then a
TRAIN |pred| tau so small gaps predict 0 (zero-move). Same VAL % MAE gate.
--try-dynamic-a trains a tiny Dynamic A encoder (VAL-gated residual blend;
steps 0 = auto one-epoch cap 400 on liquid-scale panels).
IDEA N enumerates rank sleeves on TRAIN then selects on locked VAL
(pred_r / ridged P(up) / gap-filter / CS z). Aimed at VAL 60% with
cover >= 5%. TEST never picks. Dynamic A alpha-blend is not the 60% path.
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
    p.add_argument(
        "--try-dynamic-a",
        action="store_true",
        help="train tiny Dynamic A encoder; VAL-gate overnight residual blend",
    )
    p.add_argument(
        "--dynamic-a-steps",
        type=int,
        default=0,
        help="AdamW steps for the tiny Dynamic A encoder (0=auto: one epoch, cap 400)",
    )
    p.add_argument(
        "--dynamic-a-ckpt",
        default="",
        help="checkpoint dir for the tiny Dynamic A encoders",
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
        try_dynamic_a=bool(args.try_dynamic_a),
        dynamic_a_steps=int(args.dynamic_a_steps),
        dynamic_a_ckpt=str(args.dynamic_a_ckpt or ""),
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
