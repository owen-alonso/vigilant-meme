"""Historic pretrain then recent fine-tune from the staged cuts file.

Desktop (Yahoo liquid, skip-only overnight residual)::

    python scripts/pretrain_finetune.py --data-dir data --universe liquid

Reads ``data/_panel_cache/pretrain_finetune_cuts.json`` when present, else
the checked-in ``forecast/pretrain_finetune_cuts.json``.

Phase 1 fits the frozen historic skip (train < 2022-09-08, val for
early-stop / checkpoint). Phase 2 refits on the rolling 3y window with
``time_upweight_recent`` half-life 126 sessions. Residual labels are not
rewritten. VAL-gate the FT checkpoint on cost-aware live_locate IR, not
hit-rate. ``num_workers`` stays 0 so Windows does not pickle the CS cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from forecast.training import build_arg_parser, run_pretrain_finetune_schedule


def build_parser() -> argparse.ArgumentParser:
    p = build_arg_parser()
    p.description = (
        "Pretrain (frozen historic) then fine-tune (recent, session-rank recency) "
        "from pretrain_finetune_cuts.json. Skip-only overnight by default."
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not str(getattr(args, "phase", "") or ""):
        args.phase = "both"
    if not bool(getattr(args, "skip_only", False)):
        args.skip_only = True
    if not str(getattr(args, "label_return", "") or "") or args.label_return == "close":
        args.label_return = "overnight"
    if int(getattr(args, "num_workers", 0) or 0) != 0:
        print(
            "forcing num_workers=0 (do not pickle the desktop CS / panel cache)",
            file=sys.stderr,
        )
        args.num_workers = 0
    if not str(getattr(args, "universe", "") or ""):
        args.universe = "liquid"
    if not str(getattr(args, "checkpoint_dir", "") or "") or args.checkpoint_dir == "checkpoints/forecast":
        args.checkpoint_dir = "checkpoints/forecast_ridge_overnight"
    run_pretrain_finetune_schedule(args)


if __name__ == "__main__":
    main()
