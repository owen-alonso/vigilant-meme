"""(2a) Direct P(up) overnight head — VAL-gated hit-rate, TEST report-only.

Desktop Yahoo liquid (mmap train path, num_workers=0)::

    python -m forecast.panel_mmap --data-dir data --universe liquid --write
    python -m forecast.training --universe liquid --interval daily --skip-only \\
      --label-return overnight --pup-head --num-workers 0 \\
      --mmap-manifest data/_panel_cache/mmap_manifest.json \\
      --checkpoint-dir checkpoints/forecast_pup_overnight
    python scripts/overnight_pup.py --data-dir data --universe liquid \\
      --checkpoint checkpoints/forecast_pup_overnight/best.pt \\
      --json checkpoints/forecast_pup_overnight/pup.json

    # Book IR is report-only on this cut (do not retarget TEST):
    python -m forecast.backtest --checkpoint checkpoints/forecast_pup_overnight/best.pt \\
      --holding overnight --live-costs

Cloud / CI (no Yahoo tape)::

    python scripts/overnight_pup.py --synthetic
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

import numpy as np

from forecast.accuracy import overnight_skip_data_config
from forecast.data import build_datasets
from forecast.pup import evaluate_pup, fit_pup_logistic, format_pup_block, labelled_pup_rows
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


def spec_from_checkpoint(path: Path, n_features: int) -> dict[str, Any] | None:
    from forecast.checkpoint import load_forecaster
    import torch

    model, _state = load_forecaster(path, torch.device("cpu"))
    w = model.up_skip.weight.detach().cpu().numpy().reshape(-1)
    b = float(model.up_skip.bias.detach().cpu().numpy().reshape(-1)[0])
    if w.size != int(n_features):
        return None
    if float(np.abs(w).sum()) + abs(b) < 1e-12:
        return None
    return {"weights": w, "bias": b, "ridge": 1.0, "n": 0, "source": "checkpoint"}


def run(
    *,
    data_dir: str,
    universe: str,
    checkpoint: str = "",
    json_path: str = "",
    mmap_manifest: str = "",
    pit_dir: str = "",
    pup_ridge: float = 1.0,
) -> dict[str, Any]:
    cfg = overnight_skip_data_config(data_dir, universe=universe, label_return="overnight")
    cfg.mmap_manifest = mmap_manifest
    cfg.use_mmap = True
    cfg.pit_dir = pit_dir
    bundle = build_datasets(cfg, log_fn=print)
    spec = None
    if checkpoint:
        spec = spec_from_checkpoint(Path(checkpoint), int(bundle["feature_mean"].shape[0]))
        if spec is None:
            print("checkpoint has no fitted P(up) head; fitting TRAIN logistic")
    if spec is None:
        x, y, _d = labelled_pup_rows(
            bundle["train_symbols"], bundle["feature_mean"], bundle["feature_std"]
        )
        spec = fit_pup_logistic(x, y, ridge=float(pup_ridge))
        spec["source"] = "train_refit"
    payload = evaluate_pup(bundle, spec)
    payload["mmap_manifest"] = bundle.get("mmap_manifest") or ""
    payload["mmap"] = bool(bundle.get("mmap"))
    print(format_pup_block(payload))
    if json_path:
        dest = Path(json_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
        print(f"wrote {dest}")
    return payload


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="VAL-gated P(up) overnight-up head (2a)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--universe", default="liquid")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--json", default="", dest="json_path")
    p.add_argument("--mmap-manifest", default="")
    p.add_argument("--pit-dir", default="")
    p.add_argument("--pup-ridge", type=float, default=1.0)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args(argv)
    if args.synthetic:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            write_cs_overnight_universe(data, n_names=12, n_days=160, seed=3)
            run(
                data_dir=str(data),
                universe="",
                checkpoint=args.checkpoint,
                json_path=args.json_path,
                mmap_manifest=args.mmap_manifest,
                pit_dir=args.pit_dir,
                pup_ridge=args.pup_ridge,
            )
        return
    run(
        data_dir=args.data_dir,
        universe=args.universe,
        checkpoint=args.checkpoint,
        json_path=args.json_path,
        mmap_manifest=args.mmap_manifest,
        pit_dir=args.pit_dir,
        pup_ridge=args.pup_ridge,
    )


if __name__ == "__main__":
    main()
