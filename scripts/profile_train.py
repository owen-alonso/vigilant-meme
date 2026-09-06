"""Profile the forecast training step: per-phase timing + torch.profiler.

Replicates the exact _train() step from forecast/training.py on real data,
with explicit synchronization between phases so each phase is attributed
correctly. Synchronization inflates total step time slightly; the unsynced
end-to-end step time is measured separately.

Usage:
    python scripts/profile_train.py [--steps 100] [--batch-size 16] [--trace]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from forecast.config import DataConfig, ForecastModelConfig, ForecastTrainConfig
from forecast.data import build_datasets
from forecast.model import ReturnForecaster
from forecast.training import masked_loss
from mamba_lm.reporting import clip_grad_norm_unique
from mamba_lm.training_utils import (
    autocast_context,
    build_optimizer,
    cycle_loader,
    grads_finite,
    set_seed,
)


class PhaseTimer:
    def __init__(self) -> None:
        self.phases: dict[str, list[float]] = {}
        self._t = 0.0

    def start(self) -> None:
        torch.cuda.synchronize()
        self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.phases.setdefault(name, []).append((now - self._t) * 1e3)
        self._t = now

    def report(self) -> None:
        total = sum(statistics.mean(v) for v in self.phases.values())
        print(f"\n--- per-phase mean over {len(next(iter(self.phases.values())))} steps (synced) ---")
        for name, vals in self.phases.items():
            m = statistics.mean(vals)
            print(f"{name:<16} {m:8.3f} ms  {100 * m / total:5.1f}%")
        print(f"{'total':<16} {total:8.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--interval", default="weekly")
    parser.add_argument("--trace", action="store_true", help="write torch.profiler trace")
    args = parser.parse_args()

    device = torch.device("cuda")
    set_seed(0)
    data_cfg = DataConfig(interval=args.interval)
    model_cfg = ForecastModelConfig()
    train_cfg = ForecastTrainConfig(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    bundle = build_datasets(data_cfg, log_fn=None)
    model_cfg.n_features = len(bundle["feature_names"])
    model = ReturnForecaster(model_cfg).to(device)
    model.train()
    optimizer = build_optimizer(model, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    loader = DataLoader(
        bundle["datasets"]["train"],
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    print(
        f"train windows={len(bundle['datasets']['train'])} "
        f"batches/epoch={len(loader)} batch={train_cfg.batch_size} "
        f"seq_len={data_cfg.seq_len} precision={train_cfg.precision}"
    )
    data_iter = cycle_loader(loader)

    def one_step(timer: PhaseTimer | None):
        if timer:
            timer.start()
        x, y, mask, _scale = next(data_iter)
        if timer:
            timer.mark("data_fetch")
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        if timer:
            timer.mark("h2d_transfer")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_cfg.precision):
            mean, log_sigma = model(x)
        if timer:
            timer.mark("forward")
        loss = masked_loss(mean.float(), log_sigma.float(), y, mask, train_cfg)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        if timer:
            timer.mark("loss+finite")
        loss.backward()
        if timer:
            timer.mark("backward")
        if not grads_finite(model):
            raise RuntimeError("non-finite grads")
        if timer:
            timer.mark("grads_finite")
        clip_grad_norm_unique(model, train_cfg.grad_clip)
        if timer:
            timer.mark("grad_clip")
        optimizer.step()
        if timer:
            timer.mark("optimizer")
        float(loss.detach())
        if timer:
            timer.mark("loss_logging")

    for _ in range(args.warmup):
        one_step(None)
    torch.cuda.synchronize()

    # Unsynced end-to-end step time (what training actually experiences).
    t0 = time.perf_counter()
    for _ in range(args.steps):
        one_step(None)
    torch.cuda.synchronize()
    unsynced_ms = (time.perf_counter() - t0) * 1e3 / args.steps

    timer = PhaseTimer()
    for _ in range(args.steps):
        one_step(timer)

    tokens = args.batch_size * data_cfg.seq_len
    print(f"\nend-to-end step (no phase syncs): {unsynced_ms:.3f} ms "
          f"({tokens / unsynced_ms * 1e3:,.0f} labelled-window tokens/s)")
    timer.report()
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")

    if args.trace:
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        ) as prof:
            for _ in range(10):
                one_step(None)
            torch.cuda.synchronize()
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=25
        ))
        out = _REPO_ROOT / "checkpoints" / "profile_trace.json"
        out.parent.mkdir(exist_ok=True)
        prof.export_chrome_trace(str(out))
        print(f"trace -> {out}")


if __name__ == "__main__":
    main()
