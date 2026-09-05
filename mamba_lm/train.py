"""Next-token training / validation loop with AMP and checkpointing."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mamba_lm.checkpoint import save_checkpoint
from mamba_lm.config import MambaConfig, TrainConfig
from mamba_lm.data import build_datasets
from mamba_lm.model import MambaLM, format_dynamic_diagnostics
from mamba_lm.paths import anchor_to_repo
from mamba_lm.reporting import (
    clip_grad_norm_unique,
    format_parameter_report,
    parameter_report,
)
from mamba_lm.training_utils import (
    autocast_context,
    build_optimizer,
    cycle_loader,
    grads_finite,
    keep_awake,
    lr_linear_warmup,
    require_nonempty_loader,
    select_device,
    set_seed,
)

__all__ = [
    "autocast_context",
    "build_optimizer",
    "evaluate",
    "select_device",
    "set_seed",
    "train",
]


@torch.no_grad()
def evaluate(
    model: MambaLM,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with autocast_context(device, precision):
            logits = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total_loss += float(loss.item()) * y.numel()
        total_tokens += y.numel()
        n_batches += 1
    model.train()
    if total_tokens == 0 or n_batches == 0:
        return {
            "val_loss": float("nan"),
            "val_ppl": float("nan"),
            "val_batches": 0.0,
        }
    mean_loss = total_loss / total_tokens
    return {
        "val_loss": mean_loss,
        "val_ppl": math.exp(min(mean_loss, 20.0)),
        "val_batches": float(n_batches),
    }


def train(
    model_cfg: MambaConfig,
    train_cfg: TrainConfig,
    *,
    device: torch.device | None = None,
    log_fn: Any | None = print,
) -> dict[str, Any]:
    """Run a controlled training session. Returns metrics for comparisons."""
    with keep_awake(log_fn=log_fn):
        return _train(
            model_cfg, train_cfg, device=device, log_fn=log_fn
        )


def _train(
    model_cfg: MambaConfig,
    train_cfg: TrainConfig,
    *,
    device: torch.device | None = None,
    log_fn: Any | None = print,
) -> dict[str, Any]:
    device = device or select_device()
    set_seed(train_cfg.seed)

    tokenizer, train_ds, val_ds = build_datasets(
        train_cfg.data_dir, train_cfg.seq_len, train_cfg.val_fraction
    )
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.vocab_size = tokenizer.vocab_size
    model = MambaLM(model_cfg).to(device)

    report = parameter_report(model)
    if log_fn:
        log_fn(format_parameter_report(report))
        log_fn(f"device={device} precision={train_cfg.precision} vocab={tokenizer.vocab_size}")
        autocast_context(device, train_cfg.precision, log_fn=log_fn)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=train_cfg.num_workers,
    )
    require_nonempty_loader(train_loader, "train")
    optimizer = build_optimizer(
        model, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    use_scaler = train_cfg.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    checkpoint_dir = anchor_to_repo(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if log_fn:
        log_fn(f"checkpoints -> {checkpoint_dir}")

    extra = {"tokenizer": tokenizer.to_dict(), "train_config": train_cfg.to_dict()}

    def _save(path: Path, step: int) -> None:
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_scaler else None,
            step=step,
            extra=extra,
        )

    train_losses: list[float] = []
    val_losses: list[float] = []
    grad_norms: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    tokens_seen = 0
    skipped_inf = 0
    best_val = float("inf")
    best_step = -1
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_iter = cycle_loader(train_loader)
    model.train()
    last_completed = 0
    interrupted = False
    try:
        for step in range(train_cfg.max_steps):
            lr = lr_linear_warmup(
                step, lr=train_cfg.lr, warmup_steps=train_cfg.warmup_steps
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = next(data_iter)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, train_cfg.precision):
                logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not grads_finite(model):
                skipped_inf += 1
                if log_fn:
                    log_fn(
                        f"step {step:5d}: non-finite gradients, skipping optimizer step "
                        f"(skipped={skipped_inf})"
                    )
                scaler.update()
                continue

            grad_norm = clip_grad_norm_unique(model, train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.item()))
            grad_norms.append(grad_norm)
            tokens_seen += y.numel()
            last_completed = step + 1

            if (step + 1) % train_cfg.log_interval == 0 or step == 0:
                elapsed = max(time.perf_counter() - t0, 1e-8)
                tps = tokens_seen / elapsed
                diag = model.collect_dynamic_diagnostics()
                if diag:
                    diagnostics.append({"step": step, "layers": diag})
                if log_fn:
                    msg = (
                        f"step {step:5d}/{train_cfg.max_steps}  "
                        f"loss={loss.item():.4f}  "
                        f"grad_norm={grad_norm:.3f}  "
                        f"lr={lr:.2e}  "
                        f"tok/s={tps:.0f}"
                    )
                    extra_diag = format_dynamic_diagnostics(diag)
                    if extra_diag:
                        msg += f"  {extra_diag}"
                    log_fn(msg)

            if (step + 1) % train_cfg.eval_interval == 0 or step + 1 == train_cfg.max_steps:
                val = evaluate(
                    model, val_loader, device, train_cfg.precision, train_cfg.eval_batches
                )
                val_losses.append(val["val_loss"])
                if log_fn:
                    log_fn(
                        f"eval step {step}  val_loss={val['val_loss']:.4f}  "
                        f"ppl={val['val_ppl']:.2f}"
                    )
                _save(checkpoint_dir / "last.pt", step)
                if math.isfinite(val["val_loss"]) and val["val_loss"] < best_val:
                    best_val = val["val_loss"]
                    best_step = step
                    _save(checkpoint_dir / "best.pt", step)
                    if log_fn:
                        log_fn(f"  new best val_loss={best_val:.4f} -> {checkpoint_dir / 'best.pt'}")
    except KeyboardInterrupt:
        interrupted = True
        _save(checkpoint_dir / "last.pt", max(0, last_completed - 1))
        if log_fn:
            log_fn(
                f"keyboard interrupt at step {last_completed}/{train_cfg.max_steps}; "
                f"saved {checkpoint_dir / 'last.pt'}"
            )

    elapsed = max(time.perf_counter() - t0, 1e-8)
    peak_mem = (
        float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    )
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "grad_norms": grad_norms,
        "diagnostics": diagnostics,
        "tokens_seen": tokens_seen,
        "tokens_per_sec": tokens_seen / elapsed,
        "elapsed_sec": elapsed,
        "peak_memory_bytes": peak_mem,
        "parameter_report": report,
        "config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "vocab_size": tokenizer.vocab_size,
        "device": str(device),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": best_val if math.isfinite(best_val) else None,
        "best_step": best_step,
        "interrupted": interrupted,
        "last_step": last_completed,
        "skipped_inf_steps": skipped_inf,
        "mean_grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
    }
