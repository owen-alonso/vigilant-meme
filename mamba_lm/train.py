"""Next-token training / validation loop with AMP and checkpointing."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mamba_lm.checkpoint import save_checkpoint
from mamba_lm.config import MambaConfig, TrainConfig
from mamba_lm.data import build_datasets
from mamba_lm.model import MambaLM
from mamba_lm.reporting import format_parameter_report, parameter_report


def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    enabled = True
    if device.type == "cpu" and precision == "fp16":
        # CPU fp16 autocast is not reliably implemented; fall back to bf16 if possible.
        if torch.cpu.is_bf16_supported() if hasattr(torch.cpu, "is_bf16_supported") else True:
            dtype = torch.bfloat16
        else:
            enabled = False
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def build_optimizer(model: MambaLM, train_cfg: TrainConfig) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.dim() < 2 or name.endswith("bias") or "norm" in name or name.endswith("A_log") or name.endswith(".D"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
    )


def _lr_at(step: int, train_cfg: TrainConfig) -> float:
    if train_cfg.warmup_steps <= 0:
        return train_cfg.lr
    if step < train_cfg.warmup_steps:
        return train_cfg.lr * float(step + 1) / float(train_cfg.warmup_steps)
    return train_cfg.lr


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


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
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
        total_loss += float(loss.item()) * y.numel()
        total_tokens += y.numel()
        n_batches += 1
    model.train()
    mean_loss = total_loss / max(1, total_tokens)
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
    optimizer = build_optimizer(model, train_cfg)
    use_scaler = train_cfg.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    grad_norms: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    tokens_seen = 0
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_iter = _cycle(train_loader)
    model.train()
    for step in range(train_cfg.max_steps):
        lr = _lr_at(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_cfg.precision):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip))
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(float(loss.item()))
        grad_norms.append(grad_norm)
        tokens_seen += y.numel()

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
                if diag:
                    means = ", ".join(f"L{d['layer']}={d['dynamic_A_mean']:.4f}" for d in diag)
                    msg += f"  A_scale_mean[{means}]"
                log_fn(msg)

        if (step + 1) % train_cfg.eval_interval == 0 or step + 1 == train_cfg.max_steps:
            val = evaluate(
                model, val_loader, device, train_cfg.precision, train_cfg.eval_batches
            )
            val_losses.append(val["val_loss"])
            if log_fn:
                log_fn(f"eval step {step}  val_loss={val['val_loss']:.4f}  ppl={val['val_ppl']:.2f}")
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler if use_scaler else None,
                step=step,
                extra={"tokenizer": tokenizer.to_dict(), "train_config": train_cfg.to_dict()},
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
        "mean_grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
    }
