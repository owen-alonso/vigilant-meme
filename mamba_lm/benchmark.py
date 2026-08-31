"""Forward/backward benchmark: baseline Mamba vs Dynamic A."""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch

from mamba_lm.config import MambaConfig
from mamba_lm.model import MambaLM
from mamba_lm.reporting import format_parameter_report, parameter_report
from mamba_lm.training_utils import select_device, set_seed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_loop(
    fn,
    *,
    warmup: int,
    steps: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    times: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def _stats(xs: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.pstdev(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def benchmark_model(
    config: MambaConfig,
    *,
    batch_size: int = 4,
    seq_len: int = 256,
    warmup: int = 5,
    steps: int = 20,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    device = device or select_device()
    set_seed(seed)
    model = MambaLM(config).to(device=device, dtype=dtype)
    model.train()
    report = parameter_report(model)

    vocab = config.padded_vocab_size()
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    def forward():
        return model(input_ids)

    def backward():
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        loss.backward()
        return loss

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    fwd_times = _timed_loop(forward, warmup=warmup, steps=steps, device=device)
    bwd_times = _timed_loop(backward, warmup=warmup, steps=steps, device=device)

    tokens = batch_size * seq_len
    fwd = _stats(fwd_times)
    bwd = _stats(bwd_times)
    peak_mem = (
        float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    )
    return {
        "dynamic_weights": config.dynamic_weights,
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_batch": tokens,
        "parameter_report": report,
        "forward_sec": fwd,
        "backward_sec": bwd,
        "forward_tokens_per_sec": tokens / fwd["mean"],
        "backward_tokens_per_sec": tokens / bwd["mean"],
        "peak_memory_bytes": peak_mem,
    }


def format_benchmark(result: dict[str, Any]) -> str:
    mode = "Dynamic A" if result["dynamic_weights"] else "Baseline"
    lines = [
        f"=== {mode} ===",
        f"device={result['device']} dtype={result['dtype']} "
        f"B={result['batch_size']} L={result['seq_len']}",
        format_parameter_report(result["parameter_report"]),
        (
            f"forward:  {result['forward_sec']['mean']*1e3:.2f} ms "
            f"(std {result['forward_sec']['std']*1e3:.2f})  "
            f"{result['forward_tokens_per_sec']:.0f} tok/s"
        ),
        (
            f"backward: {result['backward_sec']['mean']*1e3:.2f} ms "
            f"(std {result['backward_sec']['std']*1e3:.2f})  "
            f"{result['backward_tokens_per_sec']:.0f} tok/s"
        ),
        f"peak GPU memory: {result['peak_memory_bytes'] / (1024**2):.1f} MiB",
    ]
    return "\n".join(lines)


def run_comparison(
    *,
    d_model: int = 128,
    n_layer: int = 4,
    d_state: int = 16,
    expand: int = 2,
    vocab_size: int = 64,
    batch_size: int = 4,
    seq_len: int = 256,
    warmup: int = 5,
    steps: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shared = dict(
        d_model=d_model,
        n_layer=n_layer,
        d_state=d_state,
        vocab_size=vocab_size,
        expand=expand,
    )
    baseline = benchmark_model(
        MambaConfig(**shared, dynamic_weights=False),
        batch_size=batch_size,
        seq_len=seq_len,
        warmup=warmup,
        steps=steps,
    )
    dynamic = benchmark_model(
        MambaConfig(**shared, dynamic_weights=True, dynamic_A=True),
        batch_size=batch_size,
        seq_len=seq_len,
        warmup=warmup,
        steps=steps,
    )
    return baseline, dynamic
