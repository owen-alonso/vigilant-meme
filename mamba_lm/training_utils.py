"""Shared training utilities for LM and forecast loops."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# Windows EXECUTION_STATE flags for SetThreadExecutionState.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002
_KEEP_AWAKE_FLAGS = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED


def _set_windows_execution_state(flags: int) -> bool:
    """Return True if the call succeeded. No-op on non-Windows."""
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState.argtypes = (ctypes.c_uint,)
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint
    return int(kernel32.SetThreadExecutionState(flags)) != 0


def _is_wsl() -> bool:
    """True when this Linux process is running under WSL (host is Windows)."""
    if sys.platform == "win32":
        return False
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    lowered = version.lower()
    return "microsoft" in lowered or "wsl" in lowered


def _find_windows_powershell() -> str | None:
    for name in ("powershell.exe", "pwsh.exe"):
        found = shutil.which(name)
        if found:
            return found
    for path in (
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/pwsh.exe",
    ):
        if os.path.isfile(path):
            return path
    return None


def _wsl_keep_awake_script(interval_sec: float) -> str:
    seconds = max(1, int(interval_sec))
    # SetThreadExecutionState is per-thread: this process must stay alive.
    return (
        "Add-Type 'using System.Runtime.InteropServices; "
        "public class E { [DllImport(\"kernel32.dll\")] "
        "public static extern uint SetThreadExecutionState(uint f); }'; "
        f"for(;;){{ [void][E]::SetThreadExecutionState({_KEEP_AWAKE_FLAGS}); "
        f"Start-Sleep -Seconds {seconds} }}"
    )


def _start_wsl_keep_awake(interval_sec: float) -> subprocess.Popen[bytes] | None:
    """Hold Windows sleep/display inhibit from inside WSL via powershell.exe."""
    exe = _find_windows_powershell()
    if exe is None:
        return None
    try:
        return subprocess.Popen(
            [
                exe,
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                _wsl_keep_awake_script(interval_sec),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return None


def _stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)


@contextmanager
def keep_awake(
    log_fn: Any | None = None,
    *,
    interval_sec: float = 30.0,
) -> Iterator[None]:
    """Keep the machine and monitor awake while a training loop is running.

    On native Windows this calls ``SetThreadExecutionState``. WSL cannot
    call ``kernel32`` from Linux Python, so it keeps a ``powershell.exe``
    helper alive on the Windows host with the same flags. Other platforms
    are a no-op. Always clears the inhibit on exit, including Ctrl+C.
    """
    stop = threading.Event()
    thread: threading.Thread | None = None
    wsl_proc: subprocess.Popen[bytes] | None = None
    armed = False
    backend = ""

    if sys.platform == "win32":
        backend = "win32"
        armed = _set_windows_execution_state(_KEEP_AWAKE_FLAGS)
        if log_fn:
            if armed:
                log_fn(
                    "keep_awake: system sleep and display inhibited until training ends"
                )
            else:
                log_fn(
                    "keep_awake: SetThreadExecutionState failed; "
                    "the machine may still sleep or blank the monitor"
                )

        def _pulse() -> None:
            while not stop.wait(max(0.1, float(interval_sec))):
                _set_windows_execution_state(_KEEP_AWAKE_FLAGS)

        thread = threading.Thread(target=_pulse, name="keep_awake", daemon=True)
        thread.start()
    elif _is_wsl():
        backend = "wsl"
        wsl_proc = _start_wsl_keep_awake(interval_sec)
        armed = wsl_proc is not None and wsl_proc.poll() is None
        if log_fn:
            if armed:
                log_fn(
                    "keep_awake: WSL bridged to Windows; "
                    "sleep and display inhibited until training ends"
                )
            else:
                log_fn(
                    "keep_awake: WSL detected but powershell.exe did not start; "
                    "the Windows host may still sleep or blank the monitor"
                )

    try:
        yield
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1.0)
        if backend == "win32":
            _set_windows_execution_state(_ES_CONTINUOUS)
        elif backend == "wsl":
            _stop_process(wsl_proc)
        if armed and log_fn:
            log_fn("keep_awake: sleep and display restored")


def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str, log_fn: Any | None = None):
    """AMP context. CPU fp16 is not reliable; fall back to bf16 or fp32 and say so."""
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "fp16" and device.type == "cpu":
        bf16_ok = bool(getattr(torch.cpu, "is_bf16_supported", lambda: False)())
        if bf16_ok:
            if log_fn:
                log_fn("precision=fp16 is not available on CPU; using bf16 autocast")
            return torch.autocast(
                device_type="cpu", dtype=torch.bfloat16, enabled=True
            )
        if log_fn:
            log_fn("precision=fp16 is not available on CPU; using fp32")
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """AdamW with no-decay groups for norms, biases, and SSM skip parameters."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if (
            param.dim() < 2
            or name.endswith("bias")
            or "norm" in name
            or name.endswith("A_log")
            or name.endswith(".D")
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # Fused AdamW runs one multi-tensor CUDA kernel per step instead of
    # per-parameter kernels; profiling showed the unfused step at ~5% of a
    # training step. Requires every parameter on CUDA.
    if decay + no_decay and all(p.is_cuda for p in decay + no_decay):
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=True)
        except RuntimeError:
            pass  # fused unsupported on this build/device: use the default
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def dataloader_kwargs(device: torch.device, num_workers: int) -> dict[str, Any]:
    """DataLoader performance settings shared by the LM and forecast loops.

    Pinned host memory makes ``.to(device, non_blocking=True)`` a true async
    copy; persistent workers avoid Windows' expensive per-epoch process
    respawn; prefetch keeps batches ahead of the GPU on larger datasets.
    """
    kwargs: dict[str, Any] = {"num_workers": num_workers}
    if device.type == "cuda":
        kwargs["pin_memory"] = True
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def require_nonempty_loader(loader: DataLoader, name: str) -> None:
    if len(loader) == 0:
        raise RuntimeError(
            f"{name} DataLoader is empty (dataset size={len(loader.dataset)}, "
            f"batch_size={loader.batch_size}, drop_last={loader.drop_last}). "
            "Shrink batch_size or collect more data."
        )


def cycle_loader(loader: DataLoader) -> Iterator:
    require_nonempty_loader(loader, "train")
    while True:
        for batch in loader:
            yield batch


def grads_finite(model: torch.nn.Module, *, unique: bool = True) -> bool:
    """Return False if any gradient is non-finite.

    All per-gradient checks stay on device and are combined into one flag so
    the CPU synchronizes with the GPU once, not once per parameter (the
    per-parameter version cost ~25% of a forecast training step).
    """
    if unique:
        from mamba_lm.reporting import unique_parameters

        params = unique_parameters(model)
    else:
        params = (p for p in model.parameters() if p.requires_grad)
    grads = [p.grad.reshape(-1) for p in params if p.grad is not None]
    if not grads:
        return True
    # One cat + one isfinite + one reduction instead of ~3 kernels per
    # parameter: the copy is ~a megabyte, the launch overhead it removes
    # was milliseconds.
    flat = grads[0] if len(grads) == 1 else torch.cat(grads)
    return bool(torch.isfinite(flat).all().item())


def lr_linear_warmup(step: int, *, lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return lr
    if step < warmup_steps:
        return lr * float(step + 1) / float(warmup_steps)
    return lr


def lr_warmup_cosine(
    step: int,
    *,
    total_steps: int,
    lr: float,
    warmup_frac: float,
    min_lr_frac: float = 0.1,
) -> float:
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_frac * lr + (1.0 - min_lr_frac) * lr * cosine
