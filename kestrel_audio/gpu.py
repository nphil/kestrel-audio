"""GPU probing (NVML, falling back to nvidia-smi) and the rule that decides where a job runs.

The GPU is shared with llama-swap and others: a model is only loaded on it when enough memory is free, and the
framework is capped so this process stays small. Otherwise the job runs on the CPU.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    available: bool
    name: str | None = None
    free_mib: int | None = None
    total_mib: int | None = None


NO_GPU = GpuInfo(False)


def _query_nvml() -> GpuInfo | None:
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else str(name)
        return GpuInfo(True, name, int(mem.free // 2**20), int(mem.total // 2**20))
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _query_smi() -> GpuInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"], timeout=5, text=True)
        name, free, total = [p.strip() for p in out.splitlines()[0].split(",")]
        return GpuInfo(True, name, int(free), int(total))
    except Exception:
        return None


def query() -> GpuInfo:
    """Current state of the first visible GPU (NO_GPU when there is none)."""
    return _query_nvml() or _query_smi() or NO_GPU


def used_by_pid_mib(pid: int) -> int | None:
    """GPU memory held by one process, or None when NVML cannot attribute it (container PID namespaces can hide it)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid == pid and p.usedGpuMemory is not None:
                    return int(p.usedGpuMemory // 2**20)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None
    return None


def choose_device(info: GpuInfo, *, min_free_mib: int, breaker_open: bool = False, forced: str = "auto") -> tuple[str, str]:
    """Where should the next worker run? Returns ("cuda" | "cpu", plain-language reason)."""
    if forced == "cpu":
        return "cpu", "CPU forced by configuration"
    if not info.available or info.free_mib is None:
        return "cpu", "no GPU visible"
    if breaker_open:
        return "cpu", "GPU recently failed; using the CPU for a while"
    if forced == "cuda":
        return "cuda", "GPU forced by configuration"
    if info.free_mib < min_free_mib:
        return "cpu", f"GPU busy: {info.free_mib} MiB free, {min_free_mib} MiB needed"
    return "cuda", f"{info.free_mib} MiB free on the GPU"


def env_forced_device() -> str:
    v = os.environ.get("KESTREL_AUDIO_DEVICE", "auto").strip().lower()
    return v if v in ("auto", "cpu", "cuda") else "auto"
