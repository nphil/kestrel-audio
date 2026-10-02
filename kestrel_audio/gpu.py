"""GPU probing (NVML, falling back to nvidia-smi) and the rule that decides where a job runs.

The GPU is shared with llama-swap and others: a model is only loaded on it when enough memory is free, and the
framework is capped so this process stays small. Otherwise the job runs on the CPU.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from typing import Callable
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


def compute_processes() -> dict[int, int]:
    """{pid: MiB} of every process using the first GPU, with the PIDs the driver reports (the host's, not the container's).
    Empty when NVML and nvidia-smi are both unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return {int(p.pid): int((p.usedGpuMemory or 0) // 2**20) for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)}
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                          timeout=5, text=True)
            return {int(a): int(b) for a, b in (ln.split(",") for ln in out.strip().splitlines() if "," in ln)}
        except Exception:
            return {}
    return {}


class Footprint:
    """What THIS process holds on the GPU. A container cannot see its own host PID, and on a shared GPU other programs
    start and stop all the time, so the PID is learnt once, at the moment our own CUDA context appears: the processes that
    already used the GPU are remembered when this is created, and `claim()` (called right after the first model is loaded)
    says "that one is us" only if exactly ONE new process is there. Zero or several (another program started in the same few
    seconds) is reported as unknown (None) rather than guessed: counting every newcomer as ours would blame other
    programs' memory on this service.
    NVML stays initialised for the life of the object, which keeps a sample cheap enough to take ten times a second."""

    def __init__(self, table: "Callable[[], dict[int, int]] | None" = None) -> None:
        self._nvml = None
        self._handle = None
        self._table_fn = table
        if table is None:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml = pynvml
            except Exception:
                self._nvml = None
        self._before = set(self._table())
        self._mine: int | None = None
        self.claimed = False        # claim() ran (successfully or not)

    def _table(self) -> dict[int, int]:
        if self._table_fn is not None:
            return self._table_fn()
        if self._nvml is not None:
            try:
                procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
                return {int(p.pid): int((p.usedGpuMemory or 0) // 2**20) for p in procs}
            except Exception:
                pass
        return compute_processes()

    def claim(self) -> int | None:
        """Pin our PID: call once, right after the first CUDA session has been created. Returns it, or None if unknown."""
        if self.claimed:
            return self._mine
        new = [pid for pid in self._table() if pid not in self._before]
        self._mine = new[0] if len(new) == 1 else None
        self.claimed = True
        return self._mine

    def mib(self) -> int | None:
        """GPU memory this process holds now, or None when it cannot be told apart from other programs' (or is not loaded)."""
        if self._mine is None:
            return None
        return self._table().get(self._mine)


class PeakSampler:
    """Context manager: samples `footprint.mib()` in a background thread while a job runs and keeps the highest value."""

    def __init__(self, footprint: "Footprint | None", interval: float = 0.1):
        self._fp, self._interval = footprint, interval
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            v = self._fp.mib() if self._fp is not None else None
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakSampler":
        if self._fp is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            v = self._fp.mib() if self._fp is not None else None
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v


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
