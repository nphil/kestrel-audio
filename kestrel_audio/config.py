"""Service configuration, read once from the environment (the Unraid template sets these)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Config", "load"]

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    version: str
    data_dir: Path
    models_dir: Path
    assets_dir: Path
    host: str
    port: int
    # where jobs run
    device: str                 # auto | cpu | cuda
    min_free_vram_mib: int      # free GPU memory needed before a worker may use the GPU
    gpu_cap_mib: int            # the worker process must stay under this on the GPU
    cpu_threads: int
    cpu_nice: int
    idle_unload_s: int          # worker (and its GPU memory) goes away after this much idle time
    job_timeout_gpu_s: int
    job_timeout_cpu_s: int
    cpu_cleanup: bool           # run the clean-up search on the CPU path too (slow); False = trim only on CPU
    # storage
    retention_days: int
    cap_bytes: int
    # limits
    max_body_bytes: int
    max_clip_s: float

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def previews_dir(self) -> Path:
        return self.data_dir / "previews"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "kestrel-audio.db"

    @property
    def key_path(self) -> Path:
        return self.data_dir / "key"


def load() -> Config:
    data = Path(_env("KESTREL_AUDIO_DATA", "/data"))
    return Config(
        version=_env("KESTREL_AUDIO_VERSION", "0.0.0-dev"),
        data_dir=data,
        models_dir=Path(_env("KESTREL_AUDIO_MODELS", "/models")),
        assets_dir=Path(_env("KESTREL_AUDIO_ASSETS", str(REPO_DIR / "assets"))),
        host=_env("KESTREL_AUDIO_HOST", "0.0.0.0"),
        port=_int("KESTREL_AUDIO_PORT", 8787),
        device=_env("KESTREL_AUDIO_DEVICE", "auto").lower(),
        min_free_vram_mib=_int("KESTREL_AUDIO_MIN_FREE_VRAM_MIB", 2500),
        gpu_cap_mib=_int("KESTREL_AUDIO_GPU_CAP_MIB", 1500),
        cpu_threads=max(1, min(_int("KESTREL_AUDIO_CPU_THREADS", 4), 4)),
        cpu_nice=_int("KESTREL_AUDIO_CPU_NICE", 10),
        idle_unload_s=_int("KESTREL_AUDIO_IDLE_UNLOAD_S", 120),
        job_timeout_gpu_s=_int("KESTREL_AUDIO_JOB_TIMEOUT_GPU_S", 300),
        job_timeout_cpu_s=_int("KESTREL_AUDIO_JOB_TIMEOUT_CPU_S", 900),
        cpu_cleanup=_env("KESTREL_AUDIO_CPU_CLEANUP", "1") not in ("0", "false", "no"),
        retention_days=_int("KESTREL_AUDIO_RETENTION_DAYS", 30),
        cap_bytes=_int("KESTREL_AUDIO_CAP_MB", 2048) * 1024 * 1024,
        max_body_bytes=_int("KESTREL_AUDIO_MAX_BODY_MB", 20) * 1024 * 1024,
        max_clip_s=float(_env("KESTREL_AUDIO_MAX_CLIP_S", "60")),
    )
