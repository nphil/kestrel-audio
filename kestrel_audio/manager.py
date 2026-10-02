"""Queue manager: one worker process at a time, started when there is work and gone after 120 s of quiet.

Where a worker runs is decided when it is started: on the GPU only if enough memory is free (the GPU is shared with
llama-swap), otherwise on the CPU. A GPU fault (or a hang) closes a "breaker" for ten minutes during which workers use
the CPU, and the job that hit it is retried there. Killing the worker process is what frees the GPU memory.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable

from . import __version__, gpu
from .config import Config
from .store import Job, Store, iso

log = logging.getLogger("kestrel_audio.manager")

BREAKER_SECONDS = 600
HELLO_TIMEOUT_S = 60
STOP_GRACE_S = 15
MAX_ATTEMPTS = 2


class WorkerGone(RuntimeError):
    """The worker exited or stopped talking in the middle of a job."""


class Manager:
    def __init__(self, cfg: Config, store: Store, *, probe: Callable[[], gpu.GpuInfo] = gpu.query):
        self.cfg, self.store, self._probe = cfg, store, probe
        self._proc: asyncio.subprocess.Process | None = None
        self._device: str | None = None
        self._last_activity = time.monotonic()
        self._wake = asyncio.Event()
        self._breaker_until = 0.0
        self._task: asyncio.Task | None = None
        self._janitor_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._running_id: int | None = None
        self.vram_mib: int | None = None
        self.started = time.time()
        self.system_errors: deque[dict[str, Any]] = deque(maxlen=20)
        self._gpu_cache: tuple[float, gpu.GpuInfo] = (0.0, gpu.NO_GPU)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        n = self.store.recover()
        if n:
            log.info("recovered %d interrupted job(s)", n)
        self._task = asyncio.create_task(self._loop(), name="queue")
        self._janitor_task = asyncio.create_task(self._janitor(), name="janitor")
        self._wake.set()

    async def stop(self) -> None:
        for t in (self._task, self._janitor_task):
            if t:
                t.cancel()
        for t in (self._task, self._janitor_task):
            if t:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        await self._stop_worker("shutting down")

    def kick(self) -> None:
        self._wake.set()

    # ------------------------------------------------------------------ gpu / device choice
    def gpu_info(self) -> gpu.GpuInfo:
        now = time.monotonic()
        if now - self._gpu_cache[0] > 2.0:
            self._gpu_cache = (now, self._probe())
        return self._gpu_cache[1]

    def breaker_open(self) -> bool:
        return time.monotonic() < self._breaker_until

    def _open_breaker(self, why: str) -> None:
        self._breaker_until = time.monotonic() + BREAKER_SECONDS
        self._note_error(None, f"GPU path paused for {BREAKER_SECONDS // 60} min: {why}")

    def decide_device(self) -> tuple[str, str]:
        info = self._probe()
        self._gpu_cache = (time.monotonic(), info)
        free = info.free_mib
        if self._device == "cuda" and self._proc is not None and free is not None:
            # our own worker already holds its memory: judge the room as if it were not there
            free += self.vram_mib or 0
            info = gpu.GpuInfo(info.available, info.name, free, info.total_mib)
        return gpu.choose_device(info, min_free_mib=self.cfg.min_free_vram_mib, breaker_open=self.breaker_open(), forced=self.cfg.device)

    def _note_error(self, det: int | None, text: str) -> None:
        log.warning("%s%s", f"job {det}: " if det is not None else "", text)
        self.system_errors.append({"at": iso(time.time()), "detectionId": det, "error": text[:300]})

    # ------------------------------------------------------------------ main loop
    async def _loop(self) -> None:
        while True:
            try:
                job = self.store.claim_next()
                if job is None:
                    self._wake.clear()
                    remaining = self._idle_remaining()
                    try:
                        if remaining is None:
                            await self._wake.wait()
                        else:
                            await asyncio.wait_for(self._wake.wait(), remaining)
                    except asyncio.TimeoutError:
                        await self._stop_worker("idle")
                    continue
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception:                              # never let the loop die
                log.exception("queue loop error")
                await asyncio.sleep(2)

    def _idle_remaining(self) -> float | None:
        if self._proc is None:
            return None
        return max(0.0, self.cfg.idle_unload_s - (time.monotonic() - self._last_activity))

    async def _janitor(self) -> None:
        await asyncio.sleep(30)
        while True:
            try:
                gone = self.store.purge(max_age_days=self.cfg.retention_days, cap_bytes=self.cfg.cap_bytes)
                if gone:
                    log.info("retention removed %d preview(s)", len(gone))
            except Exception:
                log.exception("retention failed")
            await asyncio.sleep(3600)

    # ------------------------------------------------------------------ one job
    async def _run_job(self, job: Job) -> None:
        device, why = self.decide_device()
        if self._proc is not None and self._device != device:
            await self._stop_worker(f"switching to {device} ({why})")
        t0 = time.monotonic()
        timeout = self.cfg.job_timeout_gpu_s if device == "cuda" else self.cfg.job_timeout_cpu_s
        self._running_id = job.detection_id
        try:
            await self._ensure_worker(device, why)
            reply = await self._exchange(job, timeout)
        except (WorkerGone, asyncio.TimeoutError) as exc:
            text = "the worker did not answer in time" if isinstance(exc, asyncio.TimeoutError) else str(exc)
            await self._stop_worker("worker fault", kill=True)
            self._after_fault(job, device, text, gpu_fault=(device == "cuda"), took=time.monotonic() - t0)
            return
        finally:
            self._running_id = None
            self._last_activity = time.monotonic()
        took = time.monotonic() - t0
        ev = reply.get("event")
        if ev == "done":
            info = reply["info"]
            self.vram_mib = reply.get("vramMiB")
            self.store.finish(job.detection_id, info, nbytes=int(info.get("bytes", 0)), cleaned=bool(info.get("cleaned")),
                              took_s=took, device=device)
        elif ev == "failed":
            self.store.fail(job.detection_id, str(reply.get("error", "failed")), took_s=took)
        else:
            gpu_fault = bool(reply.get("gpu"))
            if gpu_fault:
                await self._stop_worker("GPU fault", kill=True)
            self._after_fault(job, device, str(reply.get("error", "unknown error")), gpu_fault=gpu_fault, took=took)

    def _after_fault(self, job: Job, device: str, error: str, *, gpu_fault: bool, took: float) -> None:
        self._note_error(job.detection_id, error)
        if gpu_fault and device == "cuda":
            self._open_breaker(error[:120])
        retry = gpu_fault and device == "cuda" and job.attempts < MAX_ATTEMPTS
        if retry:
            self.store.requeue(job.detection_id)       # runs again, on the CPU this time
        else:
            self.store.fail(job.detection_id, error, took_s=took)

    # ------------------------------------------------------------------ worker process
    async def _ensure_worker(self, device: str, why: str) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        self._proc = None
        env = dict(os.environ)
        env["KESTREL_AUDIO_WORKER_DEVICE"] = device
        threads = str(self.cfg.cpu_threads if device == "cpu" else 2)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = threads
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        nice = self.cfg.cpu_nice if device == "cpu" else max(0, self.cfg.cpu_nice // 2)

        def _lower_priority() -> None:
            try:
                os.nice(nice)
            except OSError:
                pass

        log.info("starting a %s worker (%s)", device, why)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "kestrel_audio.worker", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env, preexec_fn=_lower_priority)
        self._proc, self._device = proc, device
        self._stderr_task = asyncio.create_task(self._pump_stderr(proc))
        hello = await self._read_event(proc, HELLO_TIMEOUT_S)
        if hello.get("event") != "hello":
            raise WorkerGone(str(hello.get("error", "the worker did not start")))

    async def _pump_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            log.info("worker: %s", line.decode(errors="replace").rstrip())

    async def _read_event(self, proc: asyncio.subprocess.Process, timeout: float) -> dict[str, Any]:
        assert proc.stdout is not None
        line = await asyncio.wait_for(proc.stdout.readline(), timeout)
        if not line:
            raise WorkerGone(f"the worker exited (code {proc.returncode})")
        return json.loads(line)

    async def _exchange(self, job: Job, timeout: float) -> dict[str, Any]:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        msg = {"cmd": "job", "id": job.detection_id, "clip": str(self.store.clip_path(job.detection_id)),
               "out": str(self.store.preview_path(job.detection_id)), "scientific": job.scientific, "species": job.species}
        try:
            proc.stdin.write((json.dumps(msg) + "\n").encode())
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise WorkerGone("the worker is not running") from exc
        deadline = time.monotonic() + timeout
        while True:
            reply = await self._read_event(proc, max(1.0, deadline - time.monotonic()))
            if reply.get("id") in (job.detection_id, None) or reply.get("event") in ("done", "failed", "error"):
                return reply

    async def _stop_worker(self, why: str, *, kill: bool = False) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        log.info("stopping the %s worker: %s", self._device, why)
        self._device = None
        self.vram_mib = None
        try:
            if proc.returncode is None and not kill and proc.stdin is not None:
                try:
                    proc.stdin.write(b'{"cmd":"quit"}\n')
                    await proc.stdin.drain()
                    proc.stdin.close()
                    await asyncio.wait_for(proc.wait(), STOP_GRACE_S)
                except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
                    pass
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        finally:
            if self._stderr_task:
                self._stderr_task.cancel()
                self._stderr_task = None

    # ------------------------------------------------------------------ status
    def worker_state(self) -> dict[str, Any]:
        running = self._proc is not None and self._proc.returncode is None
        idle = None
        if running and self._running_id is None:
            idle = int(time.monotonic() - self._last_activity)
        return {"running": running, "device": self._device if running else None, "idleS": idle,
                "vramMiB": self.vram_mib if running and self._device == "cuda" else None}

    def stats(self) -> dict[str, Any]:
        counts = self.store.counts()
        waiting = counts["pending"]
        running = 1 if self._running_id is not None else 0
        info = self.gpu_info()
        w = self.worker_state()
        since = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        errors = self.store.last_errors(10)
        errors = sorted(errors + list(self.system_errors), key=lambda e: e["at"] or "", reverse=True)[:10]
        return {
            "service": "kestrel-audio", "version": __version__, "uptimeS": int(time.time() - self.started),
            "queue": {"waiting": waiting, "running": running, "depth": waiting + running},
            "worker": w,
            "gpu": {"available": info.available, "freeMiB": info.free_mib, "totalMiB": info.total_mib, "name": info.name},
            "counts": {"ready": counts["ready"], "failed": counts["failed"], "cleaned": counts["cleaned"], "pending": waiting + running},
            "today": self.store.today(since),
            "storage": {**self.store.storage(), "retentionDays": self.cfg.retention_days, "capBytes": self.cfg.cap_bytes},
            "timing": self.store.timing(),
            "mode": ("idle" if not w["running"] else ("gpu" if w["device"] == "cuda" else "cpu")),
            "lastErrors": errors,
        }
