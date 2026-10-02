"""Worker fault classification and the device-switching hysteresis."""
import asyncio

import pytest

from kestrel_audio import config, gpu
from kestrel_audio.manager import Manager
from kestrel_audio.store import Store
from kestrel_audio.worker import looks_like_gpu_fault


class OrtLike(Exception):
    pass


OrtLike.__module__ = "onnxruntime.capi.onnxruntime_pybind11_state"


def test_allocator_errors_that_never_name_the_gpu_still_count_as_gpu_faults():
    msg = "Non-zero status code returned while running Neg node. bfc_arena.cc:358 Available memory of 65328128 is smaller than requested"
    assert looks_like_gpu_fault(OrtLike(msg), "cuda")
    assert looks_like_gpu_fault(RuntimeError("CUDAExecutionProvider did not load"), "cuda")
    assert looks_like_gpu_fault(RuntimeError("CUDA out of memory"), "cpu")           # the text alone is enough
    assert looks_like_gpu_fault(OrtLike("shape mismatch"), "cuda")                   # anything onnxruntime raises on the GPU
    assert not looks_like_gpu_fault(OrtLike("shape mismatch"), "cpu")                # but not on the CPU path
    assert not looks_like_gpu_fault(ValueError("bad clip"), "cuda")


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_AUDIO_DATA", str(tmp_path))
    cfg = config.load()
    store = Store(cfg.data_dir, cfg.inbox_dir, cfg.previews_dir, cfg.db_path)
    state = {"free": 9000}
    m = Manager(cfg, store, probe=lambda: gpu.GpuInfo(True, "Tesla P40", state["free"], 24576))
    yield m, state
    store.close()


def test_a_running_cpu_worker_only_moves_to_the_gpu_when_there_is_comfortable_room(mgr):
    m, state = mgr
    state["free"] = 2600                                   # just over the 2500 MiB threshold
    assert m.decide_device()[0] == "cuda"                  # nothing running: start on the GPU
    m._proc, m._device = object(), "cpu"                   # a CPU worker is alive
    assert m.decide_device()[0] == "cpu"                   # 2600 < 2500 + 512: stay, no restart
    state["free"] = 3100
    assert m.decide_device()[0] == "cuda"                  # comfortable room: switch


def test_a_gpu_worker_is_judged_as_if_its_own_memory_were_free(mgr):
    m, state = mgr
    m._proc, m._device, m.vram_mib = object(), "cuda", 1100
    state["free"] = 1500                                   # others left only 1500 MiB, plus the 1100 MiB this worker holds
    assert m.decide_device()[0] == "cuda"
    state["free"] = 800                                    # 800 + 1100 = 1900 < 2500: not enough room even for us
    assert m.decide_device()[0] == "cpu"


def test_an_open_breaker_forces_the_cpu(mgr):
    m, state = mgr
    m._open_breaker("test fault")
    dev, why = m.decide_device()
    assert dev == "cpu" and "failed" in why
    assert m.system_errors[-1]["error"].startswith("GPU path paused")


# ---------------------------------------------------------------------------------------------- worker lifecycle (fake worker)
import dataclasses
import sys
import time
from pathlib import Path

FAKE = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]


def run_queue(tmp_path, monkeypatch, species_list, *, free=9000, settle=6.0, **overrides):
    """Start a manager with the fake worker, submit one job per species, wait until all are finished, and return
    (store, manager-state-after, detection ids)."""
    monkeypatch.setenv("KESTREL_AUDIO_DATA", str(tmp_path))
    cfg = dataclasses.replace(config.load(), idle_unload_s=1, job_timeout_gpu_s=3, job_timeout_cpu_s=3, **overrides)
    store = Store(cfg.data_dir, cfg.inbox_dir, cfg.previews_dir, cfg.db_path)
    seen = {}

    async def go():
        m = Manager(cfg, store, probe=lambda: gpu.GpuInfo(True, "Tesla P40", free, 24576), worker_cmd=FAKE)
        await m.start()
        for i, sp in enumerate(species_list):
            store.submit(100 + i, b"clip", species=sp, scientific="x", camera="c", now=1000.0 + i)
        m.kick()
        deadline = time.time() + settle
        while time.time() < deadline and any(store.get(100 + i).state in ("pending", "running") for i in range(len(species_list))):
            await asyncio.sleep(0.1)
        seen["worker_during"] = m.worker_state()
        seen["breaker"] = m.breaker_open()
        seen["errors"] = list(m.system_errors)
        await asyncio.sleep(1.8)                                        # past the 1 s idle limit
        seen["worker_after_idle"] = m.worker_state()
        await m.stop()

    asyncio.run(go())
    return store, seen


def test_jobs_run_on_the_gpu_and_the_worker_leaves_after_the_idle_time(tmp_path, monkeypatch):
    store, seen = run_queue(tmp_path, monkeypatch, ["Blue Jay", "Barred Owl"])
    jobs = [store.get(100), store.get(101)]
    assert [j.state for j in jobs] == ["ready", "ready"] and {j.device for j in jobs} == {"cuda"}
    assert seen["worker_during"]["running"] and seen["worker_during"]["device"] == "cuda" and seen["worker_during"]["vramMiB"] == 600
    assert seen["worker_after_idle"] == {"running": False, "device": None, "idleS": None, "vramMiB": None, "vramPeakMiB": None}
    store.close()


def test_a_busy_gpu_means_the_cpu_worker(tmp_path, monkeypatch):
    store, seen = run_queue(tmp_path, monkeypatch, ["Blue Jay"], free=1200)
    assert store.get(100).state == "ready" and store.get(100).device == "cpu"
    assert seen["worker_during"]["device"] == "cpu" and seen["worker_during"]["vramMiB"] is None
    store.close()


def test_a_gpu_fault_retries_the_clip_on_the_cpu_and_pauses_the_gpu(tmp_path, monkeypatch):
    store, seen = run_queue(tmp_path, monkeypatch, ["Gpu Fault"])
    job = store.get(100)
    assert job.state == "ready" and job.device == "cpu" and job.attempts == 2
    assert seen["breaker"] and any("GPU path paused" in e["error"] for e in seen["errors"])
    store.close()


def test_a_clip_that_cannot_become_a_preview_fails_without_retry_or_breaker(tmp_path, monkeypatch):
    store, seen = run_queue(tmp_path, monkeypatch, ["Bad Clip", "Blue Jay"])
    assert store.get(100).state == "failed" and store.get(101).state == "ready"
    assert store.get(100).error == "the clip is silent" and store.get(100).attempts == 1
    assert not seen["breaker"]
    store.close()


def test_a_worker_that_dies_mid_job_is_replaced_and_the_clip_retried_on_the_cpu(tmp_path, monkeypatch):
    store, seen = run_queue(tmp_path, monkeypatch, ["Crash"])
    job = store.get(100)
    assert job.state == "failed" and job.attempts == 2                                # the CPU worker crashes too: failed, not looping
    assert seen["breaker"]
    store.close()


def test_a_worker_that_hangs_is_killed_at_the_timeout(tmp_path, monkeypatch):
    t0 = time.time()
    store, seen = run_queue(tmp_path, monkeypatch, ["Hang"], settle=12.0)
    assert store.get(100).state == "failed" and "in time" in store.get(100).error
    assert time.time() - t0 < 14
    store.close()
