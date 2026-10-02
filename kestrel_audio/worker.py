"""Worker process: holds the models, handles one job at a time, exits on request.

Protocol (one JSON object per line):
  parent -> worker   {"cmd":"job","id":N,"clip":path,"out":path,"scientific":str|null,"species":str}   |   {"cmd":"quit"}
  worker -> parent   {"event":"hello","device":..,"pid":..}
                     {"event":"done","id":N,"info":{..},"vramMiB":n|null}
                     {"event":"failed","id":N,"error":str}                 the clip cannot become a preview (not a fault)
                     {"event":"error","id":N,"error":str,"gpu":bool}       something broke; gpu=true means a CUDA/memory fault

The process exiting is what frees the GPU memory: models are never unloaded in place.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from . import config
from .codec import AudioError
from .pipeline import PreviewError

_GPU_ERROR_HINTS = ("cuda", "cudnn", "cublas", "out of memory", "oom", "no kernel image", "device-side", "gpu_mem_limit")


def looks_like_gpu_fault(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(h in text for h in _GPU_ERROR_HINTS)


def main() -> int:
    cfg = config.load()
    device = os.environ.get("KESTREL_AUDIO_WORKER_DEVICE", "cpu")
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                                   # anything a library prints goes to the log, not the protocol

    def send(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    try:
        from .engine import Engine
        engine = Engine(cfg, device)
    except BaseException as exc:                    # noqa: BLE001
        send({"event": "error", "id": None, "error": f"worker failed to start: {exc}", "gpu": looks_like_gpu_fault(exc)})
        return 2
    send({"event": "hello", "device": device, "pid": os.getpid()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("cmd") == "quit":
            break
        if msg.get("cmd") != "job":
            continue
        det = msg["id"]
        try:
            info = engine.process(clip_path=msg["clip"], out_path=msg["out"], scientific=msg.get("scientific"), species=msg.get("species"))
            info["device"] = device
            vram = engine.vram_mib()
            info["vramMiB"] = vram                       # what the worker holds now (the resident weights)
            info["vramPeakMiB"] = engine.last_peak_mib    # the highest it held while this job ran
            send({"event": "done", "id": det, "info": info, "vramMiB": vram})
        except (PreviewError, AudioError) as exc:
            send({"event": "failed", "id": det, "error": str(exc)})
        except BaseException as exc:                # noqa: BLE001
            traceback.print_exc()
            send({"event": "error", "id": det, "error": f"{type(exc).__name__}: {exc}"[:400], "gpu": looks_like_gpu_fault(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
