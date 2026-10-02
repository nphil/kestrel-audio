"""Stand-in worker process for the manager tests: same JSON-lines protocol as kestrel_audio.worker, no models.

Behaviour is driven by the species name: "Gpu Fault" fails on the GPU only, "Bad Clip" is a failed (not faulty) job, "Crash"
exits mid-job, "Hang" never answers; anything else succeeds.
"""
import json
import os
import sys
import time

device = os.environ.get("KESTREL_AUDIO_WORKER_DEVICE", "cpu")
out = os.fdopen(os.dup(1), "w", buffering=1)


def send(o):
    out.write(json.dumps(o) + "\n")
    out.flush()


send({"event": "hello", "device": device, "pid": os.getpid()})
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("cmd") == "quit":
        break
    sp = msg.get("species")
    if sp == "Hang":
        time.sleep(60)
    if sp == "Crash":
        os._exit(3)
    if sp == "Gpu Fault" and device == "cuda":
        send({"event": "error", "id": msg["id"], "error": "CUDA out of memory", "gpu": True})
        continue
    if sp == "Bad Clip":
        send({"event": "failed", "id": msg["id"], "error": "the clip is silent"})
        continue
    open(msg["out"], "wb").write(b"m4a" * 100)
    send({"event": "done", "id": msg["id"], "vramMiB": 600 if device == "cuda" else None,
          "info": {"segment": {"start": 1.0, "end": 6.0, "source": "perch"}, "method": "trim", "variant": "B", "cleaned": False,
                   "scores": {"original": 0.5, "preview": 0.5}, "loudnessLufs": -16.0, "durationS": 5.0, "device": device,
                   "bytes": 300, "vramPeakMiB": 900 if device == "cuda" else None, "notes": []}})
