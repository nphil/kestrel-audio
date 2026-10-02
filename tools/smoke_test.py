#!/usr/bin/env python3
"""Smoke test for the built image (used by CI before anything is pushed; also handy locally).

Starts the container on the CPU path, drives the HTTP API with a synthetic clip and checks every contract the Home
Assistant integration relies on, then loads each ONNX model inside the container and runs it once.

    python3 tools/smoke_test.py kestrel-audio:smoke
"""
from __future__ import annotations

import io
import json
import math
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "kestrel-audio:smoke"
NAME = "ka-smoke"
PORT = 18787
BASE = f"http://127.0.0.1:{PORT}"


def sh(*args: str, check: bool = True) -> str:
    p = subprocess.run(args, capture_output=True, text=True)
    if check and p.returncode != 0:
        fail(f"{' '.join(args)} -> {p.returncode}: {p.stderr.strip()[-500:]}")
    return p.stdout.strip()


def fail(msg: str) -> None:
    print(f"SMOKE TEST FAILED: {msg}", file=sys.stderr)
    print(subprocess.run(["docker", "logs", "--tail", "60", NAME], capture_output=True, text=True).stdout[-3000:], file=sys.stderr)
    subprocess.run(["docker", "rm", "-f", "-v", NAME], capture_output=True)
    sys.exit(1)


def http(method: str, path: str, *, key: str | None = None, body: bytes | None = None, headers: dict | None = None):
    h = dict(headers or {})
    if key:
        h["X-Kestrel-Audio-Key"] = key
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def synthetic_clip(seconds: int = 15, sr: int = 48000) -> bytes:
    """Very quiet hiss plus three short chirps, like a real (far too quiet) camera clip."""
    rnd = random.Random(1)
    n = seconds * sr
    x = [0.002 * rnd.gauss(0, 1) for _ in range(n)]
    for start in (3.0, 6.5, 10.0):
        for i in range(int(0.3 * sr)):
            t = i / sr
            x[int(start * sr) + i] += 0.02 * math.sin(2 * math.pi * (2500 + 2000 * t) * t) * math.sin(math.pi * i / (0.3 * sr))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(int(max(-1, min(1, v)) * 32767).to_bytes(2, "little", signed=True) for v in x))
    return buf.getvalue()


def main() -> None:
    sh("docker", "rm", "-f", "-v", NAME, check=False)
    sh("docker", "run", "-d", "--name", NAME, "-p", f"{PORT}:8787", "-e", "KESTREL_AUDIO_DEVICE=cpu", "-e", "KESTREL_AUDIO_CPU_CLEANUP=0", IMAGE)

    for _ in range(60):
        try:
            if http("GET", "/healthz")[0] == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        fail("the service never answered /healthz")
    print("healthz ok")

    code, _, page = http("GET", "/")
    assert code == 200 and b"Kestrel Audio" in page, f"status page: {code}"
    code, hdr, png = http("GET", "/icon-512.png")
    assert code == 200 and png[:4] == b"\x89PNG", f"icon: {code}"
    assert http("GET", "/favicon.ico")[0] == 200
    code, _, body = http("GET", "/api/status")
    assert code == 200 and json.loads(body)["service"] == "kestrel-audio"
    assert http("GET", "/v1/stats")[0] == 401 and http("GET", "/v1/stats", key="nope")[0] == 401
    print("open endpoints and auth ok")

    key = sh("docker", "exec", NAME, "cat", "/data/key")
    assert len(key) == 64, "the key file was not created"
    code, _, _ = http("GET", "/v1/previews/1/info", key=key)
    assert code == 404

    q = "detection_id=1&species=Blue%20Jay&scientific=Cyanocitta%20cristata&camera=smoke"
    code, _, body = http("POST", f"/v1/jobs?{q}", key=key, body=synthetic_clip(), headers={"Content-Type": "audio/wav"})
    assert code == 202 and json.loads(body)["state"] == "pending", f"submit: {code} {body[:200]}"
    code, _, body = http("POST", f"/v1/jobs?{q}", key=key, body=b"ignored", headers={"Content-Type": "audio/wav"})
    assert code in (200, 202), "repeat submit must be a no-op"

    deadline = time.time() + 420
    info = {}
    while time.time() < deadline:
        code, _, body = http("GET", "/v1/previews/1/info", key=key)
        info = json.loads(body)
        if info.get("state") in ("ready", "failed"):
            break
        time.sleep(3)
    assert info.get("state") == "ready", f"job did not finish ready: {info}"
    assert info["segment"] and info["method"] == "trim" and info["cleaned"] is False, info
    assert -17.5 < info["loudnessLufs"] < -14.5, f"loudness {info['loudnessLufs']}"
    print("job ready:", {k: info[k] for k in ("segment", "method", "loudnessLufs", "durationS")})

    code, hdr, audio = http("GET", "/v1/previews/1", key=key)
    assert code == 200 and hdr.get("content-type") == "audio/mp4" and len(audio) > 5000
    code, hdr, part = http("GET", "/v1/previews/1", key=key, headers={"Range": "bytes=0-99"})
    assert code == 206 and len(part) == 100, f"range: {code}"
    probe = sh("docker", "exec", NAME, "ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels",
               "-of", "default=nw=1", "/data/previews/1.m4a")
    assert "codec_name=aac" in probe and "sample_rate=48000" in probe and "channels=1" in probe, probe
    print("preview ok:", probe.replace("\n", " "))

    # every ONNX model loads and runs on the CPU inside the image
    code = (
        "import numpy as np\n"
        "from kestrel_audio.mixit import OrtMixit\n"
        "from kestrel_audio.perch import OrtPerch\n"
        "x = (0.1 * np.random.default_rng(0).standard_normal(22050 * 6)).astype('float32')\n"
        "for k in (4, 8):\n"
        "    y = OrtMixit(f'/models/mixit{k}.onnx', k, device='cpu', threads=2, vram_mib=0).separate(x)\n"
        "    assert y.shape == (k, len(x)) and np.isfinite(y).all(), y.shape\n"
        "    print('mixit', k, 'ok', y.shape)\n"
        "p = OrtPerch('/models', device='cpu', threads=2, vram_mib=0)\n"
        "s = p.score([x], 0)[0]\n"
        "assert len(s.starts) >= 2 and np.isfinite(s.conf).all()\n"
        "print('perch ok', len(p.labels), 'labels,', len(s.starts), 'windows')\n"
    )
    print(sh("docker", "exec", NAME, "python", "-c", code))

    assert json.loads(http("DELETE", "/v1/previews/1", key=key)[2]) == {"deleted": True}
    assert http("GET", "/v1/previews/1", key=key)[0] == 404
    sh("docker", "rm", "-f", "-v", NAME)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
