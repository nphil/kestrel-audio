"""Audio file I/O through ffmpeg, and the resamplers. Decoding reads files (not pipes) so any container works,
including MP4/M4A with the index at the end."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SR = 22050            # working rate: MixIT's rate, plenty for a camera mic that tops out near 8 kHz
PERCH_SR = 32000
OUT_SR = 48000
AAC_BITRATE = "96k"


class AudioError(RuntimeError):
    """The clip could not be decoded or encoded."""


def _run(cmd: list[str], data: bytes | None = None, timeout: int = 60) -> bytes:
    try:
        p = subprocess.run(cmd, input=data, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise AudioError("ffmpeg timed out") from e
    if p.returncode != 0:
        raise AudioError(p.stderr.decode(errors="replace").strip().splitlines()[-1][:300] if p.stderr else "ffmpeg failed")
    return p.stdout


def decode_file(path: str | Path, sr: int = SR, max_seconds: float | None = None) -> np.ndarray:
    """Any audio file -> mono float32 at `sr` (values in [-1, 1])."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr)]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.3f}"]
    raw = _run(cmd + ["-f", "f32le", "pipe:1"])
    x = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if x.size == 0:
        raise AudioError("the file holds no audio")
    return x


def to_perch_rate(x: np.ndarray) -> np.ndarray:
    return resample_poly(np.asarray(x, dtype=np.float64), 640, 441).astype(np.float32)   # 22050 -> 32000


def to_output_rate(x: np.ndarray) -> np.ndarray:
    return resample_poly(np.asarray(x, dtype=np.float64), 320, 147).astype(np.float32)   # 22050 -> 48000


class AacCodec:
    """AAC-LC 96 kb/s mono in an .m4a, with the decoded result available so levels can be measured on what is shipped."""

    def __init__(self, sr: int = OUT_SR, bitrate: str = AAC_BITRATE):
        self.sr, self.bitrate = sr, bitrate
        self.last_bytes: bytes = b""

    def encode(self, y: np.ndarray) -> bytes:
        with tempfile.TemporaryDirectory(prefix="ka-aac-") as td:
            out = Path(td) / "out.m4a"
            _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(self.sr), "-ac", "1", "-i", "pipe:0",
                  "-c:a", "aac", "-b:a", self.bitrate, "-movflags", "+faststart", str(out)],
                 data=np.clip(y, -1.0, 1.0).astype("<f4").tobytes())
            return out.read_bytes()

    def decode(self, data: bytes, sr: int | None = None) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="ka-aac-") as td:
            f = Path(td) / "in.m4a"
            f.write_bytes(data)
            return decode_file(f, sr or self.sr)

    def encode_decode(self, y: np.ndarray) -> np.ndarray:
        self.last_bytes = self.encode(y)
        return self.decode(self.last_bytes)
