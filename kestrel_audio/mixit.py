"""Google's bird MixIT separation models (4 and 8 tracks, 22.05 kHz) through onnxruntime."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .perch import make_session

MIXIT_FILES = {4: "mixit4.onnx", 8: "mixit8.onnx"}


class OrtMixit:
    def __init__(self, path: str | Path, k: int, *, device: str, threads: int, vram_mib: int):
        self.k = k
        self.sess = make_session(path, device=device, threads=threads, vram_mib=vram_mib)
        self._in = self.sess.get_inputs()[0].name
        self._out = self.sess.get_outputs()[0].name

    def separate(self, x: np.ndarray) -> np.ndarray:
        """x: mono float32 at 22.05 kHz -> (k, len(x)) float32."""
        out = self.sess.run([self._out], {self._in: np.ascontiguousarray(x, dtype=np.float32)[None, None, :]})[0]
        y = np.asarray(out)[0]
        if y.shape[0] != self.k:
            raise RuntimeError(f"expected {self.k} tracks, the model returned {y.shape[0]}")
        return y.astype(np.float32, copy=False)
