"""Google's bird MixIT separation models (4 and 8 tracks, 22.05 kHz) through onnxruntime."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .perch import make_session, run_options

MIXIT_FILES = {4: "mixit4.onnx", 8: "mixit8.onnx"}


class OrtMixit:
    def __init__(self, path: str | Path, k: int, *, device: str, threads: int, vram_mib: int):
        self.k = k
        self.sess = make_session(path, device=device, threads=threads, vram_mib=vram_mib)
        self._in = self.sess.get_inputs()[0].name
        self._out = self.sess.get_outputs()[0].name
        self._ro = run_options(device)

    def separate(self, x: np.ndarray) -> np.ndarray:
        """x: mono float32 at 22.05 kHz -> (k, len(x)) float32."""
        out = self.sess.run([self._out], {self._in: np.ascontiguousarray(x, dtype=np.float32)[None, None, :]}, self._ro)[0]
        y = np.asarray(out)[0]
        if y.shape[0] != self.k:
            raise RuntimeError(f"expected {self.k} tracks, the model returned {y.shape[0]}")
        return y.astype(np.float32, copy=False)


class LazySeparator:
    """Opens the onnxruntime session the first time it is needed (a clip whose species Perch does not know never needs it)
    and keeps it: on the GPU only the weights stay resident between runs (see `run_options`)."""

    def __init__(self, path: str | Path, k: int, *, device: str, threads: int, vram_mib: int):
        self.k = k
        self._args = (path, k)
        self._kw = {"device": device, "threads": threads, "vram_mib": vram_mib}
        self._model: OrtMixit | None = None

    def separate(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            self._model = OrtMixit(*self._args, **self._kw)
        return self._model.separate(x)
