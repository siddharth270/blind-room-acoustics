"""C front-end parity: embedded/logmel.c must match roomest.features.logmel_np."""
import ctypes
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from roomest.features import FeatureConfig, logmel_np

EMB = Path(__file__).resolve().parents[1] / "embedded"


@pytest.fixture(scope="module")
def lib():
    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    subprocess.run(["make", "-s", "-C", str(EMB)], check=True)
    ext = "dylib" if platform.system() == "Darwin" else "so"
    return ctypes.CDLL(str(EMB / f"liblogmel.{ext}"))


def test_c_matches_numpy(lib):
    fc = FeatureConfig()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(fc.fs * 4) * 0.05).astype(np.float32)
    n = fc.n_frames(len(x))
    out = np.zeros(fc.n_mels * n, dtype=np.float32)
    state = ctypes.create_string_buffer(4 * (2 * fc.n_fft + fc.n_fft // 2 + 1))
    f = ctypes.POINTER(ctypes.c_float)
    got = lib.lm_clip(state, x.ctypes.data_as(f), len(x), out.ctypes.data_as(f), n)
    assert got == n
    ref = logmel_np(x, fc)
    assert np.max(np.abs(out.reshape(fc.n_mels, n) - ref)) < 2e-3
