import numpy as np
import torch

from roomest.features import FeatureConfig, LogMel, logmel_np, mel_filterbank


def test_numpy_and_torch_frontends_match():
    fc = FeatureConfig()
    x = np.random.default_rng(0).standard_normal(fc.fs * 2).astype(np.float32) * 0.1
    a = logmel_np(x, fc)
    b = LogMel(fc)(torch.from_numpy(x)[None])[0].numpy()
    assert a.shape == b.shape == (fc.n_mels, fc.n_frames(len(x)))
    assert np.max(np.abs(a - b)) < 1e-3


def test_gain_invariance():
    fc = FeatureConfig()
    x = np.random.default_rng(0).standard_normal(fc.fs)
    assert np.allclose(logmel_np(x, fc), logmel_np(x * 0.01, fc), atol=5e-3)  # residual from log epsilon at -40 dB


def test_filterbank_covers_spectrum():
    fb = mel_filterbank(FeatureConfig())
    assert (fb.sum(0)[5:-10] > 0).all()
