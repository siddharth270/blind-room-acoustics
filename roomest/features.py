"""Log-mel front-end, implemented twice with identical maths:

* `LogMel` (torch)  - used during training, runs batched on the accelerator.
* `logmel_np` (numpy) - used by the real-time runtime (no torch dependency) and
  serves as the reference spec for an embedded C port.

`tests/test_features.py` checks the two agree.

Normalisation: a single scalar (the mean log energy) is subtracted per clip.
This makes the feature invariant to playback/mic gain while preserving the
spectral shape, which carries per-band decay information (per-frequency
mean normalisation would remove it).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

LOG_EPS = 1e-8


@dataclass
class FeatureConfig:
    fs: int = 16000
    n_fft: int = 512
    win: int = 400
    hop: int = 160
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 7800.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "FeatureConfig":
        return cls(fs=cfg["fs"], **cfg["features"])

    def n_frames(self, n_samples: int) -> int:
        return 1 + (n_samples - self.n_fft) // self.hop

    def to_dict(self) -> dict:
        return asdict(self)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(fc: FeatureConfig) -> np.ndarray:
    """(n_mels, n_fft//2+1) triangular, area-normalised (Slaney-style) HTK-mel filterbank."""
    freqs = np.linspace(0, fc.fs / 2, fc.n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fc.fmin), _hz_to_mel(fc.fmax), fc.n_mels + 2))
    fb = np.zeros((fc.n_mels, len(freqs)), dtype=np.float64)
    for i in range(fc.n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (ce - lo)
        down = (hi - freqs) / (hi - ce)
        fb[i] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (hi - lo))
    return fb.astype(np.float32)


def _window(fc: FeatureConfig) -> np.ndarray:
    """Periodic Hann of length `win`, zero-padded (centred) to n_fft - matches torch.stft."""
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fc.win) / fc.win)
    left = (fc.n_fft - fc.win) // 2
    out = np.zeros(fc.n_fft, dtype=np.float64)
    out[left:left + fc.win] = w
    return out


def logmel_np(x: np.ndarray, fc: FeatureConfig, fb: np.ndarray | None = None) -> np.ndarray:
    """x: (n_samples,) float -> (n_mels, n_frames) float32."""
    fb = mel_filterbank(fc) if fb is None else fb
    x = np.asarray(x, dtype=np.float64)
    n = fc.n_frames(len(x))
    idx = np.arange(fc.n_fft)[None, :] + fc.hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * _window(fc), axis=1)) ** 2
    mel = np.log(fb.astype(np.float64) @ spec.T + LOG_EPS)
    return (mel - mel.mean()).astype(np.float32)


class LogMel(nn.Module):
    def __init__(self, fc: FeatureConfig):
        super().__init__()
        self.fc = fc
        self.register_buffer("fb", torch.from_numpy(mel_filterbank(fc)))
        w = torch.from_numpy(
            0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fc.win) / fc.win)).float()
        self.register_buffer("window", w)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_samples) -> (B, n_mels, n_frames)."""
        spec = torch.stft(x, self.fc.n_fft, hop_length=self.fc.hop, win_length=self.fc.win,
                          window=self.window, center=False, return_complex=True)
        mel = torch.log(torch.matmul(self.fb, spec.abs() ** 2) + LOG_EPS)
        return mel - mel.mean(dim=(1, 2), keepdim=True)
