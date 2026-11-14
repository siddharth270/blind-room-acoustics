"""Classical (non-learned) blind T60 baseline.

Free-decay-region method (after Ratnam et al. 2003 / Eaton et al. 2016, ACE
challenge baselines, simplified): per octave band, compute a short-time energy
envelope in dB, locate segments of monotonic energy decay of sufficient length
and depth (speech offsets followed by free reverberant decay), fit a line to
each, convert the slope to a T60 and take the median.

The raw estimate is biased (speech offsets are not true impulses), so - as is
standard - a per-band linear mapping is fitted on *training* data.
"""
from __future__ import annotations

import numpy as np

from .acoustics import OCTAVE_BANDS, bandpass


def free_decay_t60(x: np.ndarray, fs: int, fc: float, frame_ms=20.0, hop_ms=5.0,
                   min_len_ms=80.0, min_drop_db=10.0, tol_db=0.5) -> float:
    y = bandpass(np.asarray(x, dtype=np.float64), fc, fs)
    fl, hp = int(frame_ms * 1e-3 * fs), int(hop_ms * 1e-3 * fs)
    n = 1 + (len(y) - fl) // hp
    if n < 4:
        return np.nan
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    env = 10 * np.log10(np.mean(y[idx] ** 2, axis=1) + 1e-20)
    floor = np.percentile(env, 10)
    min_len = int(min_len_ms / hop_ms)
    estimates = []
    i = 0
    while i < n - 1:
        j = i
        while j + 1 < n and env[j + 1] <= env[j] + tol_db:
            j += 1
        seg = env[i:j + 1]
        # stop at the noise floor so the fit only covers the free decay
        valid = seg > floor + 3
        seg = seg[valid] if valid.any() else seg
        if len(seg) >= min_len and seg[0] - seg[-1] >= min_drop_db:
            t = np.arange(len(seg)) * hp / fs
            slope = np.polyfit(t, seg, 1)[0]
            if slope < 0:
                estimates.append(-60.0 / slope)
        i = j + 1
    return float(np.median(estimates)) if estimates else np.nan


def blind_t60_bands(x: np.ndarray, fs: int, bands=OCTAVE_BANDS) -> np.ndarray:
    return np.array([free_decay_t60(x, fs, fc) for fc in bands])


class CalibratedBlindT60:
    """Per-band log-linear calibration: log T60 = a * log raw + b, fitted on train data."""

    def __init__(self, fs: int):
        self.fs = fs
        self.coef = None
        self.fallback = None

    def fit(self, wavs: list[np.ndarray], t60: np.ndarray) -> "CalibratedBlindT60":
        raw = np.stack([blind_t60_bands(w, self.fs) for w in wavs])
        self.coef, self.fallback = [], np.nanmedian(t60, 0)
        for b in range(raw.shape[1]):
            ok = np.isfinite(raw[:, b]) & np.isfinite(t60[:, b])
            if ok.sum() < 5:
                self.coef.append((0.0, np.log(self.fallback[b])))
                continue
            self.coef.append(tuple(np.polyfit(np.log(raw[ok, b]), np.log(t60[ok, b]), 1)))
        return self

    def predict(self, wav: np.ndarray) -> np.ndarray:
        raw = blind_t60_bands(wav, self.fs)
        out = np.empty(len(raw))
        for b, (a, c) in enumerate(self.coef):
            out[b] = np.exp(a * np.log(raw[b]) + c) if np.isfinite(raw[b]) else self.fallback[b]
        return out
