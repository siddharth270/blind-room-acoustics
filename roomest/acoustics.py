"""Room-acoustic parameter extraction from room impulse responses (RIRs).

Implements the ISO 3382-1 style quantities used as ground-truth labels:
  * T60 per octave band (T30 with T20 fallback) via Schroeder backward integration
  * DRR  - direct-to-reverberant energy ratio
  * C50  - speech clarity (early 50 ms vs late energy)

Measured RIRs contain a noise floor; `noise_comp=True` applies a simplified
Lundeby-style truncation + Chu noise subtraction before integration.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

OCTAVE_BANDS = (125, 250, 500, 1000, 2000, 4000)
T60_NAMES = [f"t60_{f}" for f in OCTAVE_BANDS]
TARGET_NAMES = T60_NAMES + ["drr", "c50", "log10_volume"]
N_TARGETS = len(TARGET_NAMES)

_EPS = 1e-12


def octave_sos(fc: float, fs: int, order: int = 3) -> np.ndarray:
    lo = fc / np.sqrt(2.0)
    hi = min(fc * np.sqrt(2.0), 0.95 * fs / 2)
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def bandpass(x: np.ndarray, fc: float, fs: int) -> np.ndarray:
    """Zero-phase octave-band filter (keeps onset timing intact).

    The input is zero-padded instead of using filtfilt's default odd-reflection
    padding: for an RIR that starts at a strong direct impulse, reflection creates
    a large artificial step whose low-frequency transient swamps the decay.
    The output keeps the leading pad so the filter's pre-ringing is preserved
    (it is `pad` samples longer than the input; onset detection absorbs the shift).
    """
    pad = int(0.1 * fs)
    y = signal.sosfiltfilt(octave_sos(fc, fs), np.pad(np.asarray(x, dtype=np.float64), (pad, pad)),
                           padtype=None)
    return y[:-pad]


def find_onset(h: np.ndarray, threshold_db: float = -20.0) -> int:
    """First sample whose energy is within `threshold_db` of the peak (ISO 3382)."""
    e = h.astype(np.float64) ** 2
    return int(np.argmax(e >= e.max() * 10 ** (threshold_db / 10)))


def _noise_truncation(e: np.ndarray, fs: int) -> tuple[int, float]:
    """Simplified Lundeby: estimate noise floor from the last 10 % of the response
    and find where the smoothed envelope meets noise + 5 dB."""
    n = len(e)
    tail = e[int(0.9 * n):]
    win = max(1, int(0.01 * fs))
    # Only treat the tail as a noise floor if it is stationary. Many published RIRs
    # are already truncated at the noise floor, so their last 10 % is still decaying
    # signal - subtracting it would destroy the decay we want to measure.
    half = len(tail) // 2
    if half < win or abs(10 * np.log10((np.mean(tail[:half]) + _EPS) / (np.mean(tail[half:]) + _EPS))) > 2.0:
        return n, 0.0
    noise = float(np.mean(tail))
    env = np.convolve(e, np.ones(win) / win, mode="same")
    env_db = 10 * np.log10(env + _EPS)
    noise_db = 10 * np.log10(noise + _EPS)
    peak = int(np.argmax(env_db))
    below = np.nonzero(env_db[peak:] <= noise_db + 5.0)[0]
    cut = peak + int(below[0]) if len(below) else n
    return max(cut, win * 2), noise


def schroeder_edc_db(h: np.ndarray, fs: int, noise_comp: bool = False) -> np.ndarray:
    """Energy decay curve (dB, normalised to 0 dB at onset)."""
    h = np.asarray(h, dtype=np.float64)
    e = h[find_onset(h):] ** 2
    if noise_comp:
        cut, noise = _noise_truncation(e, fs)
        e = np.clip(e[:cut] - noise, 0.0, None)
    edc = np.cumsum(e[::-1])[::-1]
    return 10 * np.log10(edc / (edc[0] + _EPS) + _EPS)


def decay_time(edc_db: np.ndarray, fs: int, start_db: float, end_db: float) -> float:
    """Extrapolated -60 dB decay time from a linear fit between start_db and end_db."""
    i0 = np.argmax(edc_db <= start_db)
    if edc_db[i0] > start_db:
        return np.nan
    after = np.nonzero(edc_db[i0:] <= end_db)[0]
    if not len(after):
        return np.nan
    i1 = i0 + int(after[0])
    if i1 - i0 < 4:
        return np.nan
    t = np.arange(i0, i1) / fs
    slope = np.polyfit(t, edc_db[i0:i1], 1)[0]
    return float(-60.0 / slope) if slope < 0 else np.nan


def t60_from_edc(edc_db: np.ndarray, fs: int) -> float:
    t30 = decay_time(edc_db, fs, -5.0, -35.0)
    return t30 if np.isfinite(t30) else decay_time(edc_db, fs, -5.0, -25.0)


def band_t60(h: np.ndarray, fs: int, bands=OCTAVE_BANDS, noise_comp: bool = False) -> np.ndarray:
    return np.array(
        [t60_from_edc(schroeder_edc_db(bandpass(h, fc, fs), fs, noise_comp), fs) for fc in bands]
    )


def edt(h: np.ndarray, fs: int, noise_comp: bool = False) -> float:
    return decay_time(schroeder_edc_db(h, fs, noise_comp), fs, 0.0, -10.0)


def _direct_window(h: np.ndarray, fs: int, window_ms: float) -> tuple[int, int]:
    peak = int(np.argmax(np.abs(h)))
    w = int(round(window_ms * 1e-3 * fs))
    return max(0, peak - w), peak + w + 1


def drr(h: np.ndarray, fs: int, window_ms: float = 2.5, noise_comp: bool = False) -> float:
    h = np.asarray(h, dtype=np.float64)
    a, b = _direct_window(h, fs, window_ms)
    late = h[b:] ** 2
    if noise_comp:
        cut, noise = _noise_truncation(late, fs)
        late = np.clip(late[:cut] - noise, 0.0, None)
    return float(10 * np.log10((np.sum(h[a:b] ** 2) + _EPS) / (np.sum(late) + _EPS)))


def clarity(h: np.ndarray, fs: int, early_ms: float = 50.0, window_ms: float = 2.5,
            noise_comp: bool = False) -> float:
    h = np.asarray(h, dtype=np.float64)
    a, _ = _direct_window(h, fs, window_ms)
    split = a + int(round(early_ms * 1e-3 * fs))
    late = h[split:] ** 2
    if noise_comp:
        cut, noise = _noise_truncation(late, fs)
        late = np.clip(late[:cut] - noise, 0.0, None)
    return float(10 * np.log10((np.sum(h[a:split] ** 2) + _EPS) / (np.sum(late) + _EPS)))


def compute_labels(h: np.ndarray, fs: int, volume: float = np.nan, noise_comp: bool = False) -> np.ndarray:
    """Label vector in TARGET_NAMES order. Unavailable quantities are NaN (masked in training)."""
    t60 = band_t60(h, fs, noise_comp=noise_comp)
    vol = np.log10(volume) if np.isfinite(volume) and volume > 0 else np.nan
    return np.concatenate([t60, [drr(h, fs, noise_comp=noise_comp),
                                 clarity(h, fs, noise_comp=noise_comp), vol]]).astype(np.float32)


def sabine_t60(volume: float, surface_absorption: float) -> float:
    """Sabine's formula, T60 = 0.161 V / A (A in m^2 Sabins)."""
    return 0.161 * volume / max(surface_absorption, _EPS)


def synthetic_rir(t60: float, fs: int, length_s: float = 2.0, drr_db: float = 0.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Exponentially decaying noise RIR with a known T60 / DRR (used for tests)."""
    rng = rng or np.random.default_rng(0)
    n = int(length_s * fs)
    t = np.arange(n) / fs
    tail = rng.standard_normal(n) * np.exp(-3 * np.log(10) * t / t60)
    tail[:int(0.003 * fs)] = 0.0  # gap after direct sound
    h = tail
    rev_energy = np.sum(tail ** 2)
    h[0] = np.sqrt(rev_energy * 10 ** (drr_db / 10))
    return h
