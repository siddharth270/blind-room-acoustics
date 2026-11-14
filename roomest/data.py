"""Training-data pipeline: dry source x RIR + noise, generated on the fly.

Every training example is a fresh random combination of
    (dry speech/music/synthetic source) * (RIR) + (coloured noise @ random SNR) * gain
so the model sees effectively unlimited distinct reverberant signals from a
fixed RIR bank. Val/test examples are seeded per index -> fully reproducible.

Leakage control:
  * RIRs are split by room (see simulate.assign_splits)
  * dry sources are split by speaker / parent folder
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch.utils.data import Dataset

from .acoustics import N_TARGETS, T60_NAMES, TARGET_NAMES, compute_labels, find_onset
from .config import resolve

SPLITS = {"train": 0, "val": 1, "test": 2}
AUDIO_EXT = {".wav", ".flac", ".ogg"}


# --------------------------------------------------------------------------- RIRs
class RirStore:
    """RIR bank backed by a memmapped .npy (opened lazily so DataLoader workers
    don't pickle the whole array) or by an in-memory array (measured RIRs)."""

    def __init__(self, sim_dir: str | Path | None = None, rirs: np.ndarray | None = None,
                 labels: np.ndarray | None = None, split: np.ndarray | None = None, names=None):
        self.sim_dir = Path(sim_dir) if sim_dir else None
        self._rirs = rirs
        if self.sim_dir is not None:
            self.labels = np.load(self.sim_dir / "labels.npy")
            meta = np.load(self.sim_dir / "meta.npz")
            self.split = meta["split"]
            self.meta = {k: meta[k] for k in meta.files}
        else:
            self.labels, self.split = labels, split if split is not None else np.full(len(labels), 2)
            self.meta = {}
        self.names = names

    def __len__(self):
        return len(self.labels)

    def __getstate__(self):
        st = self.__dict__.copy()
        if self.sim_dir is not None:
            st["_rirs"] = None
        return st

    def rir(self, i: int) -> np.ndarray:
        if self._rirs is None:
            self._rirs = np.load(self.sim_dir / "rirs.npy", mmap_mode="r")
        return np.asarray(self._rirs[i], dtype=np.float32)

    def indices(self, split: str) -> np.ndarray:
        return np.nonzero(self.split == SPLITS[split])[0]


def load_real_rirs(rir_dir: str | Path, fs: int, max_len_s: float = 2.0,
                   min_dynamic_range_db: float = 20.0, exclude_regex: str | None = None) -> RirStore:
    """Load measured RIRs (any sample rate), resample, compute labels with noise
    compensation, and drop responses too short / noisy to label reliably.
    `exclude_regex` drops files by name (e.g. outdoor spaces: no reverberant field)."""
    import re
    from math import gcd
    files = sorted(p for p in Path(rir_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if exclude_regex:
        files = [p for p in files if not re.search(exclude_regex, p.stem)]
    rirs, labels, names = [], [], []
    n_max = int(max_len_s * fs)
    for p in files:
        h, sr = sf.read(p, dtype="float64", always_2d=True)
        h = h[:, 0]
        if sr != fs:
            g = gcd(sr, fs)
            h = signal.resample_poly(h, fs // g, sr // g)
        onset = find_onset(h)
        h = h[max(0, onset - int(0.002 * fs)):]
        peak_db = 20 * np.log10(np.max(np.abs(h)) + 1e-12)
        noise_db = 10 * np.log10(np.mean(h[int(0.9 * len(h)):] ** 2) + 1e-20)
        if len(h) < int(0.1 * fs) or peak_db - noise_db < min_dynamic_range_db:
            continue
        lab = compute_labels(h, fs, noise_comp=True)
        h = h[:n_max] / (np.max(np.abs(h)) + 1e-12)
        buf = np.zeros(n_max, dtype=np.float32)
        buf[:len(h)] = h
        rirs.append(buf), labels.append(lab), names.append(p.stem)
    return RirStore(rirs=np.stack(rirs), labels=np.stack(labels), names=names)


# ------------------------------------------------------------------------- sources
def discover_sources(dirs: list[str], fractions: dict, seed: int) -> dict[str, list[Path]]:
    """Find dry audio files and split them by group (LibriSpeech speaker = grandparent dir)."""
    files = []
    for d in dirs:
        d = resolve(d)
        if d.exists():
            files += sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        return {"train": [], "val": [], "test": []}
    groups = sorted({p.parent.parent.name for p in files})
    perm = list(np.random.default_rng(seed).permutation(groups))
    n_tr = int(round(fractions["train"] * len(perm)))
    n_va = max(1, int(round(fractions["val"] * len(perm))))
    split_of = {g: "train" for g in perm[:n_tr]}
    split_of.update({g: "val" for g in perm[n_tr:n_tr + n_va]})
    split_of.update({g: "test" for g in perm[n_tr + n_va:]})
    out = {"train": [], "val": [], "test": []}
    for p in files:
        out[split_of[p.parent.parent.name]].append(p)
    return out


def synth_source(rng: np.random.Generator, n: int, fs: int) -> np.ndarray:
    """Speech-like / impulsive synthetic source: harmonic or noise bursts separated by
    silent gaps. Gaps expose free reverberant decay, onsets expose the direct path."""
    x = np.zeros(n)
    t = 0
    while t < n:
        t += int(rng.uniform(0.02, 0.35) * fs)  # gap
        dur = int(rng.uniform(0.03, 0.4) * fs)
        seg = np.arange(min(dur, max(0, n - t)))
        if not len(seg):
            break
        kind = rng.random()
        if kind < 0.5:  # voiced: harmonics with vibrato
            f0 = rng.uniform(80, 300) * (1 + 0.03 * np.sin(2 * np.pi * rng.uniform(3, 7) * seg / fs))
            phase = 2 * np.pi * np.cumsum(f0) / fs
            amps = rng.uniform(0, 1, 12) / np.arange(1, 13)
            burst = sum(a * np.sin((k + 1) * phase) for k, a in enumerate(amps))
        elif kind < 0.85:  # band-limited noise (fricatives, rustle)
            lo = rng.uniform(100, 3000)
            sos = signal.butter(2, [lo, min(lo * rng.uniform(1.5, 6), 0.45 * fs)], "bandpass", fs=fs, output="sos")
            burst = signal.sosfilt(sos, rng.standard_normal(len(seg)))
        else:  # impulsive (clap, knock, door)
            burst = rng.standard_normal(len(seg)) * np.exp(-seg / (rng.uniform(0.002, 0.02) * fs))
        env = np.minimum(1, np.minimum(seg, len(seg) - seg) / (0.005 * fs + 1))
        x[t:t + len(seg)] = burst * env * rng.uniform(0.3, 1)
        t += len(seg)
    return x


class SourceBank:
    def __init__(self, files: list[Path], fs: int):
        self.fs = fs
        self.files, self.frames, self.srs = [], [], []
        for p in files:
            info = sf.info(p)
            if info.frames / info.samplerate > 1.0:
                self.files.append(p), self.frames.append(info.frames), self.srs.append(info.samplerate)

    def __len__(self):
        return len(self.files)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        i = int(rng.integers(len(self.files)))
        sr = self.srs[i]
        need = int(np.ceil(n * sr / self.fs))
        start = int(rng.integers(0, max(1, self.frames[i] - need)))
        x, _ = sf.read(self.files[i], start=start, frames=need, dtype="float64", always_2d=True)
        x = x.mean(1)
        if sr != self.fs:
            from math import gcd
            g = gcd(sr, self.fs)
            x = signal.resample_poly(x, self.fs // g, sr // g)
        if len(x) < n:
            x = np.tile(x, int(np.ceil(n / max(len(x), 1))))
        return x[:n]


# ------------------------------------------------------------------- augmentation
def coloured_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """White / pink / brown noise via spectral shaping (exponent 0, 1, 2)."""
    beta = rng.choice([0.0, 1.0, 2.0])
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=float)
    f[0] = 1.0
    return np.fft.irfft(spec / f ** (beta / 2), n)


def _rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-20)


# ------------------------------------------------------------------- targets
class TargetScaler:
    """T60 values are modelled in log-space (errors are perceived relatively);
    all targets are then standardised with train-set statistics."""
    log_mask = np.array([n in T60_NAMES for n in TARGET_NAMES])

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def _fwd(self, y):
        y = np.array(y, dtype=np.float64, copy=True)
        y[..., self.log_mask] = np.log(np.clip(y[..., self.log_mask], 1e-3, None))
        return y

    def fit(self, labels: np.ndarray) -> "TargetScaler":
        z = self._fwd(labels)
        self.mean = np.nanmean(z, 0).astype(np.float32)
        self.std = (np.nanstd(z, 0) + 1e-6).astype(np.float32)
        return self

    def transform(self, y):
        return ((self._fwd(y) - self.mean) / self.std).astype(np.float32)

    def inverse(self, z):
        y = np.asarray(z, dtype=np.float64) * self.std + self.mean
        y[..., self.log_mask] = np.exp(y[..., self.log_mask])
        return y

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ------------------------------------------------------------------- dataset
class ReverbDataset(Dataset):
    """Returns (waveform[n], normalised_target[N_TARGETS], rir_index)."""

    def __init__(self, store: RirStore, rir_indices: np.ndarray, sources: SourceBank | None,
                 cfg: dict, scaler: TargetScaler, train: bool, length: int, seed: int = 0):
        self.store, self.idx, self.sources, self.scaler = store, np.asarray(rir_indices), sources, scaler
        self.train, self.length, self.seed = train, length, seed
        d = cfg["data"]
        self.fs = cfg["fs"]
        self.n = int(d["clip_s"] * self.fs)
        self.warm = int(d["warmup_s"] * self.fs)
        self.snr, self.p_noise = d["snr_db"], d["p_noise"]
        self.p_synth = 1.0 if sources is None or len(sources) == 0 else d["p_synth_source"]

    def __len__(self):
        return self.length

    def render(self, rng: np.random.Generator, h: np.ndarray) -> np.ndarray:
        total = self.n + self.warm
        x = synth_source(rng, total, self.fs) if rng.random() < self.p_synth else self.sources.sample(rng, total)
        x = x / _rms(x)
        y = signal.fftconvolve(x, h)[self.warm:self.warm + self.n]
        if rng.random() < self.p_noise:
            noise = coloured_noise(rng, self.n)
            snr = rng.uniform(*self.snr)
            y = y + noise * _rms(y) / _rms(noise) * 10 ** (-snr / 20)
        y = y / _rms(y) * 10 ** (rng.uniform(-40, -15) / 20)   # random playback level
        peak = np.max(np.abs(y))
        if peak > 0.99:
            y *= 0.99 / peak
        return y.astype(np.float32)

    def __getitem__(self, i):
        if self.train:
            rng = np.random.default_rng()
            j = int(self.idx[rng.integers(len(self.idx))])
        else:
            rng = np.random.default_rng([self.seed, i])
            j = int(self.idx[i % len(self.idx)])
        wav = self.render(rng, self.store.rir(j))
        tgt = self.scaler.transform(self.store.labels[j])
        return torch.from_numpy(wav), torch.from_numpy(tgt), j


def worker_init(_):
    np.random.seed(torch.initial_seed() % 2**32)


assert N_TARGETS == len(TARGET_NAMES)
