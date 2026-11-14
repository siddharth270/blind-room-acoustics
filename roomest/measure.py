"""Measure real room impulse responses with the exponential sine sweep (ESS) method
(Farina 2000) and analyse them with the same code that labels the training data.

    # play sweeps through the default output, record from the default input
    python -m roomest.measure sweep --out data/measured/livingroom_pos1.wav --repeats 3
    # analyse any RIR file (prints labels, saves EDC plot)
    python -m roomest.measure analyze data/measured/livingroom_pos1.wav

A matching MATLAB implementation lives in matlab/ (measure_rir.m, analyze_rir.m).
Tips: speaker ~1-3 m from the mic, nobody moving, moderate level (no clipping);
3+ repeats are averaged to gain ~5 dB SNR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .acoustics import OCTAVE_BANDS, TARGET_NAMES, bandpass, compute_labels, schroeder_edc_db


def ess(f1: float, f2: float, duration: float, fs: int, fade_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Exponential sine sweep and its amplitude-compensated inverse filter.
    conv(sweep, inverse) ~ band-limited delta at index len(sweep)-1, peak = 1."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    R = np.log(f2 / f1)
    x = np.sin(2 * np.pi * f1 * duration / R * (np.exp(t * R / duration) - 1))
    nf = int(fade_s * fs)
    fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
    x[:nf] *= fade
    x[-nf:] *= fade[::-1]
    inv = x[::-1] * np.exp(-t * R / duration)  # -6 dB/oct to whiten the pink sweep energy
    inv /= np.max(np.abs(signal.fftconvolve(x, inv)))
    return x, inv


def deconvolve(recording: np.ndarray, sweep: np.ndarray, inverse: np.ndarray, ir_len: int) -> np.ndarray:
    full = signal.fftconvolve(recording, inverse)
    start = len(sweep) - 1
    return full[start:start + ir_len]


def measure(fs=48000, f1=40.0, f2=None, duration=5.0, tail_s=3.0, repeats=3, level=0.3, device=None):
    import sounddevice as sd
    f2 = f2 or 0.45 * fs
    sweep, inv = ess(f1, f2, duration, fs)
    play = np.concatenate([sweep * level, np.zeros(int(tail_s * fs))]).astype(np.float32)
    irs = []
    for r in range(repeats):
        print(f"sweep {r + 1}/{repeats} ...", flush=True)
        rec = sd.playrec(play, samplerate=fs, channels=1, device=device, blocking=True)[:, 0]
        if np.max(np.abs(rec)) > 0.98:
            print("  WARNING: input clipped - lower --level or the mic gain")
        irs.append(deconvolve(rec / level, sweep, inv, int(tail_s * fs)))
    # align repeats on their peak before averaging (absorbs I/O jitter)
    peaks = [int(np.argmax(np.abs(h))) for h in irs]
    p0 = min(peaks)
    aligned = [h[p - p0:p - p0 + len(h) - (max(peaks) - p0)] for h, p in zip(irs, peaks)]
    return np.mean(aligned, axis=0), fs


def analyze(path: str | Path, plot: bool = True) -> dict:
    h, fs = sf.read(path, dtype="float64", always_2d=True)
    h = h[:, 0]
    lab = compute_labels(h, fs, noise_comp=True)
    res = {n: (None if not np.isfinite(v) else round(float(v), 3)) for n, v in zip(TARGET_NAMES, lab)}
    res.pop("log10_volume")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for fc in OCTAVE_BANDS:
            edc = schroeder_edc_db(bandpass(h, fc, fs), fs, noise_comp=True)
            ax.plot(np.arange(len(edc)) / fs, edc, lw=1.6, label=f"{fc} Hz")
        ax.set_ylim(-70, 2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("energy decay (dB)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.set_title(f"Schroeder decay curves - {Path(path).stem}")
        fig.tight_layout()
        out = Path(path).with_suffix(".edc.png")
        fig.savefig(out, dpi=130)
        res["plot"] = str(out)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep", help="measure an RIR")
    s.add_argument("--out", required=True)
    s.add_argument("--fs", type=int, default=48000)
    s.add_argument("--seconds", type=float, default=5.0)
    s.add_argument("--tail", type=float, default=3.0)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--level", type=float, default=0.3)
    s.add_argument("--device", default=None)
    a = sub.add_parser("analyze", help="compute T60/DRR/C50 for an RIR file")
    a.add_argument("path")
    args = ap.parse_args()
    if args.cmd == "sweep":
        h, fs = measure(args.fs, duration=args.seconds, tail_s=args.tail, repeats=args.repeats,
                        level=args.level, device=args.device)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.out, (h / np.max(np.abs(h)) * 0.9).astype(np.float32), fs, subtype="FLOAT")
        print(f"saved {args.out}")
        print(json.dumps(analyze(args.out), indent=2))
    else:
        print(json.dumps(analyze(args.path), indent=2))


if __name__ == "__main__":
    main()
