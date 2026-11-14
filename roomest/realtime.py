"""Real-time blind room-acoustics estimation (numpy + onnxruntime; no torch).

Live microphone:
    python -m roomest.realtime --model exported/tcn_student
File playback at real-time pace (no mic needed), optionally through a known RIR
so the dashboard shows ground truth next to the estimate:
    python -m roomest.realtime --model exported/tcn_student --wav speech.flac \
        --rir data/raw/mit_ir/Audio/h001_Bedroom_65txts.wav

Pipeline per hop (default 0.5 s): ring buffer (4 s) -> activity gate -> log-mel ->
int8 ONNX model -> de-normalise -> exponential smoothing -> dashboard / CSV log.
The gate holds the last estimate during silence, when there is nothing to observe.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import time
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy import signal

from .acoustics import OCTAVE_BANDS, compute_labels
from .features import FeatureConfig, logmel_np, mel_filterbank


class RoomEstimator:
    """Stateful streaming estimator. Feed arbitrary-size blocks with `push`."""

    def __init__(self, model_dir: str | Path, variant: str = "int8", threads: int = 1):
        model_dir = Path(model_dir)
        rt = json.loads((model_dir / "runtime.json").read_text())
        self.fc = FeatureConfig(**rt["features"])
        self.fb = mel_filterbank(self.fc)
        self.mean, self.std = np.array(rt["scaler"]["mean"]), np.array(rt["scaler"]["std"])
        self.names = rt["targets"]
        self.log_mask = np.array([n.startswith("t60_") for n in self.names])
        self.cfg = rt["realtime"]
        self.n = int(rt["clip_s"] * self.fc.fs)
        self.hop = int(self.cfg["hop_s"] * self.fc.fs)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_dir / f"model_{variant}.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.buf = np.zeros(self.n, dtype=np.float32)
        self.filled = 0
        self.since = 0
        self.state: np.ndarray | None = None
        self.last_latency_ms = 0.0
        self.updates = 0

    def _active(self) -> tuple[bool, float]:
        frames = self.buf[: (self.n // 400) * 400].reshape(-1, 400)
        db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
        rms_db = 10 * np.log10(np.mean(self.buf ** 2) + 1e-12)
        active = np.mean(db > self.cfg["min_rms_dbfs"])
        return (rms_db > self.cfg["min_rms_dbfs"] and active >= self.cfg["min_active_frac"]), rms_db

    def infer(self) -> np.ndarray:
        t = time.perf_counter()
        x = logmel_np(self.buf, self.fc, self.fb)[None]
        z = self.sess.run(None, {self.inp: x})[0][0]
        self.last_latency_ms = 1000 * (time.perf_counter() - t)
        y = z * self.std + self.mean
        y[self.log_mask] = np.exp(y[self.log_mask])
        return y

    def push(self, block: np.ndarray) -> dict | None:
        block = np.asarray(block, dtype=np.float32)
        k = min(len(block), self.n)
        self.buf = np.roll(self.buf, -k)
        self.buf[-k:] = block[-k:]
        self.filled = min(self.n, self.filled + k)
        self.since += len(block)
        if self.filled < self.n or self.since < self.hop:
            return None
        self.since = 0
        ok, level = self._active()
        if ok:
            y = self.infer()
            a = self.cfg["ema"]
            self.state = y if self.state is None else a * self.state + (1 - a) * y
            self.updates += 1
        return {"active": ok, "level_dbfs": level, "estimate": self.state, "latency_ms": self.last_latency_ms}


# ----------------------------------------------------------------- display
def _bar(v, vmax, width=28):
    n = int(np.clip(v / vmax, 0, 1) * width)
    return "█" * n + "·" * (width - n)


def render(est: RoomEstimator, info: dict, truth: np.ndarray | None, src: str):
    from rich.panel import Panel
    from rich.table import Table
    tab = Table(expand=True, show_edge=False, pad_edge=False)
    tab.add_column("quantity", style="bold")
    tab.add_column("estimate", justify="right")
    tab.add_column("", ratio=1)
    if truth is not None:
        tab.add_column("ground truth", justify="right", style="dim")
    y = info["estimate"]
    if y is None:
        tab.add_row("waiting for sound…", "", "")
    else:
        for i, f in enumerate(OCTAVE_BANDS):
            row = [f"T60 {f if f < 1000 else str(f // 1000) + 'k'} Hz", f"{y[i]:.2f} s", _bar(y[i], 2.0)]
            if truth is not None:
                row.append(f"{truth[i]:.2f} s")
            tab.add_row(*row)
        for j, (lab, unit) in enumerate([("DRR", "dB"), ("C50", "dB")]):
            row = [lab, f"{y[6 + j]:+.1f} {unit}", _bar(y[6 + j] + 20, 50)]
            if truth is not None:
                row.append(f"{truth[6 + j]:+.1f} {unit}")
            tab.add_row(*row)
        row = ["Volume", f"{10 ** y[8]:.0f} m³", _bar(y[8] - 1, 2.5)]
        if truth is not None:
            row.append("–")
        tab.add_row(*row)
    status = "[green]listening (updating)" if info["active"] else "[yellow]holding (too quiet / silence)"
    sub = (f"{status}[/]  ·  input {info['level_dbfs']:.0f} dBFS  ·  inference {info['latency_ms']:.1f} ms  ·  "
           f"updates {est.updates}  ·  {src}")
    return Panel(tab, title="Blind room acoustics estimator", subtitle=sub)


# ----------------------------------------------------------------- sources
def _load(path, fs):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(1)
    if sr != fs:
        g = gcd(sr, fs)
        x = signal.resample_poly(x, fs // g, sr // g)
    return x


def file_blocks(path, fs, block, rir=None, realtime=True, min_len=0):
    x = _load(path, fs)
    if len(x) < min_len:  # shorter than the analysis window: loop it
        print(f"note: {Path(path).name} is {len(x) / fs:.1f} s; looping it to {min_len / fs:.0f} s")
        x = np.tile(x, int(np.ceil(min_len / len(x))))[:min_len]
    if rir is not None:
        x = signal.fftconvolve(x, rir)[: len(x)]
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
    for i in range(0, len(x) - block, block):
        if realtime:
            time.sleep(block / fs)
        yield x[i:i + block]


def mic_blocks(fs, block, device=None):
    import sounddevice as sd
    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=fs, channels=1, blocksize=block, device=device, dtype="float32",
                        callback=lambda indata, *_: q.put(indata[:, 0].copy())):
        while True:
            yield q.get()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="export dir with model_*.onnx + runtime.json")
    ap.add_argument("--variant", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--wav", help="stream this file instead of the microphone")
    ap.add_argument("--rir", help="convolve --wav with this RIR (shows ground truth)")
    ap.add_argument("--fast", action="store_true", help="process the file as fast as possible")
    ap.add_argument("--device", help="sounddevice input device id/name")
    ap.add_argument("--log", help="CSV log of every estimate")
    ap.add_argument("--no-ui", action="store_true")
    args = ap.parse_args()

    est = RoomEstimator(args.model, args.variant)
    fs, block = est.fc.fs, int(0.1 * est.fc.fs)
    truth, rir = None, None
    if args.rir:
        rir = _load(args.rir, fs)
        truth = compute_labels(rir, fs, noise_comp=True)
    blocks = (file_blocks(args.wav, fs, block, rir, realtime=not args.fast, min_len=est.n + 4 * est.hop)
              if args.wav
              else mic_blocks(fs, block, args.device))
    src = Path(args.wav).name if args.wav else "microphone"
    if args.rir:
        src += f" ⊛ {Path(args.rir).stem}"

    writer = None
    if args.log:
        f = open(args.log, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["t", "active", "level_dbfs", "latency_ms", *est.names])
    t0 = time.time()

    def handle(info):
        if writer and info["estimate"] is not None:
            writer.writerow([round(time.time() - t0, 2), int(info["active"]), round(info["level_dbfs"], 1),
                             round(info["latency_ms"], 2), *np.round(info["estimate"], 4)])

    last = None
    try:
        if args.no_ui:
            for b in blocks:
                info = est.push(b)
                if info:
                    handle(info)
                    last = info
        else:
            from rich.live import Live
            with Live(refresh_per_second=8) as live:
                for b in blocks:
                    info = est.push(b)
                    if info:
                        handle(info)
                        last = info
                        live.update(render(est, info, truth, src))
    except KeyboardInterrupt:
        pass
    if last is not None and last["estimate"] is not None:
        y = last["estimate"]
        print("final estimate:", {n: round(float(v), 3) for n, v in zip(est.names, y)})
        if truth is not None:
            print("ground truth:  ", {n: round(float(v), 3) for n, v in zip(est.names[:8], truth[:8])})
    else:
        print("no estimate produced: the input never had enough active sound "
              f"({est.n / fs:.0f} s window, gate {est.cfg['min_rms_dbfs']} dBFS)")


if __name__ == "__main__":
    main()
