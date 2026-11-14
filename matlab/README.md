# MATLAB measurement & analysis

Mirror of the Python measurement/analysis code (`roomest/measure.py`, `roomest/acoustics.py`)
so that ground truth can be produced and cross-checked in either environment.

| file | purpose |
|---|---|
| `generate_ess.m` | exponential sine sweep + inverse filter (Farina 2000) |
| `measure_rir.m` | play/record N sweeps, deconvolve, align, average → RIR `.wav` |
| `analyze_rir.m` | T20 / T30 / T60 / EDT per octave band, DRR, C50, Schroeder plot |
| `simulate_check.m` | self-test: recovers a known T60 / DRR from a synthetic RIR |

```matlab
>> simulate_check                                   % sanity check, prints OK
>> measure_rir('livingroom_pos1.wav', 'repeats', 3)  % speaker ~1-3 m from mic
>> p = analyze_rir('livingroom_pos1.wav');
```

Cross-check against Python (results should agree to within a few ms / 0.1 dB):

```bash
python -m roomest.measure analyze livingroom_pos1.wav
```

`measure_rir` uses `audioPlayerRecorder` (Audio Toolbox) for sample-synchronous I/O when
available, otherwise `audioplayer`/`audiorecorder`; the resulting arbitrary latency is harmless
because analysis aligns on the direct-path onset. Requires the Signal Processing Toolbox
(`butter`, `filtfilt`).
