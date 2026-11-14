"""Synthetic RIR dataset generation with pyroomacoustics (hybrid ISM + ray tracing).

Each room gets random geometry, per-surface frequency-dependent absorption and
scattering, one source and several microphone positions. Labels are computed
from the *simulated RIR itself* (not from the sampling parameters), so the
ground truth is consistent with how measured RIRs are analysed.

Usage:
    python -m roomest.simulate --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm

from .acoustics import OCTAVE_BANDS, TARGET_NAMES, compute_labels, find_onset
from .config import load_config, resolve

SURFACES = ("east", "west", "north", "south", "floor", "ceiling")
_BANDS = np.array(OCTAVE_BANDS, dtype=float)


def sample_geometry(rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.1:  # occasional large spaces (lecture halls, atria)
        L, W, H = rng.uniform(12, 25), rng.uniform(8, 18), rng.uniform(4, 8)
    else:  # domestic / office rooms
        L, W, H = rng.uniform(3, 12), rng.uniform(2.5, 9), rng.uniform(2.4, 4.0)
    return np.array([max(L, W), min(L, W), H])


def sample_absorption(rng: np.random.Generator) -> np.ndarray:
    """Per-band energy absorption for one surface: log-uniform base with spectral tilt
    (porous absorbers rise with frequency, panel absorbers fall)."""
    base = np.exp(rng.uniform(np.log(0.02), np.log(0.6)))
    tilt = rng.uniform(-0.4, 0.8)
    coeffs = base * (_BANDS / 1000.0) ** tilt * rng.uniform(0.85, 1.15, size=len(_BANDS))
    return np.clip(coeffs, 0.01, 0.95)


def _surface_areas(d: np.ndarray) -> np.ndarray:
    L, W, H = d
    return np.array([W * H, W * H, L * H, L * H, L * W, L * W])  # order matches SURFACES


def _random_point(rng, dims, margin, z_range=None):
    p = rng.uniform(margin, dims - margin)
    if z_range is not None:
        p[2] = rng.uniform(*z_range)
    return p


def simulate_room(room_id: int, cfg: dict) -> list[dict]:
    rng = np.random.default_rng([cfg["seed"], room_id])
    sim, fs = cfg["sim"], cfg["fs"]
    # Draw a target mid-band T60 log-uniformly, then scale the random absorption
    # spectra to hit it (Sabine). This balances the label distribution instead of
    # leaving it to the product of geometry and material priors.
    lo, hi = sim["t60_range"]
    target = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    for _ in range(200):  # keep the target fixed; resample geometry/materials until feasible
        dims = sample_geometry(rng)
        alpha = np.stack([sample_absorption(rng) for _ in SURFACES])  # (6, bands)
        A = (_surface_areas(dims)[:, None] * alpha).sum(0)
        alpha = alpha * (0.161 * np.prod(dims) / target) / A[2:4].mean()  # 500 Hz/1 kHz
        if alpha.max() <= 0.95 and alpha.min() >= 0.005:
            break
    alpha = np.clip(alpha, 0.005, 0.95)
    t60_sab = 0.161 * np.prod(dims) / (_surface_areas(dims)[:, None] * alpha).sum(0)
    scattering = float(rng.uniform(0.05, 0.6))
    materials = {s: pra.Material({"coeffs": alpha[i].tolist(), "center_freqs": list(OCTAVE_BANDS)},
                                 scattering) for i, s in enumerate(SURFACES)}
    room = pra.ShoeBox(dims, fs=fs, materials=materials, max_order=sim["ism_order"],
                       ray_tracing=True, air_absorption=True)
    room.set_ray_tracing(receiver_radius=0.5, n_rays=sim["n_rays"], energy_thres=1e-7)

    margin = 0.4
    src = _random_point(rng, dims, margin, (1.0, min(1.9, dims[2] - margin)))
    # Source-mic distance ~ log-uniform [0.7, 6] m (talker-to-device / device-to-listener),
    # realised by rejection sampling of mic positions.
    mics = []
    d_lo, d_hi = sim["distance_range"]
    for _ in range(2000):
        if len(mics) == sim["mics_per_room"]:
            break
        want = np.exp(rng.uniform(np.log(d_lo), np.log(d_hi)))
        for _ in range(200):
            m = _random_point(rng, dims, margin, (0.6, min(1.8, dims[2] - margin)))
            if abs(np.linalg.norm(m - src) - want) < 0.15 * want:
                mics.append(m)
                break
    while len(mics) < sim["mics_per_room"]:  # tiny rooms: accept any valid position
        m = _random_point(rng, dims, margin, (0.6, min(1.8, dims[2] - margin)))
        if np.linalg.norm(m - src) >= d_lo:
            mics.append(m)
    room.add_source(src)
    room.add_microphone(np.array(mics).T)
    room.compute_rir()

    max_len = int(sim["max_rir_s"] * fs)
    out = []
    for k, m in enumerate(mics):
        h = np.asarray(room.rir[k][0], dtype=np.float64)
        labels = compute_labels(h, fs, volume=float(np.prod(dims)))
        start = max(0, find_onset(h) - int(0.002 * fs))  # keep 2 ms pre-onset
        h = h[start:start + max_len]
        h = h / (np.max(np.abs(h)) + 1e-12)
        buf = np.zeros(max_len, dtype=np.float16)
        buf[:len(h)] = h
        out.append(dict(rir=buf, labels=labels, room_id=room_id, dims=dims.astype(np.float32),
                        distance=np.float32(np.linalg.norm(m - src)),
                        mean_alpha=alpha.mean(0).astype(np.float32),
                        sabine=t60_sab.astype(np.float32)))
    return out


def _worker(args):
    room_id, cfg = args
    try:
        return simulate_room(room_id, cfg)
    except Exception as e:  # pragma: no cover - a failed room should not kill the run
        print(f"room {room_id} failed: {e}")
        return []


def assign_splits(room_ids: np.ndarray, fractions: dict, seed: int) -> np.ndarray:
    """Split by ROOM so the same room never appears in train and test."""
    uniq = np.unique(room_ids)
    perm = np.random.default_rng(seed).permutation(uniq)
    n_tr = int(round(fractions["train"] * len(uniq)))
    n_va = int(round(fractions["val"] * len(uniq)))
    split_of = {r: 0 for r in perm[:n_tr]}
    split_of.update({r: 1 for r in perm[n_tr:n_tr + n_va]})
    split_of.update({r: 2 for r in perm[n_tr + n_va:]})
    return np.array([split_of[r] for r in room_ids], dtype=np.int8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    sim = cfg["sim"]
    out_dir = resolve(cfg["paths"]["sim_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    jobs = [(i, cfg) for i in range(sim["n_rooms"])]
    records = []
    with mp.get_context("spawn").Pool(sim["workers"]) as pool:
        for rec in tqdm(pool.imap_unordered(_worker, jobs, chunksize=4), total=len(jobs), desc="rooms"):
            records.extend(rec)
    records.sort(key=lambda r: r["room_id"])

    room_ids = np.array([r["room_id"] for r in records])
    split = assign_splits(room_ids, sim["split"], cfg["seed"])
    np.save(out_dir / "rirs.npy", np.stack([r["rir"] for r in records]))
    labels = np.stack([r["labels"] for r in records])
    np.save(out_dir / "labels.npy", labels)
    np.savez(out_dir / "meta.npz", room_id=room_ids, split=split,
             dims=np.stack([r["dims"] for r in records]),
             distance=np.array([r["distance"] for r in records]),
             mean_alpha=np.stack([r["mean_alpha"] for r in records]),
             sabine=np.stack([r["sabine"] for r in records]))

    summary = {
        "n_rirs": len(records), "n_rooms": int(len(np.unique(room_ids))), "fs": cfg["fs"],
        "split_counts": {n: int((split == i).sum()) for i, n in enumerate(["train", "val", "test"])},
        "label_stats": {n: {"min": float(np.nanmin(labels[:, j])), "median": float(np.nanmedian(labels[:, j])),
                            "max": float(np.nanmax(labels[:, j])), "nan_frac": float(np.isnan(labels[:, j]).mean())}
                        for j, n in enumerate(TARGET_NAMES)},
        "elapsed_s": round(time.time() - t0, 1), "config": sim,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_rirs", "n_rooms", "split_counts", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
