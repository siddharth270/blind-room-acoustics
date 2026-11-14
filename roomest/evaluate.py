"""Evaluate a checkpoint on held-out simulated rooms and on measured RIRs.

    python -m roomest.evaluate --ckpt runs/crnn_teacher/best.pt --baselines
    python -m roomest.evaluate --ckpt runs/tcn_student/best.pt --real

Reports (markdown + JSON + plots) are written to <reports_dir>/<run>/.
Compared against:
  * mean predictor     - predicts the training-set mean (the floor any model must beat)
  * classical blind    - free-decay-region T60 estimator with train-set calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .acoustics import OCTAVE_BANDS, T60_NAMES, TARGET_NAMES  # noqa: E402
from .baselines import CalibratedBlindT60  # noqa: E402
from .common import build_sources, frontend, load_checkpoint, make_dataset, pick_device, sim_store  # noqa: E402
from .config import resolve  # noqa: E402
from .data import load_real_rirs  # noqa: E402

UNITS = {**{n: "s" for n in T60_NAMES}, "drr": "dB", "c50": "dB", "log10_volume": "log10 m³"}
# Just-noticeable differences (ISO 3382-1 Annex A for T / C50; Larsen et al. 2008 for DRR)
JND = {**{n: ("rel", 0.05) for n in T60_NAMES}, "drr": ("abs", 2.0), "c50": ("abs", 1.0),
       "log10_volume": ("abs", np.log10(2.0))}  # volume: within a factor of 2

# validated categorical palette (light mode), see dataviz skill
C_MODEL, C_BASE, C_MEAN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------- metrics
def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    out = {}
    for j, name in enumerate(TARGET_NAMES):
        p, t = pred[:, j], true[:, j]
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() < 3:
            continue
        p, t = p[ok], t[ok]
        err = p - t
        kind, thr = JND[name]
        within = np.abs(err) / t <= thr if kind == "rel" else np.abs(err) <= thr
        m = {"n": int(ok.sum()), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
             "bias": float(np.mean(err)), "pearson": float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 else 0.0,
             "within_jnd": float(np.mean(within))}
        if name in T60_NAMES:
            m["median_rel_err"] = float(np.median(np.abs(err) / t))
        out[name] = m
    return out


def summarize(m: dict) -> dict:
    t60 = [m[n] for n in T60_NAMES if n in m]
    return {
        "t60_mae_ms": 1000 * float(np.mean([x["mae"] for x in t60])),
        "t60_median_rel_err": float(np.mean([x["median_rel_err"] for x in t60])),
        "t60_pearson": float(np.mean([x["pearson"] for x in t60])),
        **{f"{k}_mae": m[k]["mae"] for k in ("drr", "c50", "log10_volume") if k in m},
    }


# ----------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, scaler, ds, cfg, device, batch=64, keep_wavs=False):
    feat = frontend(cfg, device)
    dl = DataLoader(ds, batch_size=batch, num_workers=4, shuffle=False)
    preds, trues, wavs, idx = [], [], [], []
    for wav, tgt, j in dl:
        preds.append(model(feat(wav.to(device))).cpu().numpy())
        trues.append(tgt.numpy())
        idx.append(j.numpy())
        if keep_wavs:
            wavs.extend(wav.numpy())
    return scaler.inverse(np.concatenate(preds)), scaler.inverse(np.concatenate(trues)), np.concatenate(idx), wavs


# ----------------------------------------------------------------- plots
def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def scatter_grid(pred, true, path, title):
    names = ["t60_500", "t60_1000", "t60_4000", "drr", "c50", "log10_volume"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, name in zip(axes.flat, names):
        j = TARGET_NAMES.index(name)
        p, t = pred[:, j], true[:, j]
        ok = np.isfinite(p) & np.isfinite(t)
        _style(ax)
        if not ok.any():
            ax.text(0.5, 0.5, "no ground truth", ha="center", va="center", color=INK2, transform=ax.transAxes)
            ax.set_title(name, color=INK, fontsize=10)
            continue
        lo, hi = np.percentile(np.r_[p[ok], t[ok]], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", zorder=1)
        ax.scatter(t[ok], p[ok], s=9, color=C_MODEL, alpha=0.45, edgecolors="none", zorder=2)
        r = np.corrcoef(p[ok], t[ok])[0, 1]
        ax.set_title(f"{name}   MAE {np.mean(np.abs(p[ok] - t[ok])):.3g} {UNITS[name]}   r={r:.2f}",
                     color=INK, fontsize=9.5)
        ax.set_xlabel(f"true ({UNITS[name]})", color=INK2, fontsize=8.5)
        ax.set_ylabel(f"predicted ({UNITS[name]})", color=INK2, fontsize=8.5)
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def band_bars(results: dict[str, dict], path, title):
    colors = {"model": C_MODEL, "classical blind": C_BASE, "mean predictor": C_MEAN}
    fig, ax = plt.subplots(figsize=(8, 4))
    _style(ax)
    k = len(results)
    w = 0.8 / k
    x = np.arange(len(OCTAVE_BANDS))
    for i, (label, m) in enumerate(results.items()):
        vals = [1000 * m[n]["mae"] if n in m else np.nan for n in T60_NAMES]
        ax.bar(x + (i - (k - 1) / 2) * w, vals, w * 0.92, color=colors.get(label, INK2), label=label)
    ax.set_xticks(x, [f"{f} Hz" if f < 1000 else f"{f // 1000} kHz" for f in OCTAVE_BANDS])
    ax.set_ylabel("T60 MAE (ms)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_vs_t60(pred, true, path):
    j = TARGET_NAMES.index("t60_1000")
    t, rel = true[:, j], np.abs(pred[:, j] - true[:, j]) / true[:, j]
    edges = np.array([0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0])
    centers, med, q25, q75 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t < b)
        if sel.sum() >= 5:
            centers.append(f"{a:g}–{b:g}")
            med.append(100 * np.median(rel[sel]))
            q25.append(100 * np.percentile(rel[sel], 25))
            q75.append(100 * np.percentile(rel[sel], 75))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _style(ax)
    xs = np.arange(len(centers))
    ax.fill_between(xs, q25, q75, color=C_MODEL, alpha=0.18, lw=0)
    ax.plot(xs, med, color=C_MODEL, lw=2, marker="o", ms=5)
    ax.axhline(5, color=INK2, lw=1, ls="--")
    ax.text(xs[-1], 5, " 5% JND", color=INK2, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(xs, centers)
    ax.set_xlabel("true T60 @ 1 kHz (s)", color=INK2, fontsize=9)
    ax.set_ylabel("relative error (%)", color=INK2, fontsize=9)
    ax.set_title("T60 @ 1 kHz relative error by reverberation range (median, IQR band)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- report
def md_table(results: dict[str, dict]) -> str:
    rows = ["| target | unit | " + " | ".join(f"{k} MAE" for k in results) + " | model r | model within JND |",
            "|---|---|" + "---|" * len(results) + "---|---|"]
    model = next(iter(results.values()))
    for name in TARGET_NAMES:
        if name not in model:
            continue
        cells = [f"{results[k][name]['mae']:.3f}" if name in results[k] else "–" for k in results]
        rows.append(f"| {name} | {UNITS[name]} | " + " | ".join(cells) +
                    f" | {model[name]['pearson']:.2f} | {100 * model[name]['within_jnd']:.0f}% |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real", action="store_true", help="also evaluate on measured RIRs (paths.real_rir_dir)")
    ap.add_argument("--baselines", action="store_true", help="also run the classical blind T60 baseline")
    ap.add_argument("--renders-per-real-rir", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, scaler, cfg, ck = load_checkpoint(args.ckpt, device)
    run = Path(args.ckpt).parent.name
    out = resolve(cfg["paths"]["reports_dir"]) / run
    out.mkdir(parents=True, exist_ok=True)
    store = sim_store(cfg)
    sources = build_sources(cfg)
    train_mean = scaler.inverse(np.zeros((1, len(TARGET_NAMES))))[0]
    report = {"checkpoint": str(args.ckpt), "epoch": ck["epoch"], "model": ck["model_name"]}
    md = [f"# Evaluation: `{run}` ({ck['model_name']}, epoch {ck['epoch']})\n"]

    # ---------------- simulated held-out rooms
    ds = make_dataset(cfg, store, "test", sources, scaler)
    pred, true, _, wavs = predict(model, scaler, ds, cfg, device, keep_wavs=args.baselines)
    res = {"model": metrics(pred, true), "mean predictor": metrics(np.tile(train_mean, (len(true), 1)), true)}
    if args.baselines:
        print("fitting classical baseline on training renders ...")
        fit_ds = make_dataset(cfg, store, "train", sources, scaler, length=300, seed_offset=99)
        fit_wavs = [fit_ds[i][0].numpy() for i in range(len(fit_ds))]
        fit_true = scaler.inverse(np.stack([fit_ds[i][1].numpy() for i in range(len(fit_ds))]))
        blind = CalibratedBlindT60(cfg["fs"]).fit(fit_wavs, fit_true[:, :len(T60_NAMES)])
        bp = np.full_like(pred, np.nan)
        bp[:, :len(T60_NAMES)] = np.stack([blind.predict(w) for w in wavs])
        res = {"model": res["model"], "classical blind": metrics(bp, true), "mean predictor": res["mean predictor"]}
    report["sim_test"] = {k: {"summary": summarize(v), "per_target": v} for k, v in res.items()}
    scatter_grid(pred, true, out / "sim_scatter.png", f"{run}: held-out simulated rooms (n={len(true)})")
    band_bars(res, out / "sim_t60_by_band.png", "T60 error per octave band – held-out simulated rooms")
    error_vs_t60(pred, true, out / "sim_error_vs_t60.png")
    md += [f"## Held-out simulated rooms (n={len(true)}, unseen rooms and unseen speakers)\n", md_table(res), "",
           "![](sim_scatter.png)\n", "![](sim_t60_by_band.png)\n", "![](sim_error_vs_t60.png)\n"]
    print(md_table(res))

    # ---------------- measured RIRs (sim-to-real)
    if args.real:
        real = load_real_rirs(resolve(cfg["paths"]["real_rir_dir"]), cfg["fs"], cfg["sim"]["max_rir_s"],
                              exclude_regex=cfg["paths"].get("real_rir_exclude"))
        n = len(real) * args.renders_per_real_rir
        rds = make_dataset(cfg, real, "test", sources, scaler, length=n, seed_offset=1)
        rp, rt, ridx, _ = predict(model, scaler, rds, cfg, device)
        rres = {"model": metrics(rp, rt), "mean predictor": metrics(np.tile(train_mean, (len(rt), 1)), rt)}
        report["real"] = {"n_rirs": len(real), "n_renders": n,
                          **{k: {"summary": summarize(v), "per_target": v} for k, v in rres.items()}}
        # per-RIR predictions (averaged over renders) for error analysis
        per = {}
        for i in range(len(real)):
            sel = ridx == i
            per[real.names[i]] = {"true": real.labels[i][:8].round(3).tolist(),
                                  "pred_mean": np.nanmean(rp[sel], 0)[:8].round(3).tolist()}
        (out / "real_per_rir.json").write_text(json.dumps(per, indent=1))
        scatter_grid(rp, rt, out / "real_scatter.png", f"{run}: measured RIRs – MIT IR Survey (n={n})")
        band_bars(rres, out / "real_t60_by_band.png", "T60 error per octave band – measured RIRs")
        md += [f"## Measured RIRs: sim-to-real ({len(real)} RIRs x {args.renders_per_real_rir} renders)\n",
               "Volume is unknown for these rooms and is excluded. Outdoor spaces are excluded "
               f"(regex `{cfg['paths'].get('real_rir_exclude')}`).\n", md_table(rres), "",
               "![](real_scatter.png)\n", "![](real_t60_by_band.png)\n"]
        print(md_table(rres))

    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text("\n".join(md))
    print(f"report -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
