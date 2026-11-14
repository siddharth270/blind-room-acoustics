"""Train a blind room-acoustics estimator (optionally distilled from a teacher).

    python -m roomest.train --config configs/default.yaml --name crnn_teacher
    python -m roomest.train --config configs/default.yaml --name tcn_student \
        train.model=tcn train.teacher=runs/crnn_teacher/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .acoustics import N_TARGETS, TARGET_NAMES
from .common import (build_sources, frontend, load_checkpoint, make_dataset, pick_device, sim_store)
from .config import load_config, resolve
from .data import TargetScaler, worker_init
from .models import build_model, count_params, masked_loss


def physical_mae(pred_z, tgt_z, scaler: TargetScaler) -> np.ndarray:
    p, t = scaler.inverse(pred_z), scaler.inverse(tgt_z)
    return np.nanmean(np.abs(p - t), axis=0)


def run_epoch(model, loader, feat, device, weights, opt=None, teacher=None, alpha=1.0, clip=5.0):
    train = opt is not None
    model.train(train)
    tot, n, preds, tgts = 0.0, 0, [], []
    for wav, tgt, _ in loader:
        wav, tgt = wav.to(device, non_blocking=True), tgt.to(device)
        x = feat(wav)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = masked_loss(out, tgt, weights)
            if teacher is not None and train:
                with torch.no_grad():
                    soft = teacher(x)
                loss = alpha * loss + (1 - alpha) * masked_loss(out, soft, weights)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        tot += loss.item() * len(wav)
        n += len(wav)
        if not train:
            preds.append(out.cpu().numpy()), tgts.append(tgt.cpu().numpy())
    return tot / n, (np.concatenate(preds), np.concatenate(tgts)) if preds else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", required=True, help="run name (output: <runs_dir>/<name>/)")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    tc = cfg["train"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = pick_device(tc["device"])
    out_dir = resolve(cfg["paths"]["runs_dir"]) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    store = sim_store(cfg)
    scaler = TargetScaler().fit(store.labels[store.indices("train")])
    sources = build_sources(cfg)
    print(f"device={device} | RIRs train/val/test = "
          f"{[len(store.indices(s)) for s in ('train', 'val', 'test')]} | dry sources = "
          f"{ {k: (len(v) if v else 0) for k, v in sources.items()} }")

    dl = dict(batch_size=tc["batch_size"], num_workers=tc["num_workers"], worker_init_fn=worker_init,
              persistent_workers=tc["num_workers"] > 0, pin_memory=False)
    train_dl = DataLoader(make_dataset(cfg, store, "train", sources, scaler, train=True), shuffle=False, **dl)
    val_dl = DataLoader(make_dataset(cfg, store, "val", sources, scaler), shuffle=False, **dl)

    model = build_model(tc["model"], cfg, N_TARGETS).to(device)
    teacher, alpha = None, 1.0
    if tc.get("teacher"):
        teacher, _, _, _ = load_checkpoint(tc["teacher"], device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        alpha = tc["distill_alpha"]
        print(f"distilling from {tc['teacher']} (alpha={alpha})")
    print(f"model={tc['model']} params={count_params(model):,}")

    weights = torch.tensor(tc["target_weights"], dtype=torch.float32, device=device)
    feat = frontend(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=tc["epochs"] * len(train_dl),
                                                pct_start=0.1)

    class _SchedOpt:  # step the scheduler every batch
        def zero_grad(self, **kw): opt.zero_grad(**kw)
        def step(self): opt.step(); sched.step()

    best, history, start = float("inf"), [], 1
    if args.resume and (out_dir / "last.pt").exists():
        st = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        if "optimizer" not in st:
            raise SystemExit("last.pt has no optimizer state (written by an older version); retrain instead")
        if st["config"]["train"]["epochs"] != tc["epochs"]:
            raise SystemExit("--resume needs the same train.epochs as the original run (LR schedule)")
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        best, history, start = st["best"], st["history"], st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']} (best val {best:.4f})")
    log_path = out_dir / "history.csv"
    for ep in range(start, tc["epochs"] + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, train_dl, feat, device, weights, _SchedOpt(), teacher, alpha, tc["grad_clip"])
        va_loss, (pz, tz) = run_epoch(model, val_dl, feat, device, weights)
        mae = physical_mae(pz, tz, scaler)
        row = {"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "lr": sched.get_last_lr()[0],
               "time_s": time.time() - t0, **{f"val_mae_{n}": float(m) for n, m in zip(TARGET_NAMES, mae)}}
        history.append(row)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        ck = {"state_dict": model.state_dict(), "model_name": tc["model"], "config": cfg,
              "scaler": scaler.state_dict(), "epoch": ep, "val_loss": va_loss}
        flag = ""
        if va_loss < best:
            best = va_loss
            torch.save(ck, out_dir / "best.pt")
            flag = " *"
        torch.save({**ck, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "history": history, "best": best}, out_dir / "last.pt")
        print(f"ep {ep:3d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
              f"T60@1k MAE {mae[3] * 1000:.0f} ms | DRR {mae[6]:.2f} dB | C50 {mae[7]:.2f} dB | "
              f"{row['time_s']:.0f}s{flag}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps({"best_val_loss": best, "epochs": tc["epochs"],
                                                      "params": count_params(model)}, indent=2))


if __name__ == "__main__":
    main()
