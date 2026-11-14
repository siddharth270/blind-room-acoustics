"""Helpers shared by train / evaluate / export."""
from __future__ import annotations

from pathlib import Path

import torch

from .acoustics import N_TARGETS
from .config import resolve
from .data import ReverbDataset, RirStore, SourceBank, TargetScaler, discover_sources
from .features import FeatureConfig, LogMel
from .models import build_model


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sources(cfg: dict) -> dict[str, SourceBank | None]:
    files = discover_sources(cfg["paths"]["source_dirs"], cfg["data"]["source_split"], cfg["seed"])
    return {k: (SourceBank(v, cfg["fs"]) if v else None) for k, v in files.items()}


def make_dataset(cfg, store, split, sources, scaler, train=False, length=None, seed_offset=0):
    n = length or cfg["data"][{"train": "train_samples_per_epoch", "val": "val_samples",
                               "test": "test_samples"}[split]]
    return ReverbDataset(store, store.indices(split) if store.sim_dir else range(len(store)),
                         sources[split], cfg, scaler, train=train, length=n,
                         seed=cfg["seed"] + 7919 * (1 + seed_offset) + {"train": 0, "val": 1, "test": 2}[split])


def load_checkpoint(path: str | Path, device="cpu"):
    ck = torch.load(resolve(path), map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model(ck["model_name"], cfg, N_TARGETS)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    scaler = TargetScaler(**ck["scaler"])
    return model, scaler, cfg, ck


def frontend(cfg, device) -> LogMel:
    return LogMel(FeatureConfig.from_cfg(cfg)).to(device)


def sim_store(cfg) -> RirStore:
    return RirStore(resolve(cfg["paths"]["sim_dir"]))
