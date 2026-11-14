"""Export a checkpoint for deployment: ONNX fp32 -> int8 (static, QDQ, per-channel),
then measure what compression costs.

    python -m roomest.export --ckpt runs/tcn_student/best.pt

Outputs (in <export_dir>/<run>/):
    model_fp32.onnx, model_int8.onnx
    runtime.json   - feature config + target scaler; everything the numpy runtime needs
    export_report.{md,json} - params, MACs, file size, latency, accuracy fp32 vs int8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .acoustics import TARGET_NAMES
from .common import build_sources, frontend, load_checkpoint, make_dataset, sim_store
from .config import resolve
from .evaluate import metrics, summarize
from .features import FeatureConfig
from .models import count_macs, count_params


def features_for(ds, cfg, n=None):
    """Render clips in parallel workers, then compute log-mel features."""
    from torch.utils.data import DataLoader, Subset
    feat = frontend(cfg, "cpu")
    ds = Subset(ds, range(min(n or len(ds), len(ds))))
    xs, ys = [], []
    for wav, tgt, _ in DataLoader(ds, batch_size=32, num_workers=cfg["train"]["num_workers"]):
        xs.append(feat(wav).numpy())
        ys.append(tgt.numpy())
    return np.concatenate(xs), np.concatenate(ys)


class _Calib(CalibrationDataReader):
    def __init__(self, x, name):
        self.it = iter([{name: x[i:i + 1]} for i in range(len(x))])

    def get_next(self):
        return next(self.it, None)


def ort_session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_ort(sess, x):
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])


def latency_ms(sess, x, n=200):
    name = sess.get_inputs()[0].name
    for _ in range(10):
        sess.run(None, {name: x[:1]})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {name: x[:1]})
        ts.append(1000 * (time.perf_counter() - t))
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-calib", type=int, help="calibration clips (default: export.n_calib)")
    ap.add_argument("--n-eval", type=int, help="evaluation clips (default: export.n_eval)")
    args = ap.parse_args()

    model, scaler, cfg, ck = load_checkpoint(args.ckpt, "cpu")
    args.n_calib = args.n_calib or cfg.get("export", {}).get("n_calib", 200)
    args.n_eval = args.n_eval or cfg.get("export", {}).get("n_eval", 800)
    run = Path(args.ckpt).parent.name
    out = resolve(cfg["paths"]["export_dir"]) / run
    out.mkdir(parents=True, exist_ok=True)
    fc = FeatureConfig.from_cfg(cfg)
    T = fc.n_frames(int(cfg["data"]["clip_s"] * cfg["fs"]))
    dummy = torch.zeros(1, fc.n_mels, T)

    # ---- ONNX fp32 (fixed input shape: what an embedded target would use)
    fp32, pre, int8 = out / "model_fp32.onnx", out / "model_pre.onnx", out / "model_int8.onnx"
    torch.onnx.export(model, (dummy,), str(fp32), input_names=["logmel"], output_names=["targets_z"],
                      opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)

    # ---- data: calibration from TRAIN split, evaluation on TEST split
    store, sources = sim_store(cfg), build_sources(cfg)
    calib_x, _ = features_for(make_dataset(cfg, store, "train", sources, scaler, length=args.n_calib,
                                           seed_offset=42), cfg)
    test_x, test_z = features_for(make_dataset(cfg, store, "test", sources, scaler), cfg, args.n_eval)

    # GRUs are not statically quantisable in ORT; their MatMuls stay fp32 (reported below).
    quantize_static(str(pre), str(int8), _Calib(calib_x, "logmel"), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    pre.unlink(missing_ok=True)

    # ---- accuracy
    with torch.no_grad():
        p_torch = model(torch.from_numpy(test_x)).numpy()
    s32, s8 = ort_session(fp32), ort_session(int8)
    p32, p8 = run_ort(s32, test_x), run_ort(s8, test_x)
    true = scaler.inverse(test_z)
    res = {name: metrics(scaler.inverse(p), true) for name, p in
           [("torch_fp32", p_torch), ("onnx_fp32", p32), ("onnx_int8", p8)]}
    parity = float(np.max(np.abs(p_torch - p32)))

    lat32, lat8 = latency_ms(s32, test_x), latency_ms(s8, test_x)
    macs = count_macs(model, dummy)
    report = {
        "run": run, "model": ck["model_name"], "params": count_params(model), "macs": macs,
        "input_shape": [1, fc.n_mels, T], "torch_vs_onnx_max_abs_diff": parity,
        "size_kb": {"fp32": fp32.stat().st_size / 1024, "int8": int8.stat().st_size / 1024},
        "latency_ms_1thread": {"fp32": {"p50": lat32[0], "p95": lat32[1]}, "int8": {"p50": lat8[0], "p95": lat8[1]}},
        "accuracy": {k: {"summary": summarize(v), "per_target": v} for k, v in res.items()},
        "n_eval": len(test_x),
    }
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    (out / "runtime.json").write_text(json.dumps({
        "features": fc.to_dict(), "scaler": scaler.state_dict(), "targets": TARGET_NAMES,
        "clip_s": cfg["data"]["clip_s"], "input_frames": T, "realtime": cfg["realtime"],
        "model": ck["model_name"], "source_checkpoint": str(args.ckpt)}, indent=2))

    rows = ["| variant | size (KB) | latency p50 / p95 (ms) | T60 MAE (ms) | T60 med. rel. err | DRR MAE (dB) | C50 MAE (dB) | log10 V MAE |",
            "|---|---|---|---|---|---|---|---|"]
    for k, size, lat in [("onnx_fp32", report["size_kb"]["fp32"], lat32), ("onnx_int8", report["size_kb"]["int8"], lat8)]:
        s = report["accuracy"][k]["summary"]
        rows.append(f"| {k} | {size:.0f} | {lat[0]:.2f} / {lat[1]:.2f} | {s['t60_mae_ms']:.0f} | "
                    f"{100 * s['t60_median_rel_err']:.1f}% | {s['drr_mae']:.2f} | {s['c50_mae']:.2f} | {s['log10_volume_mae']:.3f} |")
    md = [f"# Export: `{run}` ({ck['model_name']})\n",
          f"* parameters: **{report['params']:,}**  ·  MACs per 4 s inference: **{macs / 1e6:.1f} M**",
          f"* input: log-mel {report['input_shape']} (4 s @ 16 kHz)",
          f"* torch vs ONNX fp32 max |Δ| (normalised units): {parity:.2e}",
          f"* latency: onnxruntime CPU, 1 thread, batch 1 (host machine; not an embedded target)\n",
          "\n".join(rows), "",
          f"Evaluated on {len(test_x)} held-out simulated test renders; int8 calibrated on {args.n_calib} train renders."]
    (out / "export_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
