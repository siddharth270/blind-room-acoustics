# Export: `crnn_teacher` (crnn)

* parameters: **880,426**  ·  MACs per 4 s inference: **1201.8 M**
* input: log-mel [1, 64, 397] (4 s @ 16 kHz)
* torch vs ONNX fp32 max |Δ| (normalised units): 1.91e-06
* latency: onnxruntime CPU, 1 thread, batch 1 (host machine; not an embedded target)

| variant | size (KB) | latency p50 / p95 (ms) | T60 MAE (ms) | T60 med. rel. err | DRR MAE (dB) | C50 MAE (dB) | log10 V MAE |
|---|---|---|---|---|---|---|---|
| onnx_fp32 | 3446 | 28.50 / 31.10 | 91 | 8.8% | 1.92 | 1.01 | 0.216 |
| onnx_int8 | 1485 | 8.01 / 8.59 | 91 | 8.7% | 1.92 | 1.01 | 0.216 |

Evaluated on 800 held-out simulated test renders; int8 calibrated on 200 train renders.