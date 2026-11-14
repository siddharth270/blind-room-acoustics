# Export: `tcn_student` (tcn)

* parameters: **59,970**  ·  MACs per 4 s inference: **7.5 M**
* input: log-mel [1, 64, 397] (4 s @ 16 kHz)
* torch vs ONNX fp32 max |Δ| (normalised units): 1.52e-06
* latency: onnxruntime CPU, 1 thread, batch 1 (host machine; not an embedded target)

| variant | size (KB) | latency p50 / p95 (ms) | T60 MAE (ms) | T60 med. rel. err | DRR MAE (dB) | C50 MAE (dB) | log10 V MAE |
|---|---|---|---|---|---|---|---|
| onnx_fp32 | 240 | 0.36 / 0.37 | 110 | 10.9% | 2.68 | 1.22 | 0.219 |
| onnx_int8 | 120 | 0.26 / 0.27 | 109 | 11.1% | 2.69 | 1.23 | 0.219 |

Evaluated on 800 held-out simulated test renders; int8 calibrated on 200 train renders.