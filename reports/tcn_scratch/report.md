# Evaluation: `tcn_scratch` (tcn, epoch 21)

## Held-out simulated rooms (n=1600, unseen rooms and unseen speakers)

| target | unit | model MAE | mean predictor MAE | model r | model within JND |
|---|---|---|---|---|---|
| t60_125 | s | 0.207 | 0.538 | 0.93 | 16% |
| t60_250 | s | 0.148 | 0.485 | 0.95 | 20% |
| t60_500 | s | 0.100 | 0.427 | 0.95 | 27% |
| t60_1000 | s | 0.083 | 0.378 | 0.95 | 33% |
| t60_2000 | s | 0.088 | 0.328 | 0.93 | 28% |
| t60_4000 | s | 0.090 | 0.264 | 0.91 | 22% |
| drr | dB | 2.796 | 5.512 | 0.86 | 43% |
| c50 | dB | 1.287 | 5.368 | 0.97 | 52% |
| log10_volume | log10 m³ | 0.217 | 0.240 | 0.45 | 76% |

![](sim_scatter.png)

![](sim_t60_by_band.png)

![](sim_error_vs_t60.png)

## Measured RIRs: sim-to-real (193 RIRs x 4 renders)

Volume is unknown for these rooms and is excluded. Outdoor spaces are excluded (regex `(?i)outside|outdoor|street|field|campground|parking|skatepark|bridge|backyard|balcony|forest|beach|woods`).

| target | unit | model MAE | mean predictor MAE | model r | model within JND |
|---|---|---|---|---|---|
| t60_125 | s | 0.291 | 0.413 | 0.62 | 8% |
| t60_250 | s | 0.178 | 0.364 | 0.73 | 15% |
| t60_500 | s | 0.129 | 0.335 | 0.76 | 20% |
| t60_1000 | s | 0.134 | 0.288 | 0.84 | 18% |
| t60_2000 | s | 0.160 | 0.251 | 0.75 | 12% |
| t60_4000 | s | 0.135 | 0.175 | 0.71 | 9% |
| drr | dB | 4.203 | 11.384 | 0.74 | 31% |
| c50 | dB | 3.001 | 10.112 | 0.88 | 21% |

![](real_scatter.png)

![](real_t60_by_band.png)
