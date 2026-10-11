# Publication Results: ResNet50 TensorRT FP16-Enabled Execution on Jetson Nano

| Item | Result |
|---|---:|
| Images | 261 |
| Accuracy | 0.977011 |
| Macro precision | 0.955108 |
| Macro recall | 0.962434 |
| Macro F1 | 0.958480 |
| Weighted F1 | 0.977156 |
| Engine latency | 187.148 ms |
| End-to-end latency | 273.363 ms |
| Engine FPS | 5.343 |
| Transfer-inclusive FPS | 5.265 |
| Wall-clock pipeline FPS | 3.653 |
| Average Power TOT | 7.599 W |
| Gross energy/image | 2.087 J |
| Idle-corrected energy/image | 1.298 J |
| FP32/FP16 prediction agreement | 100.000000% |
| FP16 engine speedup over FP32 | 1.860x |
| FP16 pipeline speedup over FP32 | 1.596x |

# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: ResNet50 classification, TensorRT FP16-enabled execution (`--fp16`, `--noTF32`, no INT8).
- Input/output binding dtypes: DataType.FLOAT / ['DataType.FLOAT'].
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode, RGB, direct bilinear resize, divide by 255,
  mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, HWC-to-CHW.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: 261 validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- Comparison reference: validated FP32 package `/home/jetson/Downloads/classification_benchmark/resnet50_final/resnet50_tensorrt_fp32_640x640_batch1_warmup10_20260716_182655`.
