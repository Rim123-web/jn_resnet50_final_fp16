# ResNet50 TensorRT FP16-Enabled Jetson Nano Benchmark

Fresh publication-oriented benchmark package.

- Model: ResNet50 classification
- Precision: TensorRT FP16-enabled execution with TF32 disabled
- Input: 1x3x640x640
- Batch: 1
- Warm-up: exactly 10 excluded engine executions
- Dataset: /home/jetson/Downloads/classification_jetson/test_imgs
- Measured images: all validated images once
- Power: Jetson module-level jtop Power TOT

Historical folders are not modified. See `summary/` and `validation/` after completion.
