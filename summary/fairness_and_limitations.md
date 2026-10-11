# Fairness and Limitations

- This is a single-device, batch-1 deployment benchmark.
- TensorRT was built locally for this Jetson and is not portable across TensorRT/GPU versions.
- Power TOT is module-level input power, not GPU-only power.
- jtop sampling is slower than per-image inference stages; per-image power pairing is nearest-sample only.
- CUDA event synchronization improves stage attribution but can reduce throughput relative to an asynchronous pipeline.
- FP16 builder mode permits TensorRT to retain FP32 fallback layers when required; the verbose builder log is preserved, but every internal layer is not independently proven FP16.
- Direct 640x640 resize follows the historical model pipeline and does not preserve aspect ratio.
