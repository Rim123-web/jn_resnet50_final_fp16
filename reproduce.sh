#!/usr/bin/env bash
set -euo pipefail
RUN=/home/jetson/Downloads/classification_benchmark/resnet50_final/resnet50_tensorrt_fp16_640x640_batch1_warmup10_20260716_201533
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/prepare_project.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/inspect_onnx.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/validate_onnx.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/validate_onnx_dataset.py"
bash "$RUN/scripts/build_engine.sh"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_correctness.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/run_idle_monitor.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/benchmark_resnet50_fp16_warmup10.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/finalize_results.py"
