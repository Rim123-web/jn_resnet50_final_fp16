from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from common import (
    CLASS_NAMES_PATH,
    DATASET_ROOT,
    FP32_REFERENCE,
    HISTORICAL_SCRIPT,
    IMAGE_ORDER_PATH,
    IMG_EXTS,
    INPUT_SHAPE,
    PACKAGED_MODEL,
    PACKAGED_ONNX,
    PREPROCESSING_PATH,
    RUN,
    SOURCE_CLASS_NAMES,
    SOURCE_MODEL,
    SOURCE_ONNX,
    SOURCE_PREPROCESSING,
    SOURCE_ROOT,
    WARMUP_COUNT,
    class_names,
    copy_preserve,
    ensure_dirs,
    file_record,
    load_model,
    preprocessing_config,
    read_json,
    sha256,
    write_csv,
    write_json,
)


def command_output(command: str) -> str:
    return subprocess.run(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).stdout


def environment_inventory():
    preflight = {
        "hostname": socket.gethostname(),
        "username": os.environ.get("USER"),
        "date_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "device_model": Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode(),
        "l4t": Path("/etc/nv_tegra_release").read_text().strip(),
        "os_release": Path("/etc/os-release").read_text(),
        "kernel": platform.release(),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "cpu_governors": [
            path.read_text().strip()
            for path in sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
        ],
        "nvpmodel": command_output("nvpmodel -q 2>&1"),
        "jetson_clocks": "OFF (verified from jtop preflight; direct --show requires root)",
        "memory": command_output("free -h"),
        "swap": command_output("swapon --show"),
        "storage": command_output("df -h /home/jetson/Downloads"),
        "processes": command_output(
            "ps -eo pid,ppid,etimes,pcpu,pmem,args | "
            "grep -Ei 'benchmark|trtexec|inference' | grep -v grep || true"
        ),
        "thermal": command_output(
            "for z in /sys/class/thermal/thermal_zone*; do "
            "printf '%s|' \"$z\"; cat \"$z/type\" | tr '\\n' '|'; cat \"$z/temp\"; done"
        ),
        "cuda_nvcc": command_output("/usr/local/cuda/bin/nvcc --version"),
        "tensorrt_packages": command_output(
            "dpkg-query -W -f='${Package}|${Version}\\n' 'libnvinfer*' 2>/dev/null | sort"
        ),
    }
    write_json(RUN / "environment/system_state_before.json", preflight)
    (RUN / "environment/preflight.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in preflight.items()), encoding="utf-8"
    )
    (RUN / "environment/storage_before.txt").write_text(preflight["storage"], encoding="utf-8")
    process_rows = []
    for line in preflight["processes"].splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6:
            process_rows.append(dict(zip(["pid", "ppid", "elapsed_s", "cpu", "memory", "command"], parts)))
    write_csv(
        RUN / "environment/processes_before.csv",
        process_rows,
        ["pid", "ppid", "elapsed_s", "cpu", "memory", "command"],
    )
    thermal_rows = []
    for line in preflight["thermal"].splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            thermal_rows.append(
                {
                    "zone": parts[0],
                    "sensor": parts[1],
                    "temperature_millicelsius": parts[2],
                    "temperature_c": float(parts[2]) / 1000.0 if parts[2] else None,
                }
            )
    write_csv(RUN / "environment/thermal_state_before.csv", thermal_rows)

    versions = {}
    for module in ["torch", "torchvision", "numpy", "PIL", "cv2", "jtop", "psutil", "matplotlib"]:
        try:
            imported = __import__(module)
            versions[module] = {
                "version": getattr(imported, "__version__", "NA"),
                "path": getattr(imported, "__file__", "NA"),
            }
        except Exception as exc:
            versions[module] = {"error": repr(exc)}
    try:
        import sys as _sys
        _sys.path.insert(0, "/home/jetson/.local/lib/python3.8/site-packages")
        import tensorrt
        versions["tensorrt_python"] = {
            "version": tensorrt.__version__,
            "path": tensorrt.__file__,
        }
    except Exception as exc:
        versions["tensorrt_python"] = {"error": repr(exc)}
    write_json(RUN / "environment/software_versions.json", versions)
    (RUN / "environment/pip_freeze.txt").write_text(
        command_output(f"{sys.executable} -m pip freeze"), encoding="utf-8"
    )
    reference_path = FP32_REFERENCE / "environment/system_state_before.json"
    if not reference_path.exists():
        raise RuntimeError(f"FP32 environment reference missing: {reference_path}")
    reference = read_json(reference_path)
    checks = {
        "hostname": preflight["hostname"] == reference.get("hostname"),
        "architecture": preflight["architecture"] == reference.get("architecture"),
        "device_model": preflight["device_model"] == reference.get("device_model"),
        "kernel": preflight["kernel"] == reference.get("kernel"),
        "l4t": preflight["l4t"] == reference.get("l4t"),
        "cpu_online": preflight["cpu_online"] == reference.get("cpu_online"),
        "cpu_governors": preflight["cpu_governors"] == reference.get("cpu_governors"),
        "nvpmodel": preflight["nvpmodel"] == reference.get("nvpmodel"),
    }
    comparison = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "critical_checks": checks,
        "current": preflight,
        "fp32_reference": reference,
        "notes": {
            "memory": "Snapshot values are recorded but intentionally not equality-gated.",
            "swap": "Snapshot values are recorded but intentionally not equality-gated.",
            "temperature": "Thermal snapshots are expected to vary between runs.",
            "jetson_clocks": "Both packages use the same recorded OFF policy.",
        },
    }
    write_json(RUN / "environment/fp32_environment_comparison.json", comparison)
    if comparison["status"] != "PASS":
        raise RuntimeError(f"Critical environment parity failed: {checks}")


def inventory_dataset():
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(f"Validated dataset root is missing: {DATASET_ROOT}")
    names = class_names()
    directories = {path.name: path for path in DATASET_ROOT.iterdir() if path.is_dir()}
    if set(directories) != set(names):
        raise RuntimeError(f"Class-folder mismatch: expected {names}, found {sorted(directories)}")
    rows = []
    failures = []
    hidden = []
    unsupported = []
    content_hashes = defaultdict(list)
    for class_id, name in enumerate(names):
        for path in sorted(directories[name].rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(DATASET_ROOT).parts):
                hidden.append(str(path))
            if path.suffix.lower() not in IMG_EXTS:
                unsupported.append(str(path))
                continue
            status = "OK"
            error = ""
            width = height = channels = None
            try:
                if path.stat().st_size == 0:
                    raise ValueError("zero-byte file")
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    channels = len(image.getbands())
            except Exception as exc:
                status = "FAILED"
                error = repr(exc)
                failures.append({"path": str(path), "error": error})
            digest = sha256(path)
            content_hashes[digest].append(str(path))
            rows.append(
                {
                    "absolute_path": str(path),
                    "relative_path": str(path.relative_to(DATASET_ROOT)),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "class_folder": name,
                    "class_id": class_id,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "file_size_bytes": path.stat().st_size,
                    "sha256": digest,
                    "decoding_status": status,
                    "error": error,
                }
            )
    if failures or unsupported or hidden:
        raise RuntimeError(
            f"Dataset validation failed: corrupt={len(failures)}, unsupported={len(unsupported)}, hidden={len(hidden)}"
        )
    if len(rows) != 261:
        raise RuntimeError(f"Expected historical dataset size 261, found {len(rows)}")
    order = [row["absolute_path"] for row in rows]
    IMAGE_ORDER_PATH.write_text("\n".join(order) + "\n", encoding="utf-8")
    write_csv(RUN / "dataset/dataset_inventory.csv", rows)
    write_json(RUN / "dataset/dataset_inventory.json", rows)
    counts = Counter(row["class_folder"] for row in rows)
    distribution = [
        {"class_id": index, "class_name": name, "image_count": counts[name]}
        for index, name in enumerate(names)
    ]
    write_csv(RUN / "dataset/class_distribution.csv", distribution)
    write_json(RUN / "dataset/class_mapping.json", {str(i): name for i, name in enumerate(names)})
    duplicate_content = {key: value for key, value in content_hashes.items() if len(value) > 1}
    hash_summary = {
        "dataset_root": str(DATASET_ROOT),
        "total_images": len(rows),
        "combined_ordered_sha256": sha256(IMAGE_ORDER_PATH),
        "duplicate_content_groups": duplicate_content,
        "all_images_decoded": True,
        "hidden_files": hidden,
        "unsupported_files": unsupported,
        "zero_byte_or_corrupt_files": failures,
    }
    write_json(RUN / "dataset/dataset_hash_summary.json", hash_summary)
    report = [
        "# Dataset Validation Report",
        "",
        f"- Source: `{DATASET_ROOT}`",
        f"- Total validated images: {len(rows)}",
        f"- Classes: {len(names)}",
        f"- Class mapping: {dict(enumerate(names))}",
        f"- All images decoded: YES",
        f"- Duplicate paths: 0",
        f"- Duplicate-content groups: {len(duplicate_content)}",
        f"- Hidden/unsupported/corrupt/zero-byte files: 0",
        "- Ordering: deterministic class order from class_names.json, then lexicographic path order.",
    ]
    (RUN / "dataset/dataset_validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    reference_hash = read_json(FP32_REFERENCE / "dataset/dataset_hash_summary.json")
    reference_mapping = read_json(FP32_REFERENCE / "dataset/class_mapping.json")
    parity_checks = {
        "image_count": len(rows) == reference_hash.get("total_images"),
        "image_order_sha256": sha256(IMAGE_ORDER_PATH)
        == reference_hash.get("combined_ordered_sha256"),
        "dataset_inventory_sha256": sha256(RUN / "dataset/dataset_inventory.csv")
        == sha256(FP32_REFERENCE / "dataset/dataset_inventory.csv"),
        "class_mapping": {str(i): name for i, name in enumerate(names)} == reference_mapping,
    }
    parity = {
        "status": "PASS" if all(parity_checks.values()) else "FAIL",
        "checks": parity_checks,
        "fp16_dataset_root": str(DATASET_ROOT),
        "fp32_dataset_root": reference_hash.get("dataset_root"),
        "fp16_image_count": len(rows),
        "fp32_image_count": reference_hash.get("total_images"),
        "fp16_image_order_sha256": sha256(IMAGE_ORDER_PATH),
        "fp32_image_order_sha256": reference_hash.get("combined_ordered_sha256"),
    }
    write_json(RUN / "validation/fp32_fp16_dataset_parity.json", parity)
    if parity["status"] != "PASS":
        raise RuntimeError(f"FP32/FP16 dataset parity failed: {parity_checks}")


def preserve_and_inspect_model():
    for required in [SOURCE_MODEL, SOURCE_ONNX, SOURCE_CLASS_NAMES, SOURCE_PREPROCESSING, HISTORICAL_SCRIPT]:
        if not required.exists():
            raise RuntimeError(f"Required source artifact missing: {required}")
    copy_preserve(SOURCE_MODEL, PACKAGED_MODEL)
    reference_onnx = FP32_REFERENCE / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"
    if not reference_onnx.exists():
        raise RuntimeError(f"Validated FP32 ONNX is missing: {reference_onnx}")
    copy_preserve(reference_onnx, PACKAGED_ONNX)
    copy_preserve(SOURCE_CLASS_NAMES, CLASS_NAMES_PATH)
    copy_preserve(SOURCE_PREPROCESSING, PREPROCESSING_PATH)
    copy_preserve(HISTORICAL_SCRIPT, RUN / "source/historical_run_resnet50_fp32_core4_benchmark.py")
    model, names, checkpoint = load_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "architecture": "ResNet50",
        "task": "classification",
        "class_count": len(names),
        "class_names": names,
        "input_shape": list(INPUT_SHAPE),
        "input_dtype": "float32",
        "checkpoint_keys": sorted(checkpoint),
        "checkpoint_model_name": checkpoint.get("model_name"),
        "checkpoint_img_size": checkpoint.get("img_size"),
        "checkpoint_dropout": checkpoint.get("dropout"),
        "parameter_count": parameter_count,
        "selected_source": file_record(SOURCE_MODEL),
        "packaged_copy": file_record(PACKAGED_MODEL),
        "copy_hash_matches": sha256(SOURCE_MODEL) == sha256(PACKAGED_MODEL),
        "fp32_reference_source_model_sha256": sha256(FP32_REFERENCE / "model/resnet50_inference_only.pt"),
        "source_model_hash_matches_fp32": sha256(SOURCE_MODEL)
        == sha256(FP32_REFERENCE / "model/resnet50_inference_only.pt"),
        "reused_fp32_onnx": file_record(reference_onnx),
        "packaged_onnx": file_record(PACKAGED_ONNX),
        "onnx_hash_matches_fp32": sha256(reference_onnx) == sha256(PACKAGED_ONNX),
    }
    if metadata["architecture"] != checkpoint.get("model_name"):
        raise RuntimeError(f"Checkpoint model_name mismatch: {checkpoint.get('model_name')}")
    if names != class_names():
        raise RuntimeError("Checkpoint class mapping conflicts with class_names.json")
    if not metadata["source_model_hash_matches_fp32"]:
        raise RuntimeError("Source model SHA-256 does not match the validated FP32 package")
    if not metadata["onnx_hash_matches_fp32"]:
        raise RuntimeError("Copied ONNX SHA-256 does not match the validated FP32 package")
    write_json(RUN / "model/model_metadata.json", metadata)
    write_json(
        RUN / "model/selected_model_provenance.json",
        {
            "selected_model": str(SOURCE_MODEL),
            "selection_reason": (
                "Exact inference-only FP32 checkpoint referenced by the executed historical "
                "ResNet50 Core4 benchmark; duplicate deployment copy has identical SHA-256."
            ),
            "historical_script": str(HISTORICAL_SCRIPT),
            "historical_script_sha256": sha256(HISTORICAL_SCRIPT),
            "highest_confidence": True,
        },
    )
    candidates = []
    for path in sorted(SOURCE_ROOT.rglob("*resnet50*")):
        if path.is_file() and path.suffix.lower() in {
            ".pt", ".pth", ".onnx", ".engine", ".plan", ".trt"
        }:
            candidates.append(
                {
                    **file_record(path),
                    "extension": path.suffix.lower(),
                    "likely_precision": "INT8" if "int8" in path.name.lower() else "FP16" if "fp16" in path.name.lower() else "FP32/unknown",
                    "selected": path == SOURCE_MODEL,
                }
            )
    write_csv(RUN / "model/model_inventory.csv", candidates)
    (RUN / "model/model_sha256.txt").write_text(
        f"{sha256(PACKAGED_MODEL)}  {PACKAGED_MODEL.name}\n", encoding="utf-8"
    )
    (RUN / "model/model_summary.txt").write_text(
        "\n".join(
            [
                "Architecture: ResNet50",
                f"Classes: {len(names)}",
                f"Class names: {names}",
                f"Parameters: {parameter_count}",
                f"Source: {SOURCE_MODEL}",
                f"Source SHA-256: {sha256(SOURCE_MODEL)}",
                f"Packaged copy: {PACKAGED_MODEL}",
                "Model load: PASS",
            "FP32 source parameters: VERIFIED",
            f"Source model hash matches FP32 package: {metadata['source_model_hash_matches_fp32']}",
            f"ONNX hash matches FP32 package: {metadata['onnx_hash_matches_fp32']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def preprocessing_audit():
    config = preprocessing_config()
    expected = {
        "img_size": 640,
        "resize": [640, 640],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_shape": [1, 3, 640, 640],
    }
    if config != expected:
        raise RuntimeError(f"Preprocessing differs from verified historical config: {config}")
    write_json(
        RUN / "config/normalization_verification.json",
        {
            "selected_mean": config["mean"],
            "selected_std": config["std"],
            "source_file": str(SOURCE_PREPROCESSING),
            "source_sha256": sha256(SOURCE_PREPROCESSING),
            "exact_lines": "4-5",
            "historical_code_evidence": {
                "path": str(HISTORICAL_SCRIPT),
                "lines": "189-198 and 601-608",
            },
            "evidence_strength": "HIGH",
            "conflicting_values": [],
        },
    )
    evidence = [
        {
            "property": "decoder",
            "value": "PIL.Image.open followed by load/convert",
            "source_file": str(HISTORICAL_SCRIPT),
            "line": "194",
        },
        {
            "property": "color_order",
            "value": "RGB",
            "source_file": str(HISTORICAL_SCRIPT),
            "line": "194",
        },
        {
            "property": "resize",
            "value": "direct 640x640 PIL bilinear; no crop or aspect preservation",
            "source_file": str(HISTORICAL_SCRIPT),
            "line": "194",
        },
        {
            "property": "normalization",
            "value": f"mean={config['mean']}, std={config['std']}",
            "source_file": str(SOURCE_PREPROCESSING),
            "line": "4-5",
        },
        {
            "property": "layout_dtype",
            "value": "HWC-to-CHW, contiguous FP32, batch dimension",
            "source_file": str(HISTORICAL_SCRIPT),
            "line": "195-198",
        },
    ]
    write_csv(RUN / "config/historical_preprocessing_evidence.csv", evidence)
    (RUN / "config/historical_preprocessing_audit.md").write_text(
        "\n".join(
            [
                "# Historical Preprocessing Audit",
                "",
                f"- Decoder: PIL.Image.open",
                "- Color ordering: RGB",
                "- Geometry: direct bilinear resize to 640 x 640; no crop, pad, or letterbox",
                "- Pixel scale: divide by 255",
                f"- Mean: {config['mean']}",
                f"- Standard deviation: {config['std']}",
                "- Layout: HWC to CHW",
                "- Tensor: contiguous float32 [1,3,640,640]",
                f"- Primary evidence: `{HISTORICAL_SCRIPT}` lines 189-198",
                f"- Configuration evidence: `{SOURCE_PREPROCESSING}` lines 2-6",
                "- Ambiguity: none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reference_config = read_json(FP32_REFERENCE / "config/preprocessing.json")
    parity = {
        "status": "PASS" if config == reference_config else "FAIL",
        "fp16_preprocessing": config,
        "fp32_preprocessing": reference_config,
        "fp16_sha256": sha256(PREPROCESSING_PATH),
        "fp32_sha256": sha256(FP32_REFERENCE / "config/preprocessing.json"),
        "exact_match": config == reference_config,
    }
    write_json(RUN / "config/fp32_fp16_preprocessing_parity.json", parity)
    (RUN / "config/preprocessing_specification.yaml").write_text(
        "\n".join(
            [
                "decoder: PIL.Image.open",
                "color_order: RGB",
                "resize:",
                "  width: 640",
                "  height: 640",
                "  interpolation: PIL bilinear",
                "  aspect_ratio_preserved: false",
                "  crop: none",
                "  padding: none",
                "pixel_scale: divide_by_255",
                f"mean: {config['mean']}",
                f"std: {config['std']}",
                "layout: HWC_to_CHW",
                "dtype: float32",
                "contiguous: true",
                "batch_dimension: 1",
                "input_shape: [1, 3, 640, 640]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if parity["status"] != "PASS":
        raise RuntimeError("Preprocessing differs from the validated FP32 package")


def write_reproduction_files():
    manifest = {
        "run_directory": str(RUN),
        "task": "classification",
        "model": "ResNet50",
        "precision": "TensorRT FP16-enabled execution",
        "tf32_disabled": True,
        "fp16_enabled": True,
        "int8_disabled": True,
        "batch_size": 1,
        "input_shape": list(INPUT_SHAPE),
        "warmup_count": WARMUP_COUNT,
        "dataset_source": str(DATASET_ROOT),
        "source_model": str(SOURCE_MODEL),
        "source_onnx": str(FP32_REFERENCE / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"),
        "fp32_reference_package": str(FP32_REFERENCE),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PREPARED",
    }
    write_json(RUN / "run_manifest.json", manifest)
    (RUN / "requirements.txt").write_text(
        "\n".join(
            [
                "torch==1.13.0a0+git7c98e70",
                "torchvision==0.14.0a0+5ce4506",
                "numpy==1.23.5",
                "Pillow==9.2.0",
                "matplotlib==3.7.5",
                "psutil==7.2.2",
                "jetson-stats==4.3.2",
                "TensorRT==8.0.1.6 (Jetson system/user-site package; do not pip install)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reproduce = f"""#!/usr/bin/env bash
set -euo pipefail
RUN={RUN!s}
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/prepare_project.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/inspect_onnx.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/validate_onnx.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/validate_onnx_dataset.py"
bash "$RUN/scripts/build_engine.sh"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_correctness.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/run_idle_monitor.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/benchmark_resnet50_fp16_warmup10.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/finalize_results.py"
"""
    (RUN / "reproduce.sh").write_text(reproduce, encoding="utf-8")
    (RUN / "reproduce.sh").chmod(0o755)
    (RUN / "reproduce.ps1").write_text(
        "# Reproduction is supported on the Jetson through reproduce.sh.\n"
        f"Write-Host 'Run on Jetson: bash {RUN}/reproduce.sh'\n",
        encoding="utf-8",
    )
    (RUN / "README.md").write_text(
        f"""# ResNet50 TensorRT FP16-Enabled Jetson Nano Benchmark

Fresh publication-oriented benchmark package.

- Model: ResNet50 classification
- Precision: TensorRT FP16-enabled execution with TF32 disabled
- Input: 1x3x640x640
- Batch: 1
- Warm-up: exactly 10 excluded engine executions
- Dataset: {DATASET_ROOT}
- Measured images: all validated images once
- Power: Jetson module-level jtop Power TOT

Historical folders are not modified. See `summary/` and `validation/` after completion.
""",
        encoding="utf-8",
    )


def main():
    ensure_dirs()
    active = command_output(
        "ps -eo pid,args | "
        "grep -Ei '/trtexec|python[^ ]* .*benchmark_resnet|python[^ ]* .*run_resnet50' | "
        "grep -v grep | grep -v prepare_project.py || true"
    ).strip()
    if active:
        raise RuntimeError(f"Another relevant benchmark/process is active:\n{active}")
    copy_preserve(Path(__file__).resolve(), RUN / "scripts/prepare_project.py")
    environment_inventory()
    copy_preserve(SOURCE_CLASS_NAMES, CLASS_NAMES_PATH)
    copy_preserve(SOURCE_PREPROCESSING, PREPROCESSING_PATH)
    inventory_dataset()
    preserve_and_inspect_model()
    preprocessing_audit()
    reference_validation = read_json(FP32_REFERENCE / "validation/final_validation_report.json")
    if reference_validation.get("status") != "PASS":
        raise RuntimeError("Validated FP32 reference package is not PASS")
    reference_record = {
        "path": str(FP32_REFERENCE),
        "final_validation_status": reference_validation.get("status"),
        "run_manifest": read_json(FP32_REFERENCE / "run_manifest.json"),
        "source_model": file_record(FP32_REFERENCE / "model/resnet50_inference_only.pt"),
        "onnx": file_record(FP32_REFERENCE / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"),
    }
    write_json(RUN / "validation/fp32_reference_package.json", reference_record)
    parity_checks = {
        "fp32_reference_validation_pass": True,
        "source_model_sha256": sha256(PACKAGED_MODEL)
        == sha256(FP32_REFERENCE / "model/resnet50_inference_only.pt"),
        "onnx_sha256": sha256(PACKAGED_ONNX)
        == sha256(FP32_REFERENCE / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"),
        "dataset": read_json(RUN / "validation/fp32_fp16_dataset_parity.json")["status"] == "PASS",
        "preprocessing": read_json(RUN / "config/fp32_fp16_preprocessing_parity.json")["status"] == "PASS",
        "class_mapping": sha256(RUN / "dataset/class_mapping.json")
        == sha256(FP32_REFERENCE / "dataset/class_mapping.json"),
        "input_shape": list(INPUT_SHAPE)
        == read_json(FP32_REFERENCE / "run_manifest.json").get("input_shape"),
        "batch_size": 1 == read_json(FP32_REFERENCE / "run_manifest.json").get("batch_size"),
        "warmup_count": WARMUP_COUNT
        == read_json(FP32_REFERENCE / "run_manifest.json").get("warmup_count"),
        "monitor_interval_seconds": 0.5
        == read_json(FP32_REFERENCE / "monitoring/monitoring_configuration.json").get(
            "requested_interval_seconds"
        ),
    }
    parity_status = "PASS" if all(parity_checks.values()) else "FAIL"
    precheck = {
        "status": parity_status,
        "checks": parity_checks,
        "experimental_variable": "TensorRT execution precision only: FP32 reference versus FP16-enabled execution",
    }
    write_json(RUN / "validation/fp32_fp16_protocol_parity_precheck.json", precheck)
    (RUN / "validation/fp32_fp16_protocol_parity_precheck.md").write_text(
        "# FP32/FP16 Protocol Parity Precheck\n\n"
        + "\n".join(
            f"- {key}: **{'PASS' if value else 'FAIL'}**"
            for key, value in parity_checks.items()
        )
        + f"\n\nOverall: **{parity_status}**\n",
        encoding="utf-8",
    )
    if parity_status != "PASS":
        raise RuntimeError(f"FP32/FP16 protocol parity precheck failed: {parity_checks}")
    write_reproduction_files()
    write_json(RUN / "validation/preparation_status.json", {"status": "PASS"})
    print("PREPARATION PASS")
    print(f"RUN={RUN}")
    print(f"DATASET={DATASET_ROOT}")
    print(f"IMAGES={len(IMAGE_ORDER_PATH.read_text().splitlines())}")
    print(f"MODEL={SOURCE_MODEL}")
    print(f"MODEL_SHA256={sha256(SOURCE_MODEL)}")
    print(f"ONNX={PACKAGED_ONNX}")
    print(f"ONNX_SHA256={sha256(PACKAGED_ONNX)}")


if __name__ == "__main__":
    main()
