from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


RUN = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/jetson/Downloads/classification_jetson")
DATASET_ROOT = SOURCE_ROOT / "test_imgs"
SOURCE_MODEL = SOURCE_ROOT / "resnet50_inference_models/resnet50_inference_only.pt"
SOURCE_ONNX = SOURCE_ROOT / "resnet50_inference_models/resnet50_inference.onnx"
SOURCE_CLASS_NAMES = SOURCE_ROOT / "class_names.json"
SOURCE_PREPROCESSING = SOURCE_ROOT / "preprocessing.json"
FP32_REFERENCE = Path(
    "/home/jetson/Downloads/classification_benchmark/resnet50_final/"
    "resnet50_tensorrt_fp32_640x640_batch1_warmup10_20260716_182655"
)
HISTORICAL_SCRIPT = (
    SOURCE_ROOT
    / "resnet50_fp32_core4_benchmark/scripts/run_resnet50_fp32_core4_benchmark.py"
)
PACKAGED_MODEL = RUN / "model/resnet50_inference_only.pt"
PACKAGED_ONNX = RUN / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"
PACKAGED_ENGINE = (
    RUN / "engine/resnet50_classification_tensorrt_fp16_1x3x640x640.engine"
)
CLASS_NAMES_PATH = RUN / "config/class_names.json"
PREPROCESSING_PATH = RUN / "config/preprocessing.json"
IMAGE_ORDER_PATH = RUN / "dataset/image_order.txt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_SHAPE = (1, 3, 640, 640)
WARMUP_COUNT = 10


def ensure_dirs() -> None:
    for directory in [
        "source",
        "dataset",
        "model",
        "onnx",
        "engine",
        "config",
        "environment",
        "scripts",
        "raw",
        "metrics",
        "performance",
        "monitoring",
        "stability",
        "analysis",
        "comparison",
        "figures/png",
        "figures/svg",
        "figures/pdf",
        "plot_data",
        "logs",
        "validation",
        "summary",
    ]:
        (RUN / directory).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time": stat.st_mtime,
        "sha256": sha256(path),
    }


def run_command(
    command: list[str] | str,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    timeout: int | None = None,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    shell = isinstance(command, str)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    start = time.time()
    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("w", encoding="utf-8") if stdout_path else subprocess.DEVNULL
    ) as stdout_handle, (
        stderr_path.open("w", encoding="utf-8") if stderr_path else subprocess.DEVNULL
    ) as stderr_handle:
        result = subprocess.run(
            command,
            shell=shell,
            cwd=str(RUN),
            env=merged_env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            timeout=timeout,
        )
    append_executed_command(
        command=command if isinstance(command, str) else " ".join(command),
        start=start,
        end=time.time(),
        exit_status=result.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    return result


def append_executed_command(
    command: str,
    start: float,
    end: float,
    exit_status: int,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> None:
    path = RUN / "logs/executed_commands.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "command": command,
                    "start_epoch": start,
                    "end_epoch": end,
                    "duration_seconds": end - start,
                    "exit_status": exit_status,
                    "stdout_log": str(stdout_path) if stdout_path else None,
                    "stderr_log": str(stderr_path) if stderr_path else None,
                }
            )
            + "\n"
        )


def class_names() -> list[str]:
    data = read_json(CLASS_NAMES_PATH if CLASS_NAMES_PATH.exists() else SOURCE_CLASS_NAMES)
    if isinstance(data, dict):
        return [data[str(i)] if str(i) in data else data[i] for i in range(len(data))]
    return list(data)


def preprocessing_config() -> dict:
    return read_json(PREPROCESSING_PATH if PREPROCESSING_PATH.exists() else SOURCE_PREPROCESSING)


def load_image_order() -> list[dict]:
    names = class_names()
    by_name = {name: index for index, name in enumerate(names)}
    rows = []
    for line in IMAGE_ORDER_PATH.read_text(encoding="utf-8").splitlines():
        path = Path(line)
        rows.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(DATASET_ROOT)),
                "class_name": path.parent.name,
                "class_id": by_name[path.parent.name],
            }
        )
    return rows


def load_model(path: Path = PACKAGED_MODEL):
    import torch
    import torch.nn as nn
    from torchvision import models

    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint at {path}")
    names = checkpoint.get("class_names") or class_names()
    dropout = float(checkpoint.get("dropout", 0.0))
    model = models.resnet50(weights=None)
    features = model.fc.in_features
    model.fc = (
        nn.Sequential(nn.Dropout(dropout), nn.Linear(features, len(names)))
        if dropout > 0
        else nn.Linear(features, len(names))
    )
    cleaned = {}
    for key, tensor in checkpoint["model_state_dict"].items():
        name = key
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = tensor
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model, list(names), checkpoint


def preprocess_detailed(path: str | Path) -> tuple[np.ndarray, dict, tuple[int, int, int]]:
    config = preprocessing_config()
    mean = np.asarray(config["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(config["std"], dtype=np.float32).reshape(3, 1, 1)
    timing: dict[str, float] = {}
    e2e_start = time.perf_counter_ns()

    t0 = time.perf_counter_ns()
    image = Image.open(path)
    image.load()
    t1 = time.perf_counter_ns()
    timing["decode_ms"] = (t1 - t0) / 1e6
    original = (image.width, image.height, len(image.getbands()))
    preprocess_start = t1

    t0 = time.perf_counter_ns()
    rgb = image.convert("RGB")
    t1 = time.perf_counter_ns()
    timing["color_conversion_ms"] = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    target_size = (int(config["resize"][0]), int(config["resize"][1]))
    t1 = time.perf_counter_ns()
    timing["resize_calculation_ms"] = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    resized = rgb.resize(target_size, Image.BILINEAR)
    t1 = time.perf_counter_ns()
    timing["resize_ms"] = (t1 - t0) / 1e6
    timing["crop_padding_ms"] = 0.0

    t0 = time.perf_counter_ns()
    array = np.asarray(resized).astype(np.float32)
    t1 = time.perf_counter_ns()
    timing["float32_conversion_ms"] = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    array /= 255.0
    t1 = time.perf_counter_ns()
    timing["pixel_scaling_ms"] = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    chw = array.transpose(2, 0, 1)
    t1 = time.perf_counter_ns()
    timing["hwc_to_chw_ms"] = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    chw = chw - mean
    t1 = time.perf_counter_ns()
    timing["mean_subtraction_ms"] = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    chw = chw / std
    t1 = time.perf_counter_ns()
    timing["std_division_ms"] = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    batched = chw[None, ...]
    t1 = time.perf_counter_ns()
    timing["batch_insertion_ms"] = (t1 - t0) / 1e6

    t0 = time.perf_counter_ns()
    tensor = np.ascontiguousarray(batched.astype(np.float32, copy=False))
    t1 = time.perf_counter_ns()
    timing["contiguous_preparation_ms"] = (t1 - t0) / 1e6
    timing["preprocessing_ms"] = (t1 - preprocess_start) / 1e6
    timing["_end_to_end_start_ns"] = e2e_start
    return tensor, timing, original


def preprocess(path: str | Path) -> np.ndarray:
    return preprocess_detailed(path)[0]


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def stats(values: list[float]) -> dict:
    if not values:
        return {key: None for key in [
            "count", "mean", "median", "std", "min", "max", "p50", "p90",
            "p95", "p99", "coefficient_of_variation", "accumulated"
        ]}
    mean = statistics.mean(values)
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "coefficient_of_variation": statistics.pstdev(values) / mean if mean else None,
        "accumulated": sum(values),
    }


def compute_classification_metrics(rows: list[dict], names: list[str]) -> dict:
    n = len(names)
    matrix = [[0] * n for _ in range(n)]
    for row in rows:
        matrix[int(row["true_class_id"])][int(row["predicted_class_id"])] += 1
    per_class = []
    for index, name in enumerate(names):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(n) if row != index)
        fn = sum(matrix[index][column] for column in range(n) if column != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_id": index,
                "class_name": name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    weighted = lambda key: (
        sum(row[key] * row["support"] for row in per_class) / total if total else 0.0
    )
    normalized = [
        [value / sum(row) if sum(row) else 0.0 for value in row] for row in matrix
    ]
    return {
        "task": "classification",
        "total_images": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else 0.0,
        "top5_accuracy": None if n < 5 else sum(int(row["top5_correct"]) for row in rows) / total,
        "macro_precision": statistics.mean(row["precision"] for row in per_class),
        "macro_recall": statistics.mean(row["recall"] for row in per_class),
        "macro_f1": statistics.mean(row["f1"] for row in per_class),
        "weighted_precision": weighted("precision"),
        "weighted_recall": weighted("recall"),
        "weighted_f1": weighted("f1"),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_normalized": normalized,
        "class_names": names,
    }


def copy_preserve(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def human_bytes(value: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024 or unit == "GiB":
            return f"{value:.3f} {unit}"
        value /= 1024
    return str(value)
