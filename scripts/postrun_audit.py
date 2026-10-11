from __future__ import annotations

import csv
import platform
import socket
import subprocess
import time

from common import (
    DATASET_ROOT,
    FP32_REFERENCE,
    RUN,
    SOURCE_MODEL,
    SOURCE_ONNX,
    read_json,
    sha256,
    write_json,
)


def output(command):
    return subprocess.run(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def main():
    with (RUN / "dataset/dataset_inventory.csv").open(newline="", encoding="utf-8") as handle:
        inventory = list(csv.DictReader(handle))
    image_checks = [
        sha256(DATASET_ROOT / row["relative_path"]) == row["sha256"] for row in inventory
    ]
    model_metadata = read_json(RUN / "model/model_metadata.json")
    checks = {
        "source_model_unchanged": sha256(SOURCE_MODEL)
        == model_metadata["selected_source"]["sha256"],
        "source_onnx_unchanged": sha256(SOURCE_ONNX)
        == sha256(RUN / "source/resnet50_inference_original_dynamic_batch.onnx")
        if (RUN / "source/resnet50_inference_original_dynamic_batch.onnx").exists()
        else True,
        "all_test_images_unchanged": all(image_checks),
        "test_image_count_unchanged": len(image_checks) == 261,
        "fp32_reference_validation_still_pass": read_json(
            FP32_REFERENCE / "validation/final_validation_report.json"
        )["status"]
        == "PASS",
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_root": str(DATASET_ROOT),
        "image_count": len(image_checks),
    }
    write_json(RUN / "validation/original_artifact_integrity.json", report)
    postflight = {
        "hostname": socket.gethostname(),
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "date_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "memory": output("free -h"),
        "swap": output("swapon --show"),
        "storage": output("df -h /home/jetson/Downloads"),
        "nvpmodel": output("nvpmodel -q 2>&1"),
        "processes": output(
            "ps -eo pid,ppid,etimes,pcpu,pmem,args | "
            "grep -Ei 'benchmark|trtexec|inference' | grep -v grep || true"
        ),
    }
    write_json(RUN / "environment/system_state_after.json", postflight)
    if report["status"] != "PASS":
        raise RuntimeError(report)
    print("POST-RUN ARTIFACT AUDIT PASS")


if __name__ == "__main__":
    main()
