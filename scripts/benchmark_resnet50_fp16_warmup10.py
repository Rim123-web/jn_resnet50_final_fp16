from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from common import (
    PACKAGED_ENGINE,
    RUN,
    WARMUP_COUNT,
    class_names,
    load_image_order,
    preprocess,
    preprocess_detailed,
    softmax,
    write_csv,
    write_json,
)
from monitor import JetsonMonitor
from trt_runtime import TensorRTRunner


TIMING_FIELDS = [
    "decode_ms",
    "color_conversion_ms",
    "resize_calculation_ms",
    "resize_ms",
    "crop_padding_ms",
    "float32_conversion_ms",
    "pixel_scaling_ms",
    "mean_subtraction_ms",
    "std_division_ms",
    "hwc_to_chw_ms",
    "batch_insertion_ms",
    "contiguous_preparation_ms",
    "preprocessing_ms",
    "h2d_ms",
    "inference_ms",
    "d2h_ms",
    "output_vector_handling_ms",
    "softmax_ms",
    "argmax_topk_ms",
    "classification_object_ms",
    "postprocessing_ms",
    "transfer_inclusive_ms",
    "end_to_end_ms",
    "pipeline_overhead_ms",
]


def main():
    samples = load_image_order()
    names = class_names()
    if len(samples) != 261:
        raise RuntimeError(f"Validated image count changed: {len(samples)}")
    predictions = []
    timings = []
    results_jsonl = []
    failed = []
    warmup_rows = []
    monitor = None
    official_start_epoch = official_end_epoch = None
    official_start_perf = official_end_perf = None
    events = []

    def event(name, **details):
        events.append(
            {
                "event": name,
                "timestamp_epoch": time.time(),
                "monotonic_seconds": time.perf_counter(),
                **details,
            }
        )

    try:
        event("official_process_started")
        with TensorRTRunner(PACKAGED_ENGINE) as runner:
            event("engine_loaded", bindings=runner.binding_info)
            if tuple(runner.input_shape) != (1, 3, 640, 640):
                raise RuntimeError(f"Engine input shape mismatch: {runner.binding_info}")
            warmup_tensor = preprocess(samples[0]["path"])
            runner.synchronize()
            for index in range(WARMUP_COUNT):
                start = time.perf_counter_ns()
                _, gpu = runner.infer(warmup_tensor, timed=False)
                runner.synchronize()
                end = time.perf_counter_ns()
                warmup_rows.append(
                    {
                        "warmup_index": index + 1,
                        "image_path": samples[0]["path"],
                        "engine_execution_count": 1,
                        "wall_latency_ms": (end - start) / 1e6,
                        **gpu,
                        "excluded_from_official_timing": "YES",
                        "excluded_from_monitoring": "YES",
                    }
                )
                event("warmup_completed", warmup_index=index + 1)
            if len(warmup_rows) != 10:
                raise RuntimeError("Warm-up count is not exactly 10")

            monitor = JetsonMonitor("official_measured_pass", 0.5)
            monitor.start()
            runner.synchronize()
            official_start_epoch = time.time()
            official_start_perf = time.perf_counter()
            event("official_measured_window_started", expected_images=len(samples))

            for measured_index, sample in enumerate(samples, start=1):
                try:
                    image_start_epoch = time.time()
                    tensor, timing, original = preprocess_detailed(sample["path"])
                    logits, gpu = runner.infer(tensor, timed=True)
                    timing.update(gpu)

                    t0 = time.perf_counter_ns()
                    vector = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
                    t1 = time.perf_counter_ns()
                    timing["output_vector_handling_ms"] = (t1 - t0) / 1e6

                    t0 = time.perf_counter_ns()
                    probabilities = softmax(vector)
                    t1 = time.perf_counter_ns()
                    timing["softmax_ms"] = (t1 - t0) / 1e6

                    t0 = time.perf_counter_ns()
                    ranking = np.argsort(probabilities)[::-1]
                    predicted = int(ranking[0])
                    t1 = time.perf_counter_ns()
                    timing["argmax_topk_ms"] = (t1 - t0) / 1e6

                    t0 = time.perf_counter_ns()
                    prediction = {
                        "measured_index": measured_index,
                        "absolute_path": sample["path"],
                        "relative_path": sample["relative_path"],
                        "filename": Path(sample["path"]).name,
                        "true_class_id": sample["class_id"],
                        "true_class_name": sample["class_name"],
                        "predicted_class_id": predicted,
                        "predicted_class_name": names[predicted],
                        "correct": int(predicted == sample["class_id"]),
                        "confidence": float(probabilities[predicted]),
                        "top_k": json.dumps(
                            [
                                {
                                    "class_id": int(item),
                                    "class_name": names[int(item)],
                                    "probability": float(probabilities[item]),
                                }
                                for item in ranking
                            ]
                        ),
                        "top5_correct": None,
                        "original_width": original[0],
                        "original_height": original[1],
                        "original_channels": original[2],
                        "final_tensor_shape": "1x3x640x640",
                    }
                    t1 = time.perf_counter_ns()
                    timing["classification_object_ms"] = (t1 - t0) / 1e6
                    timing["postprocessing_ms"] = (
                        timing["output_vector_handling_ms"]
                        + timing["softmax_ms"]
                        + timing["argmax_topk_ms"]
                        + timing["classification_object_ms"]
                    )
                    timing["end_to_end_ms"] = (
                        time.perf_counter_ns() - timing.pop("_end_to_end_start_ns")
                    ) / 1e6
                    exclusive = [
                        "decode_ms",
                        "color_conversion_ms",
                        "resize_calculation_ms",
                        "resize_ms",
                        "crop_padding_ms",
                        "float32_conversion_ms",
                        "pixel_scaling_ms",
                        "mean_subtraction_ms",
                        "std_division_ms",
                        "hwc_to_chw_ms",
                        "batch_insertion_ms",
                        "contiguous_preparation_ms",
                        "h2d_ms",
                        "inference_ms",
                        "d2h_ms",
                        "output_vector_handling_ms",
                        "softmax_ms",
                        "argmax_topk_ms",
                        "classification_object_ms",
                    ]
                    timing["pipeline_overhead_ms"] = max(
                        0.0, timing["end_to_end_ms"] - sum(timing[key] for key in exclusive)
                    )
                    image_end_epoch = time.time()
                    timing_row = {
                        "measured_index": measured_index,
                        "absolute_path": sample["path"],
                        "relative_path": sample["relative_path"],
                        "image_start_epoch": image_start_epoch,
                        "image_end_epoch": image_end_epoch,
                        **{key: timing[key] for key in TIMING_FIELDS},
                    }
                    predictions.append(prediction)
                    timings.append(timing_row)
                    results_jsonl.append({**prediction, **timing_row, "logits": vector.tolist()})
                except Exception as exc:
                    failed.append(
                        {
                            "measured_index": measured_index,
                            "image_path": sample["path"],
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    raise
            runner.synchronize()
            official_end_perf = time.perf_counter()
            official_end_epoch = time.time()
            event("official_measured_window_ended", measured_images=len(timings))
            monitor.stop()
            monitor.save(
                RUN / "monitoring/primary_power_samples.csv",
                RUN / "monitoring/monitoring_configuration.json",
            )
            write_csv(RUN / "monitoring/primary_system_samples.csv", monitor.rows)
    finally:
        if monitor and monitor.thread and monitor.thread.is_alive():
            monitor.stop()
        write_csv(RUN / "raw/failed_images.csv", failed, ["measured_index", "image_path", "error", "traceback"])

    write_csv(RUN / "raw/warmup_execution_records.csv", warmup_rows)
    write_csv(RUN / "raw/per_image_predictions.csv", predictions)
    write_csv(RUN / "raw/per_image_timings.csv", timings)
    write_csv(
        RUN / "raw/ground_truth.csv",
        [
            {
                "measured_index": index,
                "absolute_path": sample["path"],
                "relative_path": sample["relative_path"],
                "true_class_id": sample["class_id"],
                "true_class_name": sample["class_name"],
            }
            for index, sample in enumerate(samples, 1)
        ],
    )
    (RUN / "raw/image_order.txt").write_text(
        "\n".join(sample["path"] for sample in samples) + "\n", encoding="utf-8"
    )
    with (RUN / "raw/per_image_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results_jsonl:
            handle.write(json.dumps(row) + "\n")
    provenance = {
        "warmup_executions": len(warmup_rows),
        "warmup_policy": "first deterministic image preprocessed once, then engine executed exactly 10 times",
        "warmups_excluded_from_timing": True,
        "warmups_excluded_from_monitoring": True,
        "measured_executions": len(timings),
        "expected_measured_executions": len(samples),
        "total_engine_executions": len(warmup_rows) + len(timings),
        "discarded_measured_rows": len(samples) - len(timings),
        "failed_images": len(failed),
        "official_start_epoch": official_start_epoch,
        "official_end_epoch": official_end_epoch,
        "official_measured_wall_time_seconds": official_end_perf - official_start_perf,
        "first_cold_engine_execution_included": False,
        "first_dataset_image_included_in_measured_pass": True,
        "monitor_sample_count": len(monitor.rows) if monitor else 0,
    }
    write_json(RUN / "raw/execution_count_provenance.json", provenance)
    event("official_process_completed", provenance=provenance)
    with (RUN / "raw/official_process_events.jsonl").open("w", encoding="utf-8") as handle:
        for row in events:
            handle.write(json.dumps(row) + "\n")
    warmup_protocol = {
        "warmup_count": WARMUP_COUNT,
        "warmup_image_policy": provenance["warmup_policy"],
        "excluded_from_official_timing": True,
        "excluded_from_monitoring": True,
        "measured_images": len(timings),
        "total_official_process_engine_executions": provenance["total_engine_executions"],
    }
    write_json(RUN / "summary/warmup_protocol.json", warmup_protocol)
    (RUN / "summary/warmup_protocol.md").write_text(
        "# Warm-up Protocol\n\n"
        f"- Warm-up executions: exactly {WARMUP_COUNT}\n"
        f"- Policy: {provenance['warmup_policy']}\n"
        "- Excluded from official timing: YES\n"
        "- Excluded from monitoring: YES\n"
        f"- Measured images: {len(timings)}\n"
        f"- Total official-process engine executions: {provenance['total_engine_executions']}\n",
        encoding="utf-8",
    )
    if (
        provenance["warmup_executions"] != 10
        or provenance["measured_executions"] != len(samples)
        or provenance["failed_images"] != 0
        or provenance["discarded_measured_rows"] != 0
    ):
        raise RuntimeError(f"Official execution provenance failed: {provenance}")
    print("OFFICIAL BENCHMARK PASS")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
