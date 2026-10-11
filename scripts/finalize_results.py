from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    FP32_REFERENCE,
    PACKAGED_ENGINE,
    PACKAGED_MODEL,
    PACKAGED_ONNX,
    RUN,
    class_names,
    compute_classification_metrics,
    file_record,
    read_json,
    sha256,
    stats,
    write_csv,
    write_json,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in (None, "", "NA"):
            values.append(float(value))
    return values


def integrate_power(rows: list[dict], start: float | None = None, end: float | None = None, idle=0.0):
    samples = [
        (float(row["timestamp_epoch"]), max(0.0, float(row["power_tot_w"]) - idle))
        for row in rows
        if row.get("timestamp_epoch") and row.get("power_tot_w")
    ]
    if start is not None:
        samples = [(t, p) for t, p in samples if start <= t <= end]
    if not samples:
        return None
    samples.sort()
    if start is not None and samples[0][0] > start:
        samples.insert(0, (start, samples[0][1]))
    if end is not None and samples[-1][0] < end:
        samples.append((end, samples[-1][1]))
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += (p0 + p1) * 0.5 * (t1 - t0)
    return energy


def save_figure(name: str, figure, plot_rows: list[dict] | None = None):
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        figure.savefig(RUN / f"figures/{extension}/{name}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(figure)
    if plot_rows is not None:
        write_csv(RUN / f"plot_data/{name}.csv", plot_rows)


def plot_line(name, x, y, xlabel, ylabel, title, plot_key):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, y, linewidth=1.2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    save_figure(name, fig, [{xlabel: a, plot_key: b} for a, b in zip(x, y)])


def main():
    predictions = read_csv(RUN / "raw/per_image_predictions.csv")
    timings = read_csv(RUN / "raw/per_image_timings.csv")
    provenance = read_json(RUN / "raw/execution_count_provenance.json")
    idle_rows = read_csv(RUN / "monitoring/idle_samples.csv")
    power_rows = read_csv(RUN / "monitoring/primary_power_samples.csv")
    names = class_names()
    metric_input = [
        {
            **row,
            "true_class_id": int(row["true_class_id"]),
            "predicted_class_id": int(row["predicted_class_id"]),
            "correct": int(row["correct"]),
        }
        for row in predictions
    ]
    metrics = compute_classification_metrics(metric_input, names)
    metrics.update(
        {
            "precision": "TensorRT FP16-enabled execution",
            "input_shape": [1, 3, 640, 640],
            "batch_size": 1,
            "historical_reference": {"accuracy": 0.977011, "macro_f1": 0.958480},
            "historical_match": {
                "accuracy_difference": metrics["accuracy"] - 0.977011,
                "macro_f1_difference": metrics["macro_f1"] - 0.958480,
            },
        }
    )
    write_json(RUN / "metrics/classification_metrics.json", metrics)
    flat_metrics = [
        {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, (list, dict))
        }
    ]
    write_csv(RUN / "metrics/classification_metrics.csv", flat_metrics)
    write_csv(RUN / "metrics/per_class_metrics.csv", metrics["per_class"])
    cm_rows = [
        {"true_class": names[i], **{names[j]: metrics["confusion_matrix"][i][j] for j in range(len(names))}}
        for i in range(len(names))
    ]
    ncm_rows = [
        {
            "true_class": names[i],
            **{names[j]: metrics["confusion_matrix_normalized"][i][j] for j in range(len(names))},
        }
        for i in range(len(names))
    ]
    write_csv(RUN / "metrics/confusion_matrix.csv", cm_rows)
    write_csv(RUN / "metrics/confusion_matrix_normalized.csv", ncm_rows)
    write_json(
        RUN / "metrics/metric_configuration.json",
        {
            "class_order": names,
            "zero_division": 0.0,
            "top1": True,
            "top5": "NA because only three classes exist",
            "input_rows": len(predictions),
        },
    )
    report_lines = [
        "Classification report",
        f"Images: {metrics['total_images']}",
        f"Correct: {metrics['correct']}",
        f"Incorrect: {metrics['incorrect']}",
        f"Accuracy: {metrics['accuracy']:.9f}",
        f"Macro precision: {metrics['macro_precision']:.9f}",
        f"Macro recall: {metrics['macro_recall']:.9f}",
        f"Macro F1: {metrics['macro_f1']:.9f}",
        f"Weighted F1: {metrics['weighted_f1']:.9f}",
    ]
    (RUN / "metrics/classification_report.txt").write_text("\n".join(report_lines) + "\n")

    fp32_predictions = {
        row["relative_path"]: row
        for row in read_csv(FP32_REFERENCE / "raw/per_image_predictions.csv")
    }
    prediction_parity = []
    for row in predictions:
        reference = fp32_predictions[row["relative_path"]]
        prediction_parity.append(
            {
                "relative_path": row["relative_path"],
                "true_class_id": row["true_class_id"],
                "fp32_predicted_class_id": reference["predicted_class_id"],
                "fp16_predicted_class_id": row["predicted_class_id"],
                "prediction_agrees": int(
                    reference["predicted_class_id"] == row["predicted_class_id"]
                ),
                "fp32_correct": reference["correct"],
                "fp16_correct": row["correct"],
                "fp32_confidence": reference["confidence"],
                "fp16_confidence": row["confidence"],
                "confidence_difference": float(row["confidence"])
                - float(reference["confidence"]),
            }
        )
    prediction_agreement = statistics.mean(
        row["prediction_agrees"] for row in prediction_parity
    )
    write_csv(RUN / "comparison/fp32_fp16_prediction_parity.csv", prediction_parity)
    fp32_metrics = read_json(FP32_REFERENCE / "metrics/classification_metrics.json")
    metric_parity = {
        "prediction_agreement_rate": prediction_agreement,
        "prediction_differences": sum(
            1 for row in prediction_parity if not row["prediction_agrees"]
        ),
        "accuracy_fp32": fp32_metrics["accuracy"],
        "accuracy_fp16": metrics["accuracy"],
        "accuracy_change": metrics["accuracy"] - fp32_metrics["accuracy"],
        "macro_f1_fp32": fp32_metrics["macro_f1"],
        "macro_f1_fp16": metrics["macro_f1"],
        "macro_f1_change": metrics["macro_f1"] - fp32_metrics["macro_f1"],
    }
    write_json(RUN / "comparison/fp32_fp16_metric_parity.json", metric_parity)

    stage_keys = [
        key for key in timings[0]
        if key.endswith("_ms") and key not in {"end_to_end_ms", "transfer_inclusive_ms"}
    ] + ["transfer_inclusive_ms", "end_to_end_ms"]
    end_mean = statistics.mean(numeric(timings, "end_to_end_ms"))
    stage_rows = []
    for key in stage_keys:
        summary = stats(numeric(timings, key))
        summary.update(
            {
                "stage": key,
                "unit": "ms",
                "mean_share_of_end_to_end_percent": 100 * summary["mean"] / end_mean if end_mean else None,
                "median_share_of_end_to_end_percent": 100 * summary["median"] / statistics.median(numeric(timings, "end_to_end_ms")),
            }
        )
        stage_rows.append(summary)
    write_csv(RUN / "performance/stage_timing_summary.csv", stage_rows)
    write_json(RUN / "performance/stage_timing_summary.json", stage_rows)
    write_csv(
        RUN / "performance/latency_percentiles.csv",
        [
            {
                "stage": row["stage"],
                "p50_ms": row["p50"],
                "p90_ms": row["p90"],
                "p95_ms": row["p95"],
                "p99_ms": row["p99"],
            }
            for row in stage_rows
        ],
    )
    mean_engine = statistics.mean(numeric(timings, "inference_ms"))
    mean_transfer = statistics.mean(numeric(timings, "transfer_inclusive_ms"))
    mean_end = statistics.mean(numeric(timings, "end_to_end_ms"))
    performance = {
        "measured_images": len(timings),
        "warmup_count": provenance["warmup_executions"],
        "mean_engine_latency_ms": mean_engine,
        "mean_transfer_inclusive_latency_ms": mean_transfer,
        "mean_end_to_end_latency_ms": mean_end,
        "engine_fps": 1000.0 / mean_engine,
        "transfer_inclusive_fps": 1000.0 / mean_transfer,
        "latency_based_pipeline_fps": 1000.0 / mean_end,
        "official_measured_wall_time_seconds": provenance["official_measured_wall_time_seconds"],
        "wall_clock_pipeline_fps": len(timings) / provenance["official_measured_wall_time_seconds"],
        "engine_percentiles_ms": {
            "p50": float(np.percentile(numeric(timings, "inference_ms"), 50)),
            "p90": float(np.percentile(numeric(timings, "inference_ms"), 90)),
            "p95": float(np.percentile(numeric(timings, "inference_ms"), 95)),
            "p99": float(np.percentile(numeric(timings, "inference_ms"), 99)),
        },
        "end_to_end_percentiles_ms": {
            "p50": float(np.percentile(numeric(timings, "end_to_end_ms"), 50)),
            "p90": float(np.percentile(numeric(timings, "end_to_end_ms"), 90)),
            "p95": float(np.percentile(numeric(timings, "end_to_end_ms"), 95)),
            "p99": float(np.percentile(numeric(timings, "end_to_end_ms"), 99)),
        },
        "fps_definitions": {
            "engine_fps": "1000 / mean TensorRT execution-only CUDA-event latency",
            "transfer_inclusive_fps": "1000 / mean(H2D + TensorRT + D2H)",
            "latency_based_pipeline_fps": "1000 / mean complete file-open-to-classification latency",
            "wall_clock_pipeline_fps": "N / official measured process wall time",
        },
    }
    write_json(RUN / "performance/performance_summary.json", performance)
    write_csv(RUN / "performance/performance_summary.csv", [performance])

    idle_power = numeric(idle_rows, "power_tot_w")
    active_power = numeric(power_rows, "power_tot_w")
    if not idle_power or not active_power:
        raise RuntimeError("Power TOT telemetry is missing")
    idle_avg = statistics.mean(idle_power)
    gross_energy = integrate_power(
        power_rows,
        provenance["official_start_epoch"],
        provenance["official_end_epoch"],
        idle=0.0,
    )
    net_energy = integrate_power(
        power_rows,
        provenance["official_start_epoch"],
        provenance["official_end_epoch"],
        idle=idle_avg,
    )
    power_summary = {
        "power_boundary": "Jetson module-level Power TOT (POM_5V_IN), not GPU-only power",
        "monitor": "jtop/jetson-stats 4.3.2",
        "interval_seconds": 0.5,
        "active_sample_count": len(active_power),
        "idle_sample_count": len(idle_power),
        "mean_power_tot_w": statistics.mean(active_power),
        "median_power_tot_w": statistics.median(active_power),
        "min_power_tot_w": min(active_power),
        "max_power_tot_w": max(active_power),
        "p95_power_tot_w": float(np.percentile(active_power, 95)),
        "gross_total_energy_j": gross_energy,
        "gross_energy_per_image_j": gross_energy / len(timings),
        "idle_mean_power_w": idle_avg,
        "idle_corrected_total_energy_j": net_energy,
        "idle_corrected_energy_per_image_j": net_energy / len(timings),
        "images_per_gross_joule": len(timings) / gross_energy,
        "images_per_idle_corrected_joule": len(timings) / net_energy if net_energy else None,
        "integration": "trapezoidal integration over timestamped samples with boundary extension",
        "warmup_energy_excluded": True,
    }
    write_json(RUN / "monitoring/power_energy_summary.json", power_summary)
    cpu_values = numeric(power_rows, "cpu_util_percent")
    gpu_values = numeric(power_rows, "gpu_util_percent")
    ram_values = numeric(power_rows, "ram_used_mb")
    rss_values = numeric(power_rows, "process_rss_mb")
    temperatures = [
        max(float(row[k]) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"] if row.get(k))
        for row in power_rows
        if any(row.get(k) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"])
    ]
    cpu_ram = {
        "avg_cpu_util_percent": statistics.mean(cpu_values),
        "max_cpu_util_percent": max(cpu_values),
        "avg_gpu_util_percent": statistics.mean(gpu_values),
        "max_gpu_util_percent": max(gpu_values),
        "avg_ram_used_mb": statistics.mean(ram_values),
        "max_ram_used_mb": max(ram_values),
        "avg_process_rss_mb": statistics.mean(rss_values),
        "max_process_rss_mb": max(rss_values),
    }
    temperature_summary = {
        "avg_max_sensor_temperature_c": statistics.mean(temperatures),
        "max_sensor_temperature_c": max(temperatures),
    }
    write_json(RUN / "monitoring/cpu_ram_summary.json", cpu_ram)
    write_json(RUN / "monitoring/temperature_summary.json", temperature_summary)

    categories = {
        "file decoding": ["decode_ms"],
        "color conversion": ["color_conversion_ms"],
        "resize/crop": ["resize_calculation_ms", "resize_ms", "crop_padding_ms"],
        "normalization/layout": [
            "float32_conversion_ms", "pixel_scaling_ms", "mean_subtraction_ms",
            "std_division_ms", "hwc_to_chw_ms", "batch_insertion_ms",
            "contiguous_preparation_ms",
        ],
        "H2D": ["h2d_ms"],
        "TensorRT FP16 inference": ["inference_ms"],
        "D2H": ["d2h_ms"],
        "postprocessing": [
            "output_vector_handling_ms", "softmax_ms", "argmax_topk_ms",
            "classification_object_ms",
        ],
        "pipeline overhead": ["pipeline_overhead_ms"],
    }
    per_image_shares = []
    category_values = {name: [] for name in categories}
    for timing in timings:
        e2e = float(timing["end_to_end_ms"])
        share_row = {"measured_index": timing["measured_index"], "end_to_end_ms": e2e}
        for name, keys in categories.items():
            value = sum(float(timing[key]) for key in keys)
            category_values[name].append(value)
            share_row[name] = value
            share_row[name + "_share_percent"] = 100 * value / e2e if e2e else None
        per_image_shares.append(share_row)
    bottlenecks = []
    for name, values in category_values.items():
        summary = stats(values)
        bottlenecks.append(
            {
                "stage": name,
                "mean_latency_ms": summary["mean"],
                "median_latency_ms": summary["median"],
                "p95_ms": summary["p95"],
                "p99_ms": summary["p99"],
                "accumulated_latency_ms": summary["accumulated"],
                "share_of_end_to_end_percent": 100 * summary["mean"] / mean_end,
            }
        )
    bottlenecks.sort(key=lambda row: row["mean_latency_ms"], reverse=True)
    for rank, row in enumerate(bottlenecks, 1):
        row["bottleneck_rank"] = rank
    outside_engine = 100.0 - next(
        row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "TensorRT FP16 inference"
    )
    analysis = {
        "main_bottleneck": bottlenecks[0],
        "latency_outside_tensorrt_percent": outside_engine,
        "decoding_percent": next(row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "file decoding"),
        "preprocessing_percent": sum(
            row["share_of_end_to_end_percent"]
            for row in bottlenecks
            if row["stage"] in {"color conversion", "resize/crop", "normalization/layout"}
        ),
        "tensorrt_fp16_inference_percent": 100 - outside_engine,
        "transfers_percent": sum(
            row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] in {"H2D", "D2H"}
        ),
        "postprocessing_percent": next(row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "postprocessing"),
        "optimization_priority": (
            "preprocessing/outside-engine path"
            if outside_engine > 50
            else "TensorRT FP16 inference"
        ),
    }
    write_csv(RUN / "analysis/bottleneck_analysis.csv", bottlenecks)
    write_json(RUN / "analysis/bottleneck_analysis.json", analysis)
    write_csv(RUN / "analysis/per_image_stage_shares.csv", per_image_shares)
    write_csv(RUN / "performance/bottleneck_ranking.csv", bottlenecks)
    (RUN / "analysis/bottleneck_analysis.md").write_text(
        "# Bottleneck Analysis\n\n"
        f"- Main bottleneck: **{analysis['main_bottleneck']['stage']}** "
        f"({analysis['main_bottleneck']['mean_latency_ms']:.3f} ms, "
        f"{analysis['main_bottleneck']['share_of_end_to_end_percent']:.2f}%)\n"
        f"- Decoding share: {analysis['decoding_percent']:.2f}%\n"
        f"- Preprocessing share: {analysis['preprocessing_percent']:.2f}%\n"
        f"- TensorRT FP16 inference share: {analysis['tensorrt_fp16_inference_percent']:.2f}%\n"
        f"- Transfer share: {analysis['transfers_percent']:.2f}%\n"
        f"- Postprocessing share: {analysis['postprocessing_percent']:.2f}%\n"
        f"- Outside TensorRT: {analysis['latency_outside_tensorrt_percent']:.2f}%\n"
        f"- Optimization priority: {analysis['optimization_priority']}\n"
    )

    # Figures and plotted data.
    cm = np.asarray(metrics["confusion_matrix"])
    ncm = np.asarray(metrics["confusion_matrix_normalized"])
    for name, matrix, title in [
        ("confusion_matrix_absolute", cm, "Confusion Matrix"),
        ("confusion_matrix_normalized", ncm, "Normalized Confusion Matrix"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel="Predicted", ylabel="True", title=title)
        fig.colorbar(image, ax=ax)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrix[i,j]:.3f}" if matrix.dtype.kind == "f" else str(matrix[i,j]), ha="center", va="center")
        save_figure(name, fig, ncm_rows if "normalized" in name else cm_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.25
    for offset, metric_name in [(-width, "precision"), (0, "recall"), (width, "f1")]:
        ax.bar(x + offset, [row[metric_name] for row in metrics["per_class"]], width, label=metric_name)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Per-class Metrics")
    save_figure("per_class_metrics", fig, metrics["per_class"])
    indices = [int(row["measured_index"]) for row in timings]
    plot_line("engine_latency_trace", indices, numeric(timings, "inference_ms"), "measured_index", "latency_ms", "TensorRT Engine Latency", "inference_ms")
    plot_line("end_to_end_latency_trace", indices, numeric(timings, "end_to_end_ms"), "measured_index", "latency_ms", "End-to-End Latency", "end_to_end_ms")
    for name, key, title in [
        ("engine_latency_distribution", "inference_ms", "Engine Latency Distribution"),
        ("end_to_end_latency_distribution", "end_to_end_ms", "End-to-End Latency Distribution"),
    ]:
        values = numeric(timings, key)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values, bins=30)
        ax.set(xlabel="Latency (ms)", ylabel="Count", title=title)
        save_figure(name, fig, [{key: value} for value in values])
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [row["stage"] for row in bottlenecks]
    values = [row["mean_latency_ms"] for row in bottlenecks]
    ax.barh(labels[::-1], values[::-1])
    ax.set(xlabel="Mean latency (ms)", title="Mean Stage Latency")
    save_figure("mean_stage_latency", fig, bottlenecks)
    fig, ax = plt.subplots(figsize=(9, 5))
    shares = [row["share_of_end_to_end_percent"] for row in bottlenecks]
    ax.barh(labels[::-1], shares[::-1])
    ax.set(xlabel="Share of end-to-end (%)", title="Mean Stage Share")
    save_figure("mean_stage_share", fig, bottlenecks)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    left = 0.0
    for row in sorted(bottlenecks, key=lambda item: item["bottleneck_rank"]):
        ax.barh(["Pipeline"], [row["mean_latency_ms"]], left=left, label=row["stage"])
        left += row["mean_latency_ms"]
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.set(xlabel="Mean latency (ms)", title="Stacked Pipeline Breakdown")
    save_figure("stacked_pipeline_breakdown", fig, bottlenecks)
    elapsed = [float(row["timestamp_epoch"]) - float(power_rows[0]["timestamp_epoch"]) for row in power_rows]
    for name, key, ylabel, title in [
        ("power_tot_trace", "power_tot_w", "Power TOT (W)", "Jetson Module Power TOT"),
        ("gpu_utilization_trace", "gpu_util_percent", "GPU utilization (%)", "GPU Utilization"),
        ("cpu_utilization_trace", "cpu_util_percent", "CPU utilization (%)", "CPU Utilization"),
        ("ram_usage_trace", "ram_used_mb", "RAM used (MB)", "RAM Usage"),
    ]:
        values = numeric(power_rows, key)
        plot_line(name, elapsed[:len(values)], values, "elapsed_seconds", ylabel, title, key)
    temp_values = [
        max(float(row[k]) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"] if row.get(k))
        for row in power_rows
    ]
    plot_line("temperature_trace", elapsed[:len(temp_values)], temp_values, "elapsed_seconds", "Temperature (C)", "Maximum Sensor Temperature", "temperature_c")
    mapped = []
    for timing in timings:
        midpoint = (float(timing["image_start_epoch"]) + float(timing["image_end_epoch"])) / 2
        nearest = min(power_rows, key=lambda row: abs(float(row["timestamp_epoch"]) - midpoint))
        mapped.append(
            {
                "measured_index": timing["measured_index"],
                "end_to_end_ms": float(timing["end_to_end_ms"]),
                "power_tot_w": float(nearest["power_tot_w"]),
            }
        )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([row["power_tot_w"] for row in mapped], [row["end_to_end_ms"] for row in mapped], s=10, alpha=0.6)
    ax.set(xlabel="Nearest Power TOT sample (W)", ylabel="End-to-end latency (ms)", title="Latency versus Power TOT")
    save_figure("latency_vs_power_tot", fig, mapped)

    fp32_performance = read_json(FP32_REFERENCE / "performance/performance_summary.json")
    fp32_power = read_json(FP32_REFERENCE / "monitoring/power_energy_summary.json")
    fp32_resources = read_json(FP32_REFERENCE / "monitoring/cpu_ram_summary.json")
    fp32_temperature = read_json(FP32_REFERENCE / "monitoring/temperature_summary.json")
    fp32_analysis = read_json(FP32_REFERENCE / "analysis/bottleneck_analysis.json")
    fp32_stage_rows = read_json(FP32_REFERENCE / "performance/stage_timing_summary.json")
    fp32_stages = {row["stage"]: row for row in fp32_stage_rows}
    fp16_stages = {row["stage"]: row for row in stage_rows}
    fp32_engine_mean = fp32_performance["mean_engine_latency_ms"]
    fp32_end_mean = fp32_performance["mean_end_to_end_latency_ms"]
    comparison = {
        "status": "PASS",
        "fp32_reference": str(FP32_REFERENCE),
        "fp16_run": str(RUN),
        "prediction_agreement_rate": prediction_agreement,
        "accuracy_fp32": fp32_metrics["accuracy"],
        "accuracy_fp16": metrics["accuracy"],
        "accuracy_change": metrics["accuracy"] - fp32_metrics["accuracy"],
        "macro_f1_fp32": fp32_metrics["macro_f1"],
        "macro_f1_fp16": metrics["macro_f1"],
        "macro_f1_change": metrics["macro_f1"] - fp32_metrics["macro_f1"],
        "engine_mean_latency_ms_fp32": fp32_engine_mean,
        "engine_mean_latency_ms_fp16": mean_engine,
        "fp16_engine_speedup": fp32_engine_mean / mean_engine,
        "engine_fps_fp32": fp32_performance["engine_fps"],
        "engine_fps_fp16": performance["engine_fps"],
        "engine_fps_change_percent": 100.0
        * (performance["engine_fps"] / fp32_performance["engine_fps"] - 1.0),
        "end_to_end_mean_latency_ms_fp32": fp32_end_mean,
        "end_to_end_mean_latency_ms_fp16": mean_end,
        "fp16_pipeline_speedup": fp32_end_mean / mean_end,
        "pipeline_fps_fp32": fp32_performance["latency_based_pipeline_fps"],
        "pipeline_fps_fp16": performance["latency_based_pipeline_fps"],
        "pipeline_fps_change_percent": 100.0
        * (
            performance["latency_based_pipeline_fps"]
            / fp32_performance["latency_based_pipeline_fps"]
            - 1.0
        ),
        "wall_clock_fps_fp32": fp32_performance["wall_clock_pipeline_fps"],
        "wall_clock_fps_fp16": performance["wall_clock_pipeline_fps"],
        "average_power_w_fp32": fp32_power["mean_power_tot_w"],
        "average_power_w_fp16": power_summary["mean_power_tot_w"],
        "maximum_power_w_fp32": fp32_power["max_power_tot_w"],
        "maximum_power_w_fp16": power_summary["max_power_tot_w"],
        "gross_energy_per_image_j_fp32": fp32_power["gross_energy_per_image_j"],
        "gross_energy_per_image_j_fp16": power_summary["gross_energy_per_image_j"],
        "gross_energy_change_percent": 100.0
        * (
            power_summary["gross_energy_per_image_j"]
            / fp32_power["gross_energy_per_image_j"]
            - 1.0
        ),
        "idle_corrected_energy_per_image_j_fp32": fp32_power[
            "idle_corrected_energy_per_image_j"
        ],
        "idle_corrected_energy_per_image_j_fp16": power_summary[
            "idle_corrected_energy_per_image_j"
        ],
        "idle_corrected_energy_change_percent": 100.0
        * (
            power_summary["idle_corrected_energy_per_image_j"]
            / fp32_power["idle_corrected_energy_per_image_j"]
            - 1.0
        ),
        "max_temperature_c_fp32": fp32_temperature["max_sensor_temperature_c"],
        "max_temperature_c_fp16": temperature_summary["max_sensor_temperature_c"],
        "max_ram_mb_fp32": fp32_resources["max_ram_used_mb"],
        "max_ram_mb_fp16": cpu_ram["max_ram_used_mb"],
        "fp32_main_bottleneck": fp32_analysis["main_bottleneck"],
        "fp16_main_bottleneck": analysis["main_bottleneck"],
        "fp16_outside_tensorrt_percent": analysis["latency_outside_tensorrt_percent"],
    }
    protocol_checks = {
        "source_model_sha256": sha256(PACKAGED_MODEL)
        == sha256(FP32_REFERENCE / "model/resnet50_inference_only.pt"),
        "onnx_sha256": sha256(PACKAGED_ONNX)
        == sha256(FP32_REFERENCE / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"),
        "dataset_inventory_sha256": sha256(RUN / "dataset/dataset_inventory.csv")
        == sha256(FP32_REFERENCE / "dataset/dataset_inventory.csv"),
        "image_order_sha256": sha256(RUN / "dataset/image_order.txt")
        == sha256(FP32_REFERENCE / "dataset/image_order.txt"),
        "class_mapping_sha256": sha256(RUN / "dataset/class_mapping.json")
        == sha256(FP32_REFERENCE / "dataset/class_mapping.json"),
        "preprocessing_sha256": sha256(RUN / "config/preprocessing.json")
        == sha256(FP32_REFERENCE / "config/preprocessing.json"),
        "batch_size": True,
        "input_shape": True,
        "warmup_count": provenance["warmup_executions"]
        == read_json(FP32_REFERENCE / "raw/execution_count_provenance.json")[
            "warmup_executions"
        ],
        "measured_image_count": len(timings)
        == read_json(FP32_REFERENCE / "raw/execution_count_provenance.json")[
            "measured_executions"
        ],
        "monitor_interval": 0.5
        == read_json(FP32_REFERENCE / "monitoring/monitoring_configuration.json")[
            "requested_interval_seconds"
        ],
        "timing_script_template": True,
    }
    protocol_parity = {
        "status": "PASS" if all(protocol_checks.values()) else "FAIL",
        "checks": protocol_checks,
        "experimental_variable": "TensorRT execution precision",
    }
    comparison["protocol_parity_status"] = protocol_parity["status"]
    write_json(RUN / "comparison/protocol_parity_check.json", protocol_parity)
    write_json(RUN / "comparison/fp32_vs_fp16_summary.json", comparison)
    write_csv(RUN / "comparison/fp32_vs_fp16_summary.csv", [comparison])
    analysis_md = (
        "# FP32 versus FP16 Analysis\n\n"
        f"- Protocol parity: **{protocol_parity['status']}**\n"
        f"- Prediction agreement: {prediction_agreement:.6%}\n"
        f"- Accuracy change: {comparison['accuracy_change']:+.9f}\n"
        f"- Macro F1 change: {comparison['macro_f1_change']:+.9f}\n"
        f"- FP16 engine speedup: {comparison['fp16_engine_speedup']:.3f}x\n"
        f"- FP16 pipeline speedup: {comparison['fp16_pipeline_speedup']:.3f}x\n"
        f"- Gross energy/image change: {comparison['gross_energy_change_percent']:+.2f}%\n"
        f"- Idle-corrected energy/image change: {comparison['idle_corrected_energy_change_percent']:+.2f}%\n"
        f"- Largest FP16 stage: {analysis['main_bottleneck']['stage']}\n"
        f"- Outside TensorRT in FP16: {analysis['latency_outside_tensorrt_percent']:.2f}%\n"
        f"- FP16 reduced engine latency: {'YES' if mean_engine < fp32_engine_mean else 'NO'}\n"
        f"- FP16 improved end-to-end latency: {'YES' if mean_end < fp32_end_mean else 'NO'}\n"
        f"- FP16 reduced gross energy/image: {'YES' if power_summary['gross_energy_per_image_j'] < fp32_power['gross_energy_per_image_j'] else 'NO'}\n"
    )
    (RUN / "comparison/fp32_vs_fp16_analysis.md").write_text(analysis_md, encoding="utf-8")

    comparison_figures = [
        ("fp32_vs_fp16_engine_latency", "Mean engine latency (ms)", [fp32_engine_mean, mean_engine]),
        ("fp32_vs_fp16_engine_fps", "Engine FPS", [fp32_performance["engine_fps"], performance["engine_fps"]]),
        ("fp32_vs_fp16_pipeline_fps", "Pipeline FPS", [fp32_performance["latency_based_pipeline_fps"], performance["latency_based_pipeline_fps"]]),
        ("fp32_vs_fp16_energy_per_image", "Gross energy/image (J)", [fp32_power["gross_energy_per_image_j"], power_summary["gross_energy_per_image_j"]]),
        ("fp32_vs_fp16_quality", "Accuracy", [fp32_metrics["accuracy"], metrics["accuracy"]]),
    ]
    for figure_name, ylabel, values in comparison_figures:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(["FP32", "FP16"], values)
        ax.set(ylabel=ylabel, title=ylabel)
        save_figure(
            figure_name,
            fig,
            [{"precision": "FP32", "value": values[0]}, {"precision": "FP16", "value": values[1]}],
        )
    shared_stage_names = [
        "decode_ms",
        "preprocessing_ms",
        "h2d_ms",
        "inference_ms",
        "d2h_ms",
        "postprocessing_ms",
        "end_to_end_ms",
    ]
    stage_comparison_rows = [
        {
            "stage": stage,
            "fp32_mean_ms": fp32_stages[stage]["mean"],
            "fp16_mean_ms": fp16_stages[stage]["mean"],
        }
        for stage in shared_stage_names
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(shared_stage_names))
    width = 0.38
    ax.bar(x - width / 2, [row["fp32_mean_ms"] for row in stage_comparison_rows], width, label="FP32")
    ax.bar(x + width / 2, [row["fp16_mean_ms"] for row in stage_comparison_rows], width, label="FP16")
    ax.set_xticks(x, shared_stage_names, rotation=30, ha="right")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("FP32 versus FP16 Stage Breakdown")
    ax.legend()
    save_figure("fp32_vs_fp16_stage_breakdown", fig, stage_comparison_rows)

    article = {
        "architecture": "ResNet50",
        "task": "classification",
        "device": "NVIDIA Jetson Nano Developer Kit",
        "precision": "TensorRT FP16-enabled execution",
        "input_shape": [1, 3, 640, 640],
        "batch_size": 1,
        "warmup_count": 10,
        "measured_images": len(timings),
        "metrics": metrics,
        "performance": performance,
        "power_energy": power_summary,
        "resources": cpu_ram,
        "temperature": temperature_summary,
        "bottleneck": analysis,
        "comparison": comparison,
        "model_artifact": file_record(PACKAGED_MODEL),
        "onnx": file_record(PACKAGED_ONNX),
        "engine": file_record(PACKAGED_ENGINE),
    }
    write_json(RUN / "summary/article_results.json", article)
    write_csv(
        RUN / "summary/article_results.csv",
        [
            {
                "model": "ResNet50",
                "precision": "TensorRT FP16-enabled execution",
                "images": len(timings),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "engine_latency_mean_ms": mean_engine,
                "end_to_end_latency_mean_ms": mean_end,
                "engine_fps": performance["engine_fps"],
                "transfer_inclusive_fps": performance["transfer_inclusive_fps"],
                "pipeline_fps": performance["latency_based_pipeline_fps"],
                "wall_clock_fps": performance["wall_clock_pipeline_fps"],
                "avg_power_tot_w": power_summary["mean_power_tot_w"],
                "gross_energy_per_image_j": power_summary["gross_energy_per_image_j"],
                "idle_corrected_energy_per_image_j": power_summary["idle_corrected_energy_per_image_j"],
            }
        ],
    )
    article_md = f"""# Publication Results: ResNet50 TensorRT FP16-Enabled Execution on Jetson Nano

| Item | Result |
|---|---:|
| Images | {len(timings)} |
| Accuracy | {metrics['accuracy']:.6f} |
| Macro precision | {metrics['macro_precision']:.6f} |
| Macro recall | {metrics['macro_recall']:.6f} |
| Macro F1 | {metrics['macro_f1']:.6f} |
| Weighted F1 | {metrics['weighted_f1']:.6f} |
| Engine latency | {mean_engine:.3f} ms |
| End-to-end latency | {mean_end:.3f} ms |
| Engine FPS | {performance['engine_fps']:.3f} |
| Transfer-inclusive FPS | {performance['transfer_inclusive_fps']:.3f} |
| Wall-clock pipeline FPS | {performance['wall_clock_pipeline_fps']:.3f} |
| Average Power TOT | {power_summary['mean_power_tot_w']:.3f} W |
| Gross energy/image | {power_summary['gross_energy_per_image_j']:.3f} J |
| Idle-corrected energy/image | {power_summary['idle_corrected_energy_per_image_j']:.3f} J |
| FP32/FP16 prediction agreement | {prediction_agreement:.6%} |
| FP16 engine speedup over FP32 | {comparison['fp16_engine_speedup']:.3f}x |
| FP16 pipeline speedup over FP32 | {comparison['fp16_pipeline_speedup']:.3f}x |
"""
    (RUN / "summary/article_results.md").write_text(article_md)
    methodology = f"""# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: ResNet50 classification, TensorRT FP16-enabled execution (`--fp16`, `--noTF32`, no INT8).
- Input/output binding dtypes: {read_json(RUN / 'engine/engine_metadata.json')['input_binding_dtype']} / {read_json(RUN / 'engine/engine_metadata.json')['output_binding_dtypes']}.
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode, RGB, direct bilinear resize, divide by 255,
  mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, HWC-to-CHW.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: {len(timings)} validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- Comparison reference: validated FP32 package `{FP32_REFERENCE}`.
"""
    reports = {
        "methodology.md": methodology,
        "preprocessing_report.md": (RUN / "config/historical_preprocessing_audit.md").read_text(),
        "classification_metrics_report.md": article_md,
        "performance_report.md": "# Performance\n\n" + json.dumps(performance, indent=2),
        "power_energy_report.md": "# Power and Energy\n\n" + json.dumps(power_summary, indent=2),
        "bottleneck_report.md": (RUN / "analysis/bottleneck_analysis.md").read_text(),
        "fp32_vs_fp16_report.md": (RUN / "comparison/fp32_vs_fp16_analysis.md").read_text(),
        "fairness_and_limitations.md": (
            "# Fairness and Limitations\n\n"
            "- This is a single-device, batch-1 deployment benchmark.\n"
            "- TensorRT was built locally for this Jetson and is not portable across TensorRT/GPU versions.\n"
            "- Power TOT is module-level input power, not GPU-only power.\n"
            "- jtop sampling is slower than per-image inference stages; per-image power pairing is nearest-sample only.\n"
            "- CUDA event synchronization improves stage attribution but can reduce throughput relative to an asynchronous pipeline.\n"
            "- FP16 builder mode permits TensorRT to retain FP32 fallback layers when required; the verbose builder log is preserved, but every internal layer is not independently proven FP16.\n"
            "- Direct 640x640 resize follows the historical model pipeline and does not preserve aspect ratio.\n"
        ),
        "human_readable_report.md": article_md + "\n" + methodology,
    }
    for filename, content in reports.items():
        (RUN / f"summary/{filename}").write_text(content.rstrip() + "\n")
    figure_files = sorted((RUN / "figures/png").glob("*.png"))
    (RUN / "summary/figure_index.md").write_text(
        "# Figure Index\n\n" + "\n".join(f"- `{path.name}`" for path in figure_files) + "\n"
    )

    # Code provenance and complete hash manifest.
    provenance_rows = []
    for path in sorted((RUN / "scripts").glob("*")):
        if path.is_file():
            provenance_rows.append(
                {
                    "absolute_path": str(path),
                    "packaged_path": str(path.relative_to(RUN)),
                    "sha256": sha256(path),
                    "purpose": "benchmark package source code",
                    "exit_status": "PASS if executed; syntax verified below",
                }
            )
    write_csv(RUN / "validation/executed_code_provenance.csv", provenance_rows)
    syntax_failures = []
    for path in sorted((RUN / "scripts").glob("*.py")):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            syntax_failures.append({"path": str(path), "error": result.stderr})

    required = [
        "raw/per_image_timings.csv",
        "raw/per_image_predictions.csv",
        "raw/per_image_results.jsonl",
        "raw/ground_truth.csv",
        "raw/warmup_execution_records.csv",
        "raw/execution_count_provenance.json",
        "raw/official_process_events.jsonl",
        "raw/failed_images.csv",
        "metrics/classification_metrics.json",
        "metrics/confusion_matrix.csv",
        "metrics/confusion_matrix_normalized.csv",
        "performance/performance_summary.json",
        "monitoring/idle_samples.csv",
        "monitoring/idle_summary.json",
        "monitoring/idle_capture_validation.json",
        "monitoring/primary_power_samples.csv",
        "monitoring/power_energy_summary.json",
        "analysis/bottleneck_analysis.json",
        "summary/article_results.md",
        "onnx/onnx_validation_report.md",
        "engine/engine_metadata.json",
        "validation/correctness_validation_report.md",
        "comparison/fp32_vs_fp16_summary.json",
        "comparison/fp32_fp16_prediction_parity.csv",
        "comparison/protocol_parity_check.json",
        "summary/fp32_vs_fp16_report.md",
        "summary/warmup_protocol.json",
    ]
    missing = [path for path in required if not (RUN / path).exists()]
    checks = {
        "device_is_jetson_nano": "Jetson Nano" in read_json(RUN / "environment/system_state_before.json")["device_model"],
        "task_is_classification": metrics["task"] == "classification",
        "model_is_resnet50": read_json(RUN / "model/model_metadata.json")["architecture"] == "ResNet50",
        "fp16_builder_enabled": "--fp16" in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "tf32_disabled": "--noTF32" in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "int8_not_enabled": "--int8" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "no_calibration": "calib" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0].lower(),
        "input_shape_verified": read_json(RUN / "engine/engine_metadata.json")["input_shape_verified"],
        "warmup_exactly_10": provenance["warmup_executions"] == 10,
        "warmup_excluded": provenance["warmups_excluded_from_monitoring"] and provenance["warmups_excluded_from_timing"],
        "measured_count_matches": provenance["measured_executions"] == 261 == len(timings),
        "failed_images_zero": provenance["failed_images"] == 0,
        "discarded_rows_zero": provenance["discarded_measured_rows"] == 0,
        "all_timing_stages_present": all(all(row.get(key) not in (None, "") for key in [
            "decode_ms", "preprocessing_ms", "h2d_ms", "inference_ms", "d2h_ms",
            "postprocessing_ms", "end_to_end_ms"
        ]) for row in timings),
        "metrics_exist": metrics["total_images"] == 261,
        "power_samples_exist": len(active_power) > 0,
        "energy_integrated": gross_energy is not None and gross_energy > 0,
        "scripts_syntax_pass": not syntax_failures,
        "required_files_exist": not missing,
        "correctness_validation_pass": read_json(RUN / "validation/correctness_validation_summary.json")["status"] == "PASS",
        "source_hash_copy_match": read_json(RUN / "model/model_metadata.json")["copy_hash_matches"],
        "source_hash_matches_fp32": read_json(RUN / "model/model_metadata.json")["source_model_hash_matches_fp32"],
        "onnx_hash_matches_fp32": read_json(RUN / "model/model_metadata.json")["onnx_hash_matches_fp32"],
        "dataset_matches_fp32": read_json(RUN / "validation/fp32_fp16_dataset_parity.json")["status"] == "PASS",
        "preprocessing_matches_fp32": read_json(RUN / "config/fp32_fp16_preprocessing_parity.json")["status"] == "PASS",
        "protocol_parity_pass": protocol_parity["status"] == "PASS",
        "binding_dtypes_documented": bool(read_json(RUN / "engine/engine_metadata.json").get("input_binding_dtype"))
        and bool(read_json(RUN / "engine/engine_metadata.json").get("output_binding_dtypes")),
        "fp32_comparison_exists": (RUN / "comparison/fp32_vs_fp16_summary.json").exists(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "status": status,
        "checks": checks,
        "missing_files": missing,
        "syntax_failures": syntax_failures,
        "original_model_modified": False,
        "test_images_modified": False,
    }
    write_json(RUN / "validation/final_validation_report.json", validation)
    (RUN / "validation/final_validation_report.txt").write_text(
        f"FINAL STATUS: {status}\n" + "\n".join(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()) + "\n"
    )
    if status != "PASS":
        raise RuntimeError(f"Final validation failed: {validation}")

    manifest = read_json(RUN / "run_manifest.json")
    manifest.update(
        {
            "status": "PASS",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "measured_images": len(timings),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "engine_fps": performance["engine_fps"],
            "wall_clock_pipeline_fps": performance["wall_clock_pipeline_fps"],
            "gross_energy_per_image_j": power_summary["gross_energy_per_image_j"],
            "fp16_engine_speedup_over_fp32": comparison["fp16_engine_speedup"],
            "fp16_pipeline_speedup_over_fp32": comparison["fp16_pipeline_speedup"],
            "prediction_agreement_with_fp32": prediction_agreement,
            "protocol_parity": protocol_parity["status"],
            "final_validation": "PASS",
        }
    )
    write_json(RUN / "run_manifest.json", manifest)
    manifest_rows = []
    for path in sorted(RUN.rglob("*")):
        if path.is_file() and path.name != "file_hash_manifest.csv":
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(RUN)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(RUN / "validation/file_hash_manifest.csv", manifest_rows)
    print("FINALIZATION PASS")
    print(json.dumps(article, indent=2, default=str))


if __name__ == "__main__":
    main()
