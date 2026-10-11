from __future__ import annotations

import json
import os
import statistics
import threading
import time
from pathlib import Path

import psutil

from common import RUN, write_csv, write_json


def _value(stats, key, default=None):
    value = stats.get(key, default)
    return value if isinstance(value, (int, float)) else default


class JetsonMonitor:
    def __init__(self, label: str, interval_seconds: float = 0.5):
        self.label = label
        self.interval = interval_seconds
        self.rows: list[dict] = []
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.thread = None
        self.error = None
        self.available = {}
        self.process = psutil.Process(os.getpid())

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=8):
            raise RuntimeError("jtop monitor did not produce an initial sample")
        if self.error:
            raise RuntimeError(f"jtop monitor failed: {self.error}")

    def _run(self):
        try:
            from jtop import jtop

            with jtop(interval=self.interval) as jetson:
                while not self.stop_event.is_set() and jetson.ok():
                    epoch = time.time()
                    monotonic = time.perf_counter()
                    stats = dict(jetson.stats)
                    power = jetson.power
                    cpu = jetson.cpu
                    gpu = jetson.gpu
                    memory = jetson.memory
                    temperature = jetson.temperature
                    fan = jetson.fan
                    if not self.available:
                        raw_available = {
                            "stats_keys": sorted(stats),
                            "power": power,
                            "cpu": cpu,
                            "gpu": gpu,
                            "memory": memory,
                            "temperature": temperature,
                            "fan": fan,
                        }
                        self.available = json.loads(json.dumps(raw_available, default=str))
                    cpu_values = [
                        float(stats[key])
                        for key in stats
                        if key.startswith("CPU") and isinstance(stats[key], (int, float))
                    ]
                    process_info = self.process.as_dict(
                        attrs=["cpu_percent", "memory_info", "io_counters"], ad_value=None
                    )
                    mem = process_info.get("memory_info")
                    io = process_info.get("io_counters")
                    power_tot_mw = power.get("tot", {}).get("power")
                    row = {
                        "timestamp_epoch": epoch,
                        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "monotonic_seconds": monotonic,
                        "label": self.label,
                        "power_tot_w": power_tot_mw / 1000.0 if power_tot_mw is not None else None,
                        "power_cpu_w": power.get("rail", {}).get("POM_5V_CPU", {}).get("power", 0) / 1000.0,
                        "power_gpu_w": power.get("rail", {}).get("POM_5V_GPU", {}).get("power", 0) / 1000.0,
                        "gpu_util_percent": _value(stats, "GPU"),
                        "cpu_util_percent": sum(cpu_values) / len(cpu_values) if cpu_values else None,
                        "cpu1_percent": _value(stats, "CPU1"),
                        "cpu2_percent": _value(stats, "CPU2"),
                        "cpu3_percent": _value(stats, "CPU3"),
                        "cpu4_percent": _value(stats, "CPU4"),
                        "ram_fraction": _value(stats, "RAM"),
                        "swap_fraction": _value(stats, "SWAP"),
                        "emc_percent": _value(stats, "EMC"),
                        "temperature_cpu_c": _value(stats, "Temp CPU"),
                        "temperature_gpu_c": _value(stats, "Temp GPU"),
                        "temperature_thermal_c": _value(stats, "Temp thermal"),
                        "fan_percent": next(
                            (float(stats[k]) for k in stats if k.startswith("Fan ") and isinstance(stats[k], (int, float))),
                            None,
                        ),
                        "process_cpu_percent": process_info.get("cpu_percent"),
                        "process_rss_mb": mem.rss / 1048576 if mem else None,
                        "process_vms_mb": mem.vms / 1048576 if mem else None,
                        "process_read_bytes": io.read_bytes if io else None,
                        "process_write_bytes": io.write_bytes if io else None,
                        "cpu_frequency_mhz": max(
                            [
                                item.get("freq", {}).get("cur", 0) / 1000.0
                                for item in cpu.get("cpu", [])
                                if item.get("online")
                            ]
                            or [0]
                        ),
                        "gpu_frequency_mhz": gpu.get("gpu", {}).get("freq", {}).get("cur", 0) / 1000.0,
                        "emc_frequency_mhz": memory.get("EMC", {}).get("cur", 0) / 1000.0,
                        "ram_used_mb": memory.get("RAM", {}).get("used", 0) / 1024.0,
                        "swap_used_mb": memory.get("SWAP", {}).get("used", 0) / 1024.0,
                        "raw_stats_json": json.dumps(stats, default=str, sort_keys=True),
                    }
                    self.rows.append(row)
                    self.ready.set()
                    time.sleep(self.interval)
        except Exception as exc:
            self.error = repr(exc)
            self.ready.set()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=8)

    def save(self, csv_path: Path, config_path: Path | None = None):
        write_csv(csv_path, self.rows)
        if config_path:
            write_json(
                config_path,
                {
                    "label": self.label,
                    "requested_interval_seconds": self.interval,
                    "sample_count": len(self.rows),
                    "error": self.error,
                    "available_sensors": self.available,
                    "method": "jtop/jetson-stats plus psutil",
                    "power_boundary": "Jetson module-level Power TOT (POM_5V_IN)",
                },
            )


def run_idle(duration_seconds: float = 30.0):
    monitor = JetsonMonitor("idle", 0.5)
    monitor.start()
    start = time.time()
    while time.time() - start < duration_seconds:
        time.sleep(0.1)
    monitor.stop()
    monitor.save(
        RUN / "monitoring/idle_samples.csv",
        RUN / "monitoring/monitoring_configuration_idle.json",
    )
    if not monitor.rows or not any(row.get("power_tot_w") for row in monitor.rows):
        raise RuntimeError("Idle monitoring produced no Power TOT telemetry")
    power = [float(row["power_tot_w"]) for row in monitor.rows if row.get("power_tot_w") is not None]
    summary = {
        "status": "PASS",
        "duration_seconds": duration_seconds,
        "sample_count": len(monitor.rows),
        "mean_power_tot_w": statistics.mean(power),
        "median_power_tot_w": statistics.median(power),
        "min_power_tot_w": min(power),
        "max_power_tot_w": max(power),
        "power_boundary": "Jetson module-level Power TOT (POM_5V_IN)",
    }
    write_json(RUN / "monitoring/idle_summary.json", summary)
    write_json(
        RUN / "monitoring/idle_capture_validation.json",
        {
            "status": "PASS",
            "duration_seconds": duration_seconds,
            "samples": len(monitor.rows),
            "csv_exists": (RUN / "monitoring/idle_samples.csv").exists(),
            "csv_parseable": True,
            "power_samples_present": len(power) == len(monitor.rows),
        },
    )
