# Performance

{
  "measured_images": 261,
  "warmup_count": 10,
  "mean_engine_latency_ms": 187.14831852821555,
  "mean_transfer_inclusive_latency_ms": 189.94654054077648,
  "mean_end_to_end_latency_ms": 273.36316818390804,
  "engine_fps": 5.343355515370202,
  "transfer_inclusive_fps": 5.264639182966992,
  "latency_based_pipeline_fps": 3.658137292757886,
  "official_measured_wall_time_seconds": 71.45280476100015,
  "wall_clock_pipeline_fps": 3.652760740085841,
  "engine_percentiles_ms": {
    "p50": 186.4278106689453,
    "p90": 189.09776306152344,
    "p95": 189.67739868164062,
    "p99": 190.47097778320312
  },
  "end_to_end_percentiles_ms": {
    "p50": 268.931046,
    "p90": 312.2317,
    "p95": 344.37051,
    "p99": 386.05907659999997
  },
  "fps_definitions": {
    "engine_fps": "1000 / mean TensorRT execution-only CUDA-event latency",
    "transfer_inclusive_fps": "1000 / mean(H2D + TensorRT + D2H)",
    "latency_based_pipeline_fps": "1000 / mean complete file-open-to-classification latency",
    "wall_clock_pipeline_fps": "N / official measured process wall time"
  }
}
