#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="$RUN/onnx/resnet50_classification_fp32_1x3x640x640.onnx"
ENGINE="$RUN/engine/resnet50_classification_tensorrt_fp16_1x3x640x640.engine"
LOG="$RUN/engine/engine_build_log.txt"
STDOUT="$RUN/logs/engine_build_stdout.log"
STDERR="$RUN/logs/engine_build_stderr.log"
CACHE="$RUN/engine/timing_cache.bin"
if [[ -e "$ENGINE" ]]; then
  echo "Refusing to overwrite existing engine: $ENGINE" >&2
  exit 3
fi
cat > "$RUN/engine/engine_build_config.json" <<EOF
{
  "builder": "/usr/src/tensorrt/bin/trtexec",
  "source_onnx": "$ONNX",
  "output_engine": "$ENGINE",
  "workspace_mb": 1024,
  "input_shape": [1, 3, 640, 640],
  "batch_size": 1,
  "fp16": true,
  "int8": false,
  "tf32": false,
  "dla": false,
  "dynamic_shapes": false,
  "input_io_format": "fp32:chw",
  "output_io_format": "fp32:chw",
  "build_only": true,
  "timing_cache": "$CACHE"
}
EOF
CMD=(/usr/src/tensorrt/bin/trtexec
  "--onnx=$ONNX"
  "--saveEngine=$ENGINE"
  "--workspace=1024"
  "--fp16"
  "--noTF32"
  "--inputIOFormats=fp32:chw"
  "--outputIOFormats=fp32:chw"
  "--timingCacheFile=$CACHE"
  "--verbose"
  "--buildOnly")
printf '%q ' "${CMD[@]}" > "$LOG"
printf '\n' >> "$LOG"
printf '%q ' "${CMD[@]}" >> "$RUN/logs/executed_commands.log"
printf '\n' >> "$RUN/logs/executed_commands.log"
"${CMD[@]}" > >(tee -a "$STDOUT" "$LOG") 2> >(tee -a "$STDERR" "$LOG" >&2)
test -s "$ENGINE"
sha256sum "$ENGINE" > "$RUN/engine/engine_sha256.txt"
grep -Ei 'fp16|half|precision|fallback|layer|warning' "$LOG" \
  > "$RUN/engine/layer_precision_report.txt" || true
stat "$ENGINE" > "$RUN/engine/engine_file_stat.txt"
echo "ENGINE BUILD PASS"
