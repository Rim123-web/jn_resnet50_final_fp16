from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


RUN = Path(__file__).resolve().parents[1]
ONNX_PATH = RUN / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"


def shape(value):
    return [dimension.dim_value if dimension.dim_value else dimension.dim_param for dimension in value.type.tensor_type.shape.dim]


model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(model)
inputs = [
    {"name": value.name, "shape": shape(value), "elem_type": value.type.tensor_type.elem_type}
    for value in model.graph.input
]
outputs = [
    {"name": value.name, "shape": shape(value), "elem_type": value.type.tensor_type.elem_type}
    for value in model.graph.output
]
if len(inputs) != 1 or inputs[0]["shape"] != [1, 3, 640, 640]:
    raise RuntimeError(f"Unexpected ONNX input: {inputs}")
if len(outputs) != 1 or outputs[0]["shape"] != [1, 3]:
    raise RuntimeError(f"Unexpected ONNX output: {outputs}")
session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
result = session.run(None, {session.get_inputs()[0].name: dummy})[0]
if list(result.shape) != [1, 3]:
    raise RuntimeError(f"ONNX Runtime output shape mismatch: {result.shape}")
metadata = {
    "status": "PASS",
    "onnx_version": onnx.__version__,
    "onnxruntime_version": ort.__version__,
    "providers": session.get_providers(),
    "opset": [item.version for item in model.opset_import],
    "inputs": inputs,
    "outputs": outputs,
    "node_count": len(model.graph.node),
    "dummy_output_shape": list(result.shape),
    "dynamic_shapes": False,
    "embedded_preprocessing": False,
}
(RUN / "onnx/onnx_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(RUN / "onnx/onnx_graph_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
(RUN / "onnx/onnx_validation_report.md").write_text(
    "# ONNX Validation\n\n"
    "- onnx.checker: PASS\n"
    "- Fixed FP32 input: [1,3,640,640]\n"
    "- FP32 output: [1,3]\n"
    "- ONNX Runtime CPU dummy inference: PASS\n"
)
import hashlib
digest = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()
(RUN / "onnx/onnx_sha256.txt").write_text(f"{digest}  {ONNX_PATH.name}\n")
print("ONNX VALIDATION PASS")
