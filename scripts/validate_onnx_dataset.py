from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


RUN = Path(__file__).resolve().parents[1]
ONNX = RUN / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"
order = (RUN / "dataset/image_order.txt").read_text().splitlines()
config = json.loads((RUN / "config/preprocessing.json").read_text())
indices = sorted({round(i * (len(order) - 1) / 19) for i in range(20)})
mean = np.asarray(config["mean"], dtype=np.float32).reshape(3, 1, 1)
std = np.asarray(config["std"], dtype=np.float32).reshape(3, 1, 1)
session = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
rows = []
for index in indices:
    path = order[index]
    image = Image.open(path).convert("RGB").resize((640, 640), Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    array = array.transpose(2, 0, 1)
    array = (array - mean) / std
    tensor = np.ascontiguousarray(array[None, ...].astype(np.float32))
    logits = session.run(None, {input_name: tensor})[0].reshape(-1)
    rows.append({"index": index, "path": path, "logits": logits.tolist(), "top1": int(np.argmax(logits))})
(RUN / "validation/onnx_validation_outputs.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"ONNX DATASET VALIDATION PASS: {len(rows)} images")
