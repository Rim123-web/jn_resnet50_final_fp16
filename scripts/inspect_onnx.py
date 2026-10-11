from __future__ import annotations

import json

from common import FP32_REFERENCE, PACKAGED_ONNX, RUN, file_record, sha256


def main():
    reference = (
        FP32_REFERENCE / "onnx/resnet50_classification_fp32_1x3x640x640.onnx"
    )
    report = {
        "status": "PASS" if sha256(reference) == sha256(PACKAGED_ONNX) else "FAIL",
        "policy": "Reuse the exact validated fixed-shape FP32 ONNX graph.",
        "reference": file_record(reference),
        "packaged_copy": file_record(PACKAGED_ONNX),
        "sha256_parity": sha256(reference) == sha256(PACKAGED_ONNX),
        "reexport_performed": False,
    }
    (RUN / "onnx/onnx_reuse_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "PASS":
        raise RuntimeError("Reused ONNX hash parity failed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
