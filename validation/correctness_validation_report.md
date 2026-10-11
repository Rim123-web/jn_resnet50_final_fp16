# Correctness Validation

- Status: **PASS**
- Images: 20 deterministic samples
- Validated FP32 source/ONNX reference: PASS
- Source/ONNX/FP16 top-1 agreement: True
- Fresh FP32/FP16 top-1 agreement: True
- Maximum FP32/FP16 absolute logit error: 0.244062424
- Minimum FP32/FP16 cosine similarity: 0.999954933
- Exact source checkpoint and ONNX hashes match the validated FP32 package.
- A current native PyTorch CPU rerun produced NaNs after repeated samples; this runtime limitation is disclosed and was not used to weaken FP32/FP16 acceptance.
- FP32 and FP16 TensorRT runtimes were opened sequentially and released before exit.
