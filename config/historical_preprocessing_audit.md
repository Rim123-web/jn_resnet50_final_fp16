# Historical Preprocessing Audit

- Decoder: PIL.Image.open
- Color ordering: RGB
- Geometry: direct bilinear resize to 640 x 640; no crop, pad, or letterbox
- Pixel scale: divide by 255
- Mean: [0.485, 0.456, 0.406]
- Standard deviation: [0.229, 0.224, 0.225]
- Layout: HWC to CHW
- Tensor: contiguous float32 [1,3,640,640]
- Primary evidence: `/home/jetson/Downloads/classification_jetson/resnet50_fp32_core4_benchmark/scripts/run_resnet50_fp32_core4_benchmark.py` lines 189-198
- Configuration evidence: `/home/jetson/Downloads/classification_jetson/preprocessing.json` lines 2-6
- Ambiguity: none
