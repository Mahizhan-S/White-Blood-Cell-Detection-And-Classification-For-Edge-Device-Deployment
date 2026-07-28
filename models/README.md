# Pre-trained Model Weights

Model weights are not stored in this repository due to file size constraints.
To obtain the weights, contact the team or download them from the project release page.

## Available Models

| File | Params | mAP@0.5 | GFLOPs | Inference (ms) | Description |
|---|---|---|---|---|---|
| 0.18M.pt | 0.18M | 97.7% | 1.5 | 9.1 | Ultra-lightweight model (PyTorch) |
| 0.18M.onnx | 0.18M | 97.7% | 1.5 | ~10 | Ultra-lightweight model (ONNX) |
| 1.6M.pt | 1.6M | 98.8% | 5.4 | 22.6 | Hybridised Model 1 + AutoML (PyTorch) |
| 1.6M.onnx | 1.6M | 98.8% | 5.4 | ~25 | Hybridised Model 1 + AutoML (ONNX) |
| aug.pt | 2.6M | 99.2% | 6.3 | 26.4 | Augmented baseline (PyTorch) |
| aug.onnx | 2.6M | 99.2% | 6.3 | ~28 | Augmented baseline (ONNX) |
| noaug.pt | 2.6M | 98.9% | 6.3 | 26.1 | Baseline without augmentation (PyTorch) |
| noaug.onnx | 2.6M | 98.9% | 6.3 | ~27 | Baseline without augmentation (ONNX) |

## PyTorch Inference

```python
from ultralytics import YOLO

# Recommended for edge devices (best accuracy-to-size tradeoff)
model = YOLO("models/1.6M.pt")

# Run inference
results = model.predict("blood_smear.jpg", conf=0.5)
results[0].show()

# Export to ONNX
model.export(format="onnx", simplify=True)
```

## ONNX Inference (Edge Deployment)

```python
import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession("models/1.6M.onnx")
input_name = session.get_inputs()[0].name

img = cv2.imread("blood_smear.jpg")
img = cv2.resize(img, (800, 800))
img = img.transpose(2, 0, 1)[np.newaxis, :] / 255.0

outputs = session.run(None, {input_name: img.astype(np.float32)})
```
