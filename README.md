# White Blood Cell Detection and Classification for Edge Device Deployment

Amrita School of AI | Semester 6

Automated detection and classification of White Blood Cells using YOLOv11 with hybrid lightweight architectures designed for real-time edge device inference.

---

## Overview

White Blood Cells (WBCs) are a vital indicator of a person's immune health. Accurate detection and classification of WBCs is essential for diagnosing diseases such as Leukemia. However, manual analysis of blood smear images is time-consuming and highly dependent on expert interpretation.

This project presents an automated White Blood Cell detection and classification system using **YOLO11**, a state-of-the-art deep learning object detection model, optimised for **edge device deployment**. The system detects and classifies WBCs into **5 sub-categories** in real time, supporting faster and more reliable medical analysis.

---

## Problem Statement

Manual identification and classification of abnormal White Blood Cells from microscopic blood smear images is labour-intensive, subjective, and inefficient — especially in high-volume clinical settings. There is a need for an automated system that can accurately detect and classify WBCs with high precision and speed.

---

## Objectives

- Develop an automated system for detecting and classifying WBCs from microscopic blood smear images
- Classify detected WBCs into 5 sub-categories using a YOLOv11-based deep learning model
- Improve diagnostic efficiency by reducing dependence on manual microscopic analysis
- Evaluate performance using standard metrics (mAP, Precision, Recall) and compare with state-of-the-art approaches
- Optimise and deploy a lightweight hybridised model for edge device deployment

---

## WBC Classes

| Class ID | Cell Type |
|---|---|
| 0 | Neutrophil |
| 1 | Monocyte |
| 2 | Eosinophil |
| 3 | Lymphocyte |
| 4 | Basophil |

---

## Dataset

A high-quality, diverse, and balanced peripheral WBC dataset was created by integrating and cleaning three public datasets.

| Source | Description |
|---|---|
| BCCD | Blood Cell Count Dataset |
| LISC | Leukocyte Images for Segmentation and Classification |
| Raabin-WBC | Raabin White Blood Cell Dataset |

**Split Ratio: 70 : 20 : 10**

| Split | Count |
|---|---|
| Train | 1680 images |
| Validation | 476 images |
| Test | 244 images |

The datasets were merged and class-mapped using the [`Combining_dataset.ipynb`](Combining_dataset.ipynb) notebook.

---

## Model Architectures

### 1. Baseline — YOLOv11n
Standard YOLOv11 nano model fine-tuned on the custom WBC dataset, with and without augmentation.

### 2. Hybridised Model 1 — GhostConv Architecture (1.6M)
Modified YOLOv11 architecture replacing standard convolutions with **GhostConv** and **C3Ghost** blocks:
- Parameter reduction: **38.46%** (2.6M to 1.6M)
- Maintains high detection accuracy suitable for mid-tier edge devices

### 3. Hybridised Model 2 — Ultra-Lightweight Architecture (0.18M)
Aggressively compressed architecture designed for ultra-low-power embedded systems:
- Parameter reduction: **93.08%** (2.6M to 0.18M)
- Targets microcontroller-class edge hardware

### AI Domains Applied

#### AutoML — Hyperparameter Optimisation
The AutoML module from Ultralytics was used to explore training parameters including optimisation settings, loss coefficients, and augmentation factors. Each configuration was trained for 50 epochs over **100 iterations**. The fitness function used to select the best model:

```
Fitness = 0.1 x mAP@50 + 0.9 x mAP@50-95
```

#### Explainable AI — EigenCAM
**EigenCAM** from the Grad-CAM library was applied to the final feature layer to generate visual attention heatmaps. These heatmaps verify that the model focuses on relevant WBC regions rather than background artifacts, improving model transparency and clinical trust.

---

## Results

### Model Performance Comparison

| Model | Params | GFLOPs | Precision | Recall | mAP@0.5 | Inference (ms) |
|---|---|---|---|---|---|---|
| YOLOv11n (Baseline) | 2.6M | 6.3 | 0.986 | 0.985 | 0.989 | 26.06 |
| YOLOv11n (Augmented) | 2.6M | 6.3 | 0.976 | 0.986 | 0.992 | 26.4 |
| Hybridised Model 1 | 1.6M | 5.4 | 0.980 | 0.960 | 0.987 | 20.2 |
| Hybridised Model 2 | 0.18M | 1.5 | 0.944 | 0.937 | 0.973 | 9.1 |
| Hybridised 1 + AutoML | 1.6M | 5.4 | 0.988 | 0.978 | 0.988 | 22.63 |
| Hybridised 2 + AutoML | 0.18M | 1.5 | 0.956 | 0.968 | 0.977 | 10.6 |

### Comparison with State-of-the-Art

| Reference | Model | Dataset | mAP@50 | Precision | Recall | Params |
|---|---|---|---|---|---|---|
| Praveen et al., 2021 | YOLOv3-based | LISC | 90.0% | 88.0% | 87.0% | 61.5M |
| Liu et al., 2022 | Faster R-CNN | BCCD | 91.3% | 89.2% | 88.5% | 41.0M |
| Blood Cell Detection using YOLO | YOLOv5 | BCCD | 92.6% | 90.1% | 91.4% | 7.2M |
| Wu et al., 2023 | SDE-YOLO | BCCD | 94.2% | 91.8% | 92.4% | 6.3M |
| Zhang et al., 2024 | TW-YOLO | BCCD | 95.4% | 93.0% | 93.5% | 5.1M |
| Sazak et al., 2024 | YOLOv10/YOLOv11 | BCCD | 93.8% | 92.0% | 92.5% | 7–10M |
| Abozeid et al., 2025 | Op-YOLOv8 | Private WBC | 96.1% | 93.9% | 94.7% | 6.8M |
| **Proposed (Ours)** | **Hybridised + AutoML** | **Custom WBC** | **98.8%** | **98.8%** | **97.8%** | **1.6M** |

The proposed model achieves **98.8% mAP@0.5** with only **1.6M parameters** and **5.4 GFLOPs**, outperforming all compared state-of-the-art models while being significantly more parameter-efficient.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base Model | yolo11n.pt |
| Epochs | 500 |
| Batch Size | 16 |
| Image Size | 800 x 800 |
| Optimizer | Auto |
| IOU Threshold | 0.7 |
| Patience | 500 |
| Device | GPU (Kaggle) / MPS (Apple M3 Pro) |

### Augmentation Parameters (Baseline)

| Parameter | Value |
|---|---|
| HSV Hue | 0.015 |
| HSV Saturation | 0.5 |
| HSV Value | 0.3 |
| Rotation Degrees | 10 |
| Translation | 0.05 |
| Scale | 0.2 |
| Horizontal Flip | 0.5 |
| Vertical Flip | 0.2 |

---

## Repository Structure

```
White-Blood-Cell-Detection-And-Classification-For-Edge-Device-Deployment/
|
|-- Combining_dataset.ipynb       # Dataset merging and preprocessing
|-- Project.ipynb                 # Main training and evaluation notebook
|-- requirements.txt              # Python dependencies
|
|-- configs/
|   |-- ghost_yolo_wbc.yaml       # Hybridised Model 1: GhostConv + C3Ghost backbone (1.6M params)
|   `-- lightweight_yolo_wbc.yaml # Hybridised Model 2: ultra-lightweight single-scale head (0.18M params)
|
|-- data/
|   `-- wbccd.yaml                # Dataset configuration (paths and class names)
|
|-- models/                       # Model weights (see models/README.md)
|   `-- README.md
|
`-- results/                      # Training metrics and visualisations
    |-- yolov11_aug/              # Augmented baseline results
    |-- no_aug3/                  # Baseline without augmentation results
    |-- ghost_model/              # Hybridised Model 1 results
    `-- yolov11_ghost_run/        # Ghost architecture training run
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Combine the RAABIN-WBC and custom WBC datasets using the provided notebook:

```bash
jupyter notebook Combining_dataset.ipynb
```

Update the dataset paths in the notebook:

```python
RAABIN_PATH = "/path/to/raabin.v1i.yolov11"
NEW_PATH    = "/path/to/wbc.v1i.yolov11"
OUTPUT_PATH = "/path/to/Combined_L"
```

### Training

```bash
# Baseline YOLOv11n
yolo detect train model=yolo11n.pt data=data/wbccd.yaml epochs=500 imgsz=800 batch=16

# Hybridised Model 1 — GhostConv architecture (1.6M params)
yolo detect train model=configs/ghost_yolo_wbc.yaml data=data/wbccd.yaml epochs=500 imgsz=800 batch=16

# Hybridised Model 2 — Ultra-lightweight single-scale head (0.18M params)
yolo detect train model=configs/lightweight_yolo_wbc.yaml data=data/wbccd.yaml epochs=500 imgsz=800 batch=16
```

### Inference

```python
from ultralytics import YOLO

model = YOLO("models/1.6M.pt")  # or models/0.18M.pt for ultra-lightweight
results = model.predict("path/to/blood_smear.jpg", conf=0.5)
results[0].show()
```

### Export to ONNX for Edge Deployment

```python
model = YOLO("models/1.6M.pt")
model.export(format="onnx", simplify=True)
```

---

## Methodology

```
1. Dataset Acquisition
   Download BCCD, LISC, and Raabin-WBC datasets

2. Data Preprocessing
   Class mapping, annotation verification, format standardisation

3. Dataset Merging
   Combine 3 datasets into a 70:20:10 train/val/test split

4. Model Training
   Baseline YOLOv11n (with and without augmentation)
   Hybridised Model 1 — GhostConv, 1.6M parameters
   Hybridised Model 2 — Ultra-lightweight, 0.18M parameters

5. AutoML Hyperparameter Tuning
   100 iterations x 50 epochs hyperparameter search

6. Explainable AI
   EigenCAM heatmaps for model interpretability and clinical validation

7. Export and Deployment
   ONNX export for cross-platform edge device inference
```

---

## References

1. Praveen et al., 2021 — YOLOv3-based WBC Detector (LISC / Blood smear)
2. Liu et al., 2022 — Faster R-CNN for BCCD
3. Blood Cell Detection using YOLO — YOLOv5 on BCCD
4. Wu et al., 2023 — SDE-YOLO on BCCD
5. Zhang et al., 2024 — TW-YOLO on BCCD
6. Sazak et al., 2024 — YOLOv10/YOLOv11 on BCCD
7. Abozeid et al., 2025 — Op-YOLOv8 on Private WBC Dataset
8. Ultralytics YOLO11 — https://github.com/ultralytics/ultralytics
9. Raabin-WBC Dataset — https://raabindata.com/free-data/

---

## License

This project is developed for academic purposes as part of the Semester 6 curriculum at Amrita School of AI.
