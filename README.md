# WaferShield AI: Protecting Semiconductor Yield with Edge AI

## Overview

WaferShield AI is an Edge-AI powered defect classification system designed to detect and classify semiconductor wafer defects using deep learning.

The system is built to reflect real fabrication constraints:
- Lightweight model architecture
- Balanced accuracy vs compute tradeoff
- Deployment-ready (ONNX export compatible)
- Structured dataset pipeline
- Clear evaluation and benchmarking

Final Test Accuracy Achieved: **80%**

---

## Problem Statement

Semiconductor fabrication generates large volumes of wafer inspection images. Manual review and centralized processing introduce:

- High latency
- Infrastructure overhead
- Bandwidth limitations
- Scalability challenges

WaferShield AI addresses this by enabling defect classification suitable for edge deployment.

---

## Dataset

Dataset Used: **LSWMD (Wafer Map Dataset)**  
Total wafer maps: 811,457  
Defect classes available:
- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Random
- Scratch
- Near-full
- none

### Selected Classes

We selected 8 defect classes (excluding "none"):

- Center  
- Donut  
- Edge-Loc  
- Edge-Ring  
- Loc  
- Random  
- Scratch  
- Near-full  

### Balanced Dataset Strategy

To ensure class balance:
- 149 samples per class
- Total training images: 1,192
- 70/15/15 Train/Validation/Test split
- 22 test samples per class

---

## Data Pipeline

### Extraction
- Loaded raw `.pkl` dataset
- Cleaned failureType labels
- Converted wafer maps to grayscale PNG images
- Created structured directory format

### Splitting
- Train / Validation / Test split
- Stratified by class
- Ensured balanced distribution

---

## Model Architecture

Base Model: **EfficientNet-B0 (Pretrained on ImageNet)**

### Modifications
- Resolution: 288x288
- Partial fine-tuning (last blocks unfrozen)
- Custom classifier head:
  - Dropout(0.4)
  - Linear(num_classes)

### Training Strategy
- Weighted CrossEntropy Loss
- Label smoothing (0.1)
- Adam optimizer (lr=0.0003)
- Weight decay (1e-4)
- StepLR scheduler
- 25 epochs training

---

## Final Results

### Validation Accuracy
**85.8%**

### Test Accuracy
**80%**

### Classification Summary

| Class       | F1 Score |
|------------|----------|
| Center     | 0.67     |
| Donut      | 0.95     |
| Edge-Loc   | 0.66     |
| Edge-Ring  | 0.84     |
| Loc        | 0.53     |
| Near-full  | 0.98     |
| Random     | 1.00     |
| Scratch    | 0.73     |

Macro F1 Score: **0.79**

---

## Key Observations

- EfficientNet significantly improved spatial defect recognition.
- Near-full and Random defects are highly separable.
- Loc remains the most challenging due to similarity with Edge-Loc.
- Validation and test performance are consistent → minimal overfitting.

---

## Project Structure

WaferShield AI/
│
├── data/
│ ├── train/
│ ├── val/
│ └── test/
│
├── src/
│ ├── extract_LSWMD.py
│ ├── split_dataset.py
│ ├── dataset.py
│ ├── train.py
│ └── evaluate.py
│
├── models/
│ └── model.pth
│
└── README.md

---

## How to Run

### 1️⃣ Extract Dataset
python src/extract_LSWMD.py


### 2️⃣ Split Dataset
python src/split_dataset.py


### 3️⃣ Train Model
python src/train.py


### 4️⃣ Evaluate Model
python src/evaluate.py

---

## Next Steps (Edge Deployment)

- Export trained model to ONNX
- Apply model quantization (INT8)
- Benchmark inference latency
- Deploy using NXP eIQ toolchain

---

## Engineering Highlights

- Balanced dataset construction
- Structured ML pipeline
- Controlled fine-tuning strategy
- Robust evaluation metrics
- Edge-ready architecture
- Lightweight backbone (EfficientNet-B0)

---

## Authors

Team: WaferShield AI  
Domain: AI-enabled Chip Design  
Track: Edge-AI Defect Classification