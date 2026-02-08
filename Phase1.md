# WaferShield AI  
## Phase 1 Submission  
Edge-AI Defect Classification for Semiconductor Images

---

# 1. Problem Understanding

Semiconductor fabrication generates large volumes of wafer inspection images. Manual and centralized inspection pipelines suffer from:

- High latency
- Expensive compute infrastructure
- Network bandwidth bottlenecks
- Limited scalability for real-time throughput

An Edge-AI based system enables on-device inference, low latency, and reduced cloud dependency, making it suitable for Industry 4.0 manufacturing environments.

The objective of this work is to design an Edge-AI capable defect classification system that balances:

- Accuracy
- Model size
- Compute efficiency
- Deployment portability

---

# 2. Dataset Design

## Dataset Source

Dataset used: WM-811K (LSWMD)  
Total available wafer maps: 811,457  

---

## Selected Classes (8 Total)

The following 8 classes were selected:

- Center  
- Clean (mapped from "none")  
- Donut  
- Edge-Loc  
- Edge-Ring  
- Loc  
- Random  
- Scratch  

---

## Dataset Size and Balance

- 149 images per class  
- Total dataset size: 1,192 images  
- Balanced class distribution  

---

## Data Split

- Train: 70%  
- Validation: 15%  
- Test: 15%  
- 22 test samples per class  

Stratified and balanced across all classes.

---

## Image Processing

- Grayscale wafer maps converted to PNG format
- Resized to 320 × 320
- Normalized using ImageNet mean and standard deviation
- Data augmentation applied during training

---

# 3. Model Development

## Framework

- Python
- PyTorch
- timm library
- ONNX Runtime

Training platform:
- Kaggle GPU (CUDA)

Inference and benchmarking:
- ONNX Runtime CPUExecutionProvider

---

## Model Architecture

EfficientNet-Lite0 (Transfer Learning)

Reasons for selection:

- Designed for mobile and embedded deployment
- Lightweight architecture
- Suitable for edge constraints
- Strong accuracy-to-size tradeoff

---

# 4. Model Results

## PyTorch Test Performance

Test Accuracy: 90.34%  
Macro F1 Score: ~0.90  

---

## Per-Class Performance

| Class      | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
| Center     | 0.92      | 1.00   | 0.96     |
| Clean      | 0.91      | 0.91   | 0.91     |
| Donut      | 0.88      | 1.00   | 0.94     |
| Edge-Loc   | 0.67      | 0.91   | 0.77     |
| Edge-Ring  | 1.00      | 0.86   | 0.93     |
| Loc        | 1.00      | 0.64   | 0.78     |
| Random     | 1.00      | 1.00   | 1.00     |
| Scratch    | 1.00      | 0.91   | 0.95     |

Loc remains the most challenging class due to similarity with Edge-Loc.

---

# 5. ONNX Model

## Export Configuration

- Exported using torch.onnx.export
- Opset version: 18
- Dynamic batch axis enabled

---

## Model Size

| Format | Size | Accuracy |
|--------|------|----------|
| FP16 ONNX | 6.76 MB | 89.77% |

---

# 6. Edge Benchmark Results

Platform: ONNX Runtime (CPUExecutionProvider)

Test images evaluated: 176

Results:

- Total inference time: 1.5056 seconds
- Average latency: 8.55 ms per image
- Throughput: 116.9 images per second

These results confirm real-time suitability for high-volume inspection environments.

---

# 7. Phase 1 Deliverables Summary

The following items are completed for Phase 1:

- Problem understanding and methodology documentation
- Balanced dataset (>1000 images)
- Trained ML model exported in ONNX format
- Test results including Accuracy, Precision, Recall, and Confusion Matrix
- Model size specification
- GitHub repository with complete development code

---

# Conclusion

WaferShield AI demonstrates that:

- High defect classification accuracy (~90%) can be achieved under edge constraints
- Model size can be reduced below 7 MB
- Real-time CPU inference (<10 ms per image) is achievable
- The system is portable and ready for NXP eIQ edge deployment in subsequent phases