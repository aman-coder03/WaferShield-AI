import onnxruntime as ort
import numpy as np
import torch
from dataset import get_dataloaders

_, _, test_loader, classes = get_dataloaders("data")

session = ort.InferenceSession("models/model.onnx")

input_name = session.get_inputs()[0].name

correct = 0
total = 0

for images, labels in test_loader:
    images_np = images.numpy().astype(np.float32)

    outputs = session.run(None, {input_name: images_np})[0]
    preds = np.argmax(outputs, axis=1)

    correct += (preds == labels.numpy()).sum()
    total += labels.size(0)

accuracy = correct / total
print(f"ONNX Test Accuracy: {accuracy:.4f}")
