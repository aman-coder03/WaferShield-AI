import onnxruntime as ort
import numpy as np
from dataset import get_dataloaders

_, _, test_loader, classes = get_dataloaders("data")

session = ort.InferenceSession("models/model_fp16.onnx", providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name

correct = 0
total = 0

for images, labels in test_loader:
    images_np = images.numpy().astype(np.float16)

    outputs = session.run(None, {input_name: images_np})[0]
    preds = np.argmax(outputs, axis=1)

    correct += (preds == labels.numpy()).sum()
    total += labels.size(0)

accuracy = correct / total
print(f"ONNX Test Accuracy: {accuracy:.4f}")