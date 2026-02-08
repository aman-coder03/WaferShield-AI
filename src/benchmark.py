import onnxruntime as ort
import numpy as np
import time
from dataset import get_dataloaders

# Load test data
_, _, test_loader, classes = get_dataloaders("data")

# Load ONNX model
session = ort.InferenceSession(
    "models/model.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

total_images = 0
start_time = time.time()

for images, _ in test_loader:
    images_np = images.numpy().astype(np.float32)
    session.run(None, {input_name: images_np})
    break

for images, _ in test_loader:
    images_np = images.numpy().astype(np.float32)
    session.run(None, {input_name: images_np})
    total_images += images_np.shape[0]

end_time = time.time()

total_time = end_time - start_time
avg_latency = (total_time / total_images) * 1000
throughput = total_images / total_time

print("\n===== BENCHMARK RESULTS =====")
print(f"Total Images: {total_images}")
print(f"Total Time: {total_time:.4f} seconds")
print(f"Average Latency per Image: {avg_latency:.2f} ms")
print(f"Throughput: {throughput:.2f} images/sec")
