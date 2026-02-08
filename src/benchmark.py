import onnxruntime as ort
import numpy as np
import time
from dataset import get_dataloaders

_, _, test_loader, classes = get_dataloaders("data", batch_size=1)

session = ort.InferenceSession(
    "models/model_fp16.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

total_images = 0
inference_time = 0.0

for images, _ in test_loader:
    images_np = images.numpy().astype(np.float16)
    session.run(None, {input_name: images_np})
    break

for images, _ in test_loader:
    images_np = images.numpy().astype(np.float16)

    start = time.time()
    session.run(None, {input_name: images_np})
    end = time.time()

    inference_time += (end - start)
    total_images += images_np.shape[0]

avg_latency = (inference_time / total_images) * 1000
throughput = total_images / inference_time

print("\n===== BENCHMARK RESULTS =====")
print(f"Total Images: {total_images}")
print(f"Total Inference Time: {inference_time:.4f} seconds")
print(f"Average Latency per Image: {avg_latency:.2f} ms")
print(f"Throughput: {throughput:.2f} images/sec")
