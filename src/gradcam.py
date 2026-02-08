import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models
from dataset import get_dataloaders

device = torch.device("cpu")
num_classes = 8

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, num_classes)
)

model.load_state_dict(torch.load("models/model.pth", map_location=device))
model.eval()
model.to(device)

target_layer = model.features[-1]

gradients = None
activations = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

_, _, test_loader, classes = get_dataloaders("data", batch_size=1)

images, labels = next(iter(test_loader))
images = images.to(device)

output = model(images)
pred_class = torch.argmax(output, dim=1)

model.zero_grad()
output[0, pred_class].backward()

pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

activations = activations.clone()

for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = torch.relu(heatmap)

heatmap = heatmap.cpu().numpy()
heatmap /= (np.max(heatmap) + 1e-8)

heatmap = cv2.resize(heatmap, (288, 288))
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255

original = images.squeeze().permute(1, 2, 0).cpu().numpy()
original = (original - original.min()) / (original.max() - original.min() + 1e-8)

overlay = heatmap * 0.4 + original
overlay = overlay / np.max(overlay)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(original)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Overlay: {classes[pred_class.item()]}")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_result.png")
plt.show()

print("Grad-CAM saved as gradcam_result.png")
