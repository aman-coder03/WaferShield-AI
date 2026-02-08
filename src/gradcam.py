import torch
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset import get_dataloaders

device = torch.device("cpu")
num_classes = 8

model = timm.create_model(
    "efficientnet_lite0",
    pretrained=False,
    num_classes=num_classes
)
model.load_state_dict(torch.load("models/model.pth", map_location=device))
model.eval()
model.to(device)

target_layer = model.blocks[-4]

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

target_class_names = ["Center", "Random", "Edge-Loc", "Loc"]

found = {cls: False for cls in target_class_names}

for images, labels in test_loader:
    label_name = classes[labels.item()]

    if label_name in target_class_names and not found[label_name]:

        images = images.to(device)

        output = model(images)
        pred_class = torch.argmax(output, dim=1)

        model.zero_grad()
        output[0, pred_class].backward()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations_clone = activations.clone()

        for i in range(activations_clone.shape[1]):
            activations_clone[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations_clone, dim=1).squeeze()
        heatmap = torch.relu(heatmap)

        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        heatmap = np.where(heatmap > 0.4 * heatmap.max(), heatmap, 0)

        heatmap /= (np.max(heatmap) + 1e-8)

        heatmap = cv2.resize(heatmap, (320, 320))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        original = images.squeeze().permute(1, 2, 0).cpu().numpy()
        original = (original - original.min()) / (original.max() - original.min() + 1e-8)

        overlay = 0.6 * original + 0.4 * heatmap
        overlay = overlay / np.max(overlay)

        plt.figure(figsize=(8,4))

        plt.subplot(1,3,1)
        plt.title("Original")
        plt.imshow(original)
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.title("Grad-CAM")
        plt.imshow(heatmap)
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.title(f"Overlay: {label_name}")
        plt.imshow(overlay)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"gradcam_{label_name}.png", dpi=300)
        plt.close()

        print(f"Saved Grad-CAM for {label_name}")

        found[label_name] = True

    if all(found.values()):
        break
