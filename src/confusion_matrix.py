import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

_, _, test_loader, classes = get_dataloaders("data")

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - EfficientNet-Lite0 (FP16)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("Confusion matrix saved as confusion_matrix.png")
