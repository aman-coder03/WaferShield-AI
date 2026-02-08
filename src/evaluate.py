import torch
import torch.nn as nn
import timm
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, classes = get_dataloaders("data")
num_classes = len(classes)

model = timm.create_model(
    "efficientnet_lite0",
    pretrained=False,
    num_classes=num_classes
)

model.load_state_dict(torch.load("models/model.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\nConfusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))