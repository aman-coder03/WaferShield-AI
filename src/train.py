import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0
from dataset import get_dataloaders

os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, classes = get_dataloaders("data")
num_classes = len(classes)

model = efficientnet_b0(weights="IMAGENET1K_V1")

model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

for name, param in model.features.named_parameters():
    if not any(layer in name for layer in ["4", "5", "6", "7"]):
        param.requires_grad = False

model = model.to(device)

class_weights = torch.tensor(
    [1.2, 1.0, 1.5, 1.0, 1.5, 1.5, 1.0, 1.5],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0003,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=8,
    gamma=0.3
)

epochs = 25
best_val_acc = 0

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {avg_loss:.4f} | "
          f"Val Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/model.pth")

    scheduler.step()

print("\nTraining complete!")
print("Best Validation Accuracy:", best_val_acc)
