import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from dataset import get_dataloaders

os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

train_loader, val_loader, test_loader, classes = get_dataloaders("data")
num_classes = len(classes)

print("Classes:", classes)

model = timm.create_model(
    "efficientnet_lite0",
    pretrained=True,
    num_classes=num_classes
)

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
    model.parameters(),
    lr=4e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=8,
    gamma=0.3
)

epochs = 30
best_val_acc = 0.0

for epoch in range(epochs):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {avg_loss:.4f} "
          f"| Val Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/model.pth")

    scheduler.step()

print("\nTraining complete!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")