import torch
import torch.nn as nn
from torchvision import models

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

dummy_input = torch.randn(1, 3, 288, 288)

torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=18,
    do_constant_folding=True,
    dynamo=False
)

print("ONNX export complete.")
