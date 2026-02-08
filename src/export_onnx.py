import torch
import timm

device = torch.device("cpu")
num_classes = 8

model = timm.create_model(
    "efficientnet_lite0",
    pretrained=False,
    num_classes=num_classes
)

model.load_state_dict(torch.load("models/model.pth", map_location=device))
model.eval()
model.half()

dummy_input = torch.randn(1, 3, 320, 320).half()

torch.onnx.export(
    model,
    dummy_input,
    "models/model_fp16.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=18
)

print("FP16 ONNX export complete.")