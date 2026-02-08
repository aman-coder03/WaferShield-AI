from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType
)
import numpy as np
from dataset import get_dataloaders

_, val_loader, _, _ = get_dataloaders("data")

class DataReader(CalibrationDataReader):
    def __init__(self):
        self.data_iter = iter(val_loader)
        self.input_name = "input"

    def get_next(self):
        try:
            images, _ = next(self.data_iter)
            images_np = images.numpy().astype(np.float32)
            return {self.input_name: images_np}
        except StopIteration:
            return None

input_model = "models/model.onnx"
output_model = "models/model_quant.onnx"

quantize_static(
    model_input=input_model,
    model_output=output_model,
    calibration_data_reader=DataReader(),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8
)

print("Static quantized model saved successfully.")
