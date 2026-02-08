from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "models/model.onnx"
output_model = "models/model_quant.onnx"

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8
)

print("Quantized model saved as model_quant.onnx")
