from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "model.onnx"
model_quant = "model_int8.onnx"

print("Quantizing model...")
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant,
    weight_type=QuantType.QUInt8  # Reduces weights to 8-bit integers
)
print("Done!")