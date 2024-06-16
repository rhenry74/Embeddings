from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig

dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False, 
    use_symmetric_activations=True, operators_to_quantize=["MatMul"],)

quantizer = ORTQuantizer.from_pretrained(model_or_path=".")
quantizer.quantize(save_dir="./", quantization_config=dqconfig )  