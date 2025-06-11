from transformers.utils.quantization_config import QuantizationMethod
from .config import MaqQuantizationConfig
from .maq_gptq import gptq, gptq_dequantize_module
CUSTOM = {
    QuantizationMethod.GPTQ : gptq
}
DEQUANTIZE = {
    QuantizationMethod.GPTQ : gptq_dequantize_module
}

def edit_quantizer_for_maq(quantizer):
    return CUSTOM[quantizer.quantization_config.quant_method](quantizer)

def dequantize_module(quant_method, module):
    if quant_method in DEQUANTIZE.keys():
        return DEQUANTIZE[quant_method](module)
    return module