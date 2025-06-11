from transformers.utils.quantization_config import QuantizationMethod
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING, AUTO_QUANTIZATION_CONFIG_MAPPING
from .utils.config import MaqQuantizationConfig
from .quantizer import MaqQuantizer

QuantizationMethod.MAQ = "maq"

AUTO_QUANTIZATION_CONFIG_MAPPING["maq"] = MaqQuantizationConfig
AUTO_QUANTIZER_MAPPING["maq"] = MaqQuantizer

