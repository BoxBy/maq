import importlib
from typing import TYPE_CHECKING
from functools import partial
from packaging import version
from optimum.gptq.utils import get_layers
from torch import nn
import torch

from transformers.utils import logging
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

logger = logging.get_logger(__name__)

def gptq_process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
    """
    Process the model before loading weights if the quantizer is pre-quantized.

    This function checks the pre_quantized flag and converts the model using the optimum quantizer.
    For optimum versions <= 1.23.99, it uses an older conversion method.
    """
    if self.pre_quantized:
        # compat: latest optimum has gptqmodel refactor
        if version.parse(importlib.metadata.version("optimum")) <= version.parse("1.23.99"):
            model = self.optimum_quantizer.convert_model(model)
        else:
            model = self.optimum_quantizer.convert_model(model, **kwargs)

def convert_model(self, model: nn.Module, **kwargs):
    """
    Convert the model to a GPTQ model by replacing specific layers.

    This function obtains the layers to be replaced in the model and selectively quantizes them.
    If modules_in_block_to_quantize is set, only the specified submodules are quantized.
    It then selects a quantized linear layer implementation and replaces affected layers.

    Args:
        model (nn.Module): The model to be converted.
        **kwargs: Additional keyword arguments that may include device mapping and other settings.

    Returns:
        nn.Module: The converted GPTQ model.
    """
    layers_to_be_replaced = get_layers(model)
    if self.modules_in_block_to_quantize is not None:
        layers_to_keep = sum(self.modules_in_block_to_quantize, [])
        for name in list(layers_to_be_replaced.keys()):
            if not any(name.endswith(layer) for layer in layers_to_keep):
                logger.info(
                    f"Quantization disabled for {name} (only modules_in_block_to_quantize={self.modules_in_block_to_quantize} are quantized)"
                )
                del layers_to_be_replaced[name]

    self.select_quant_linear(device_map=kwargs.get("device_map", None), pack=False)

    self._replace_by_quant_layers(model, layers_to_be_replaced)

    return model

def gptq_dequantize_module(model):
    """
    Dequantize a model by replacing GPTQ linear layers with standard linear layers.

    This function iterates over the model's modules, identifies those that are instances
    of TorchQuantLinear, then creates and assigns a new nn.Linear layer with dequantized weights.
    The dequantized weights are converted to CUDA and bfloat16 format.
    
    Args:
        model (nn.Module): The GPTQ model with quantized modules.

    Returns:
        nn.Module: The dequantized model.
    """
    for name, module in model.named_modules():
        # Create a new Linear layer with dequantized weights
        if isinstance(module, TorchQuantLinear):
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to('cuda').to(torch.bfloat16))
            new_module.bias = torch.nn.Parameter(module.bias)

            # Replace the module in the model
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    return model

def gptq(quantizer):
    """
    Initialize and configure the GPTQ quantizer with model processing functions.

    This function applies partial functions to enable pre-weight-loading processing and custom model conversion.
    It sets the _process_model_before_weight_loading and convert_model functions of the optimum quantizer.

    Args:
        quantizer: The quantizer object to be configured.

    Returns:
        The configured quantizer.
    """
    quantizer._process_model_before_weight_loading = partial(gptq_process_model_before_weight_loading, quantizer)
    quantizer.optimum_quantizer.convert_model = partial(convert_model, quantizer.optimum_quantizer)
    return quantizer