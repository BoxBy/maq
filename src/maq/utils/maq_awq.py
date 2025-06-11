from typing import TYPE_CHECKING, Optional, List
from functools import partial

from transformers.utils import logging
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

logger = logging.get_logger(__name__)

def awq_process_model_before_weight_loading(
        self, model: "PreTrainedModel", keep_in_fp32_modules: Optional[List[str]] = None, **kwargs
    ):
        from transformers.integrations import replace_quantization_scales, replace_with_awq_linear
        model, has_been_replaced = replace_with_awq_linear(
            model, quantization_config=self.quantization_config)

        model = replace_quantization_scales(model, model.config.model_type)

        if not has_been_replaced:
            logger.warning(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )
    
def awq(quantizer):
    quantizer._process_model_before_weight_loading = partial(awq_process_model_before_weight_loading, quantizer)
    return quantizer