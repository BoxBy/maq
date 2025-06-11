import copy

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.quantizers import AutoQuantizationConfig
from transformers.utils.quantization_config import QuantizationConfigMixin, GPTQConfig, QuantizationMethod

@dataclass
class MaqQuantizationConfig(QuantizationConfigMixin):
    def __init__(
        self,
        quantization_config: Optional[QuantizationConfigMixin] = None,
        memory_limit: str = "70%",
        reverse_sort: bool = False,
        tokenizer: Union[str, PreTrainedTokenizerBase] = None,
        dataset: str = "wikitext2",
        use_pruning: bool = False,
        quantize_recipe = {},
        module_dict = {},
        max_seq_len = 512,
        **kwargs
    ):
        self.quant_method = QuantizationMethod.MAQ
        self.memory_limit = memory_limit
        self.reverse_sort = reverse_sort
        self.quantization_config = quantization_config
        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.dataset = dataset  
        self.use_pruning = use_pruning
        if isinstance(quantization_config, QuantizationConfigMixin):
            self.quantization_config = quantization_config
        else:
            self.quantization_config = GPTQConfig(**kwargs)
        self.quantization_config.modules_to_not_convert = []
        self.quantize_recipe = quantize_recipe
        self.module_dict = module_dict
        self.max_seq_len = max_seq_len
        
    def to_dict(self) -> Dict[str, Any]:
        dic = copy.deepcopy(self.__dict__)
        dic["quantization_config"] = self.quantization_config.to_dict()
        dic["tokenizer"] = self.tokenizer.name_or_path
        return dic
    
    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        if config_dict.get("quantization_config", False):
            config_dict["quantization_config"] = AutoQuantizationConfig.from_dict(config_dict["quantization_config"])
        return super().from_dict(config_dict, return_unused_kwargs, **kwargs)
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(use_diff=False)}"