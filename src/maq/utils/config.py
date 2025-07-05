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
        metric: str = "combined",
        reverse_sort: bool = False,
        tokenizer: Union[str, PreTrainedTokenizerBase] = None,
        dataset: str = "wikitext2",
        dataset_split="validation",
        remove_columns=[],
        n_samples: Optional[int] = None,
        shap_samples: int = 32,
        use_pruning: bool = False,
        use_bit_width_penalty: bool = True,
        penalty_factor: float = 1.5,
        quantize_recipe = {},
        module_dict: Optional[Dict] = None,
        max_seq_len = 512,
        **kwargs
    ):
        self.quant_method = QuantizationMethod.MAQ
        self.memory_limit = memory_limit
        self.metric = metric
        self.reverse_sort = reverse_sort
        self.quantization_config = quantization_config
        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.dataset = dataset  
        self.dataset_split = dataset_split
        self.remove_columns = remove_columns
        self.n_samples = n_samples
        self.shap_samples = shap_samples
        self.use_pruning = use_pruning
        self.use_bit_width_penalty = use_bit_width_penalty
        self.penalty_factor = penalty_factor
        if isinstance(quantization_config, QuantizationConfigMixin):
            self.quantization_config = quantization_config
        else:
            self.quantization_config = GPTQConfig(bits=4, tokenizer=self.tokenizer, dataset='wikitext2', **kwargs)
        self.quantization_config.modules_to_not_convert = []
        self.quantize_recipe = quantize_recipe
        self.module_dict = module_dict
        self.max_seq_len = max_seq_len
        
    def to_dict(self) -> Dict[str, Any]:
        dic = copy.deepcopy(self.__dict__)
        dic["quantization_config"] = self.quantization_config.to_dict()
        dic["tokenizer"] = self.tokenizer.name_or_path
        dic["quantization_config"]["tokenizer"] = self.tokenizer.name_or_path
        return dic
    
    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        if config_dict.get("quantization_config", False):
            config_dict["quantization_config"] = AutoQuantizationConfig.from_dict(config_dict["quantization_config"])
        return super().from_dict(config_dict, return_unused_kwargs, **kwargs)
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(use_diff=False)}"