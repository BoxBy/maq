import gc
import torch

from tqdm.auto import tqdm

from typing import List, Optional, TYPE_CHECKING

from .utils import edit_quantizer_for_maq, dequantize_module
from .utils.maq_utils import compute_model_sizes, get_dataset
from .utils.metric import METRIC_MAPPING

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.utils import logging
from transformers.quantizers import HfQuantizer, AutoHfQuantizer
from transformers.utils.quantization_config import QuantizationMethod
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

class MaqQuantizer(HfQuantizer):
    """
    A quantizer for Model Adaptive Quantization (MAQ) that extends HfQuantizer.
    It iteratively quantizes model modules to reduce memory usage while ensuring
    a target VRAM limit is met. The class also supports dataset preparation
    for quantization and provides various helper methods for the quantization process.
    """
    
    def __init__(self, quantization_config, **kwargs):
        """
        Initialize the MaqQuantizer.

        Parameters:
            quantization_config: Configuration for quantization including memory limits and dataset.
            metric: Metric or metric name string used to assess quantization impact.
            **kwargs: Additional keyword arguments (e.g., model_seqlen).
        """
        super().__init__(quantization_config, **kwargs)
        self.quantizer = AutoHfQuantizer.from_config(self.quantization_config.quantization_config, pre_quantized=False)
        self.quantizer = edit_quantizer_for_maq(self.quantizer)
        metric = kwargs.get("metric", 'combined')
        if isinstance(metric, str) and metric.lower() in METRIC_MAPPING:
            self.metric = METRIC_MAPPING[metric]
        else:
            self.metric = metric
        if not getattr(self.quantization_config, "model_seqlen", False):
            self.quantization_config.model_seqlen = kwargs.get("model_seqlen", None)
        
        self.original_weights_cpu = {} # 원본 가중치 캐시 초기화
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_mem = torch.cuda.get_device_properties(device).total_memory
        if isinstance(self.quantization_config.memory_limit, str) and self.quantization_config.memory_limit.endswith('%'):
            self.desired_bytes = int(total_mem * (float(self.quantization_config.memory_limit.rstrip('%')) / 100.0))
        else:
            try:
                value = float(self.quantization_config.memory_limit)
                if 0 < value <= 1.0:
                    self.desired_bytes = int(total_mem * value)
                else:
                    self.desired_bytes = int(value * (1024 ** 3))
            except Exception as e:
                raise ValueError("memory_limit format is not valid. (e.g., '70%', '0.7' or a number in GB)")
        self.prepare_dataset()

    def validate_environment(self, device_map, **kwargs):
        """
        Validate that the current computational environment (e.g., device map) is compatible
        with the quantizer's requirements.

        Parameters:
            device_map: Mapping of model layers to devices.
            **kwargs: Additional keyword arguments.
        """
        self.quantizer.validate_environment(device_map, **kwargs)

    def update_torch_dtype(self, torch_dtype):
        """
        Update the torch data type (dtype) used by the quantizer.
        
        Parameters:
            torch_dtype: The new data type to use.
        """
        self.quantizer.update_torch_dtype(torch_dtype)
    
    @torch.no_grad()
    def quantize_once(self, model, module, score, **kwargs):
        """
        Quantize a single module based on the provided score.
        If the module's current bitwidth can be reduced without falling below 2 bits,
        the quantizer will perform quantization.

        Parameters:
            module: The model module to quantize.
            score: The metric score used to determine quantization priority.
            **kwargs: Additional keyword arguments.

        Returns:
            True if quantization was performed; False otherwise.
        """
        if not hasattr(module, "current_bit"):
            logger.info(f'{module.name} is being quantized for the first time.')
            module.current_bit = 16
            # 레이어가 처음 양자화될 때 원본 FP16 가중치를 CPU에 캐싱
            self.original_weights_cpu[module.name] = {name: param.cpu().clone() for name, param in module.state_dict().items()}
            logger.info(f"Cached original weights for {module.name} to CPU.")
        if module.current_bit <= 2:
            logger.info(f"{module.name} is already at the minimum bit ({module.current_bit} bits).")
            return False

        new_bit = max(2, module.current_bit // 2)
        orig_bit = module.current_bit

        self.quantizer.quantization_config.bits = new_bit
        
        if self.quantizer.quantization_config.quant_method == QuantizationMethod.GPTQ:
            self.quantizer.optimum_quantizer.bits = new_bit
            self.quantizer.optimum_quantizer.block_name_to_quantize = module._modules.keys()
            kwargs = dict(kwargs, module_index_to_quantize = module.name)
            
        # 이미 양자화된 모듈을 재양자화하는 경우, 원본 가중치 복원
        if hasattr(module, "quantized") and module.quantized:
            logger.info(f'Module {module.name} was already quantized. Dequantizing and restoring original weights for re-quantization.')
            # 1단계: 모듈 구조를 nn.Linear로 복원하기 위해 역양자화
            module = dequantize_module(self.quantizer.quantization_config.quant_method, module)
            
            # 2단계: CPU 캐시에서 원본 FP16 가중치를 로드
            if module.name in self.original_weights_cpu:
                original_state_dict = self.original_weights_cpu[module.name]
                device = next(iter(module.parameters())).device
                # 가중치를 모듈의 장치로 이동 후 로드
                original_state_dict_on_device = {k: v.to(device) for k, v in original_state_dict.items()}
                module.load_state_dict(original_state_dict_on_device)
            else:
                logger.warning(f"Could not find original weights for {module.name} in cache. Re-quantizing from dequantized weights.")
            
        self.quantizer._process_model_before_weight_loading(
            model, **kwargs
        )
        for name, m in module.named_modules():
            if hasattr(m, "post_init"):
                m.post_init()  
        self.quantizer._process_model_after_weight_loading(model, **kwargs)
        
        logger.info(f"{module.name} bitwidth: {orig_bit} -> {module.self_attn.q_proj.bits} (score: {score:.2f})")
        module.current_bit = module.self_attn.q_proj.bits
        module.quantized = True
        
        return True
    
    def _process_model_before_weight_loading(self, model, **kwargs):
        """
        Preprocess the model before loading quantized weights.
        This is a placeholder method to be implemented if required.

        Parameters:
            model: The model to preprocess.
            **kwargs: Additional keyword arguments.
        """
        pass

    @torch.no_grad()
    def _process_model_after_weight_loading(self, model: "PreTrainedModel", keep_in_fp32_modules: Optional[List[str]] = None, **kwargs):
        """
        Process the model after weight loading by evaluating the VRAM usage,
        updating each module's name, and iteratively quantizing modules until the
        desired VRAM usage is achieved.

        Parameters:
            model: The model to process.
            keep_in_fp32_modules: List of module names to keep in FP32 (optional).
            **kwargs: Additional keyword arguments.
        """
        model.eval()    

        if getattr(self.quantization_config, "module_dict", False) and self.quantization_config.module_dict:
            logger.info(f"This model has already been quantized by MaqQuantizer. quantize processed by module_dict")
            return self.quantize_from_pretrained(model, **kwargs)
            
        current_usage = compute_model_sizes(model)
        
        logger.info(f"Target VRAM limit: {self.desired_bytes/(1024**3):.3f} GB, current VRAM usage: {current_usage/(1024**3):.3f} GB")
        
        modules = getattr(model, 'layers', False)
        if not modules:
            modules = getattr(model.model, 'layers')
            
        for i, module in enumerate(modules):
            module.name = f'{module.__class__.__name__}_{i}'
            
        module_num = 1
        pbar = tqdm(total=len(modules)*(3+self.quantization_config.use_pruning))
        while current_usage > self.desired_bytes:
            
            pbar.set_description(f"memory usage: {current_usage/(1024**3):.3f} GB")
            inps, outs = self.capture(model, modules)
            scores = dict(sorted(self.metric(modules, inps, outs).items(), key=lambda item:item[1], reverse=self.quantization_config.reverse_sort))
            del inps, outs
            
            for module, score in scores.items():
                if self.quantize_once(model, module, score, **kwargs):
                    self.quantization_config.quantize_recipe[module_num] = [
                        module.name,
                        module.current_bit,
                        score
                        ]
                    module_num += 1
                    break
                elif module.current_bit == 2 and self.quantization_config.use_pruning:
                    logger.info(f'Pruning the module {module.name} according to the QuantizerConfig')
                    for i, m in enumerate(modules):
                        if m.name == module.name:
                            if module.name in self.original_weights_cpu:
                                del self.original_weights_cpu[module.name] # 프루닝된 모듈의 캐시된 가중치 삭제
                            del modules[i]
                            break
                    break
                        
            current_usage = compute_model_sizes(model)
            pbar.update(1)
            self.collect_memory()
            logger.info(f"Model output: {self.quantization_config.tokenizer.decode(model.generate(**self.quantization_config.tokenizer('hello, world!', return_tensors='pt').to(model.device), generation_config=GenerationConfig(repetition_penalty=1.3, max_new_tokens=200))[0])}")

        total_bit = sum(getattr(m, "current_bit", 16) for m in modules)
        avg_bit = total_bit / len(modules) if modules else 0
        logger.info(f"Final memory usage: {current_usage/(1024**3):.3f} GB, average bits (hidden layers only): {avg_bit:.2f}")
            
    def prepare_dataset(self):
        """
        Prepare the dataset for quantization.
        If the dataset is already tokenized, it is used directly;
        otherwise, tokenization is applied using the provided tokenizer.
        """
        if isinstance(self.quantization_config.dataset, list) and not isinstance(self.quantization_config.dataset[0], str):
            dataset = self.quantization_config.dataset
            logger.info("MAQQuantizer dataset appears to be already tokenized. Skipping tokenization.")
        else:
            if isinstance(self.quantization_config.tokenizer, str):
                try:
                    self.quantization_config.tokenizer = AutoTokenizer.from_pretrained(self.quantization_config.tokenizer)
                except Exception:
                    raise ValueError(
                        f"""We were not able to get the tokenizer using `AutoTokenizer.from_pretrained`
                        with the string that you have passed {self.tokenizer}. If you have a custom tokenizer, you can pass it as input.
                        For now, we only support quantization for text model. Support for vision, speech and multimodel will come later."""
                    )
            if self.quantization_config.dataset is None:
                raise ValueError("You need to pass `dataset` in order to quantize your model")
            elif isinstance(self.quantization_config.dataset, str):
                dataset = get_dataset(self.quantization_config.dataset, self.quantization_config.tokenizer, split=self.quantization_config.dataset_split, remove_columns=self.quantization_config.remove_columns, n_samples=self.quantization_config.n_samples)
            else:
                raise ValueError(
                    f"You need to pass a list of string, a list of tokenized data or a string for `dataset`. Found: {type(dataset)}."
                )
        self.dataset = torch.cat(dataset, dim=0)
    
    @torch.no_grad()
    def capture(self, model, modules):        
        """
        Capture the inputs and outputs for each module in the model using forward hooks.
        This is used to compute evaluation metrics for quantization.

        Parameters:
            model: The model to capture data from.
            modules: The list of modules to register hooks on.

        Returns:
            A tuple of two torch tensors:
                - Stacked module inputs.
                - Stacked module outputs.
        """
        torch_inputs = []
        torch_outputs = []
        
        pbar = tqdm(desc=f"process", total=len(modules)*len(self.dataset))
        
        def store_hook(_, input, output):
            pbar.update(1)
            layer_inputs.append(input[0].detach().cpu())
            if layer_inputs[-1].dtype != torch.bfloat16:
                logger.info(f'input dtype : {layer_inputs[-1].dtype}')
            if isinstance(output, tuple):
                try:
                    layer_outputs.append(output[0].detach().cpu())
                    if layer_outputs[-1].dtype != torch.bfloat16:
                        logger.info(f'output dtype : {layer_outputs[-1].dtype}')
                except Exception:
                    logger.info(f'output : {output}')
                    pass
            else:
                try:
                    layer_outputs.append(output.detach().cpu())
                    if layer_outputs[-1].dtype != torch.bfloat16:
                        logger.info(f'output dtype : {layer_outputs[-1].dtype}')
                except Exception:
                    logger.info(f'output : {output}')
                    pass
            
        handles = [module.register_forward_hook(store_hook) for module in modules]
        dataset = self.dataset.to(next(model.parameters()).device)
        
        for data in dataset:
            try:
                layer_inputs = []
                layer_outputs = []
                model(data.unsqueeze(0))
                torch_inputs.append(torch.cat(layer_inputs, dim=0))
                torch_outputs.append(torch.cat(layer_outputs, dim=0))
            except ValueError:
                pass
        pbar.clear()
        
        del dataset
        for handle in handles:
            handle.remove()
        
        self.collect_memory()
        
        return torch.stack(torch_inputs, dim=0), torch.stack(torch_outputs, dim=0)
        
    def is_serializable(self, safe_serialization=None):
        """
        Check if the quantizer is serializable.

        Parameters:
            safe_serialization: Strategy for safe serialization (optional).

        Returns:
            Boolean indicating if the quantizer can be serialized.
        """
        return self.quantizer.is_serializable(safe_serialization)

    @property
    def is_trainable(self):
        """
        Indicates whether the quantizer supports training.
        
        Returns:
            True if the quantizer is trainable; otherwise, False.
        """
        return self.quantizer.is_trainable
    
    def collect_memory(self):
        """
        Perform garbage collection and clear GPU caches to free up memory.
        It also supports XPU if available.
        """
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        torch.cuda.ipc_collect()
        
    def quantize_model(self, model: "PreTrainedModel", keep_in_fp32_modules: Optional[List[str]] = None, **kwargs):
        """
        Quantize the given model while ensuring it fits within the desired VRAM limit.
        The method moves the model to GPU if necessary and processes weight loading for quantization.

        Parameters:
            model: The pre-trained model to quantize.
            keep_in_fp32_modules: List of module names to keep in FP32 (optional).
            **kwargs: Additional keyword arguments.
        """
        if torch.cuda.get_device_properties('cuda').total_memory > compute_model_sizes(model):
            model.to('cuda')
        self._process_model_after_weight_loading(model, keep_in_fp32_modules, **kwargs)
        
    def save_model(self, model, file_path):
        """
        Save the quantized model to disk with updated configurations.
        Before saving, it dequantizes modules that were quantized and updates the model configuration.

        Parameters:
            model: The quantized model.
            file_path: The path where the model will be saved.
        """
        modules = getattr(model, 'layers', False)
        if not modules:
            modules = getattr(model.model, 'layers')
        
        model.config.num_hidden_layers = len(modules)
        self.quantization_config.module_dict = {}
        self.quantizer.quantization_config.tokenizer = self.quantizer.quantization_config.tokenizer.name_or_path
        
        for i, module in enumerate(modules):
            self.quantization_config.module_dict[i] = getattr(module, "current_bit", 16)
            if getattr(module, "quantized", False):
                dequantize_module(self.quantizer.quantization_config.quant_method, module)

        model.config.quantization_config = self.quantization_config.to_dict()
        model.save_pretrained(file_path)
        pass
            
    def quantize_from_pretrained(self, model, **kwargs):
        """
        Load a pretrained model and apply quantization as per the saved quantization configuration.
        It processes each module and updates its bitwidth accordingly.

        Parameters:
            model: The pretrained model to be quantized.
            **kwargs: Additional keyword arguments.

        Returns:
            The quantized model.
        """
        modules = getattr(model, 'layers', False)
        if not modules:
            modules = getattr(model.model, 'layers')
            
        for i, module in enumerate(tqdm(modules)):
            if model.config.quantization_config.module_dict[str(i)] != 16:
                self.quantizer.quantization_config.bits = model.config.quantization_config.module_dict[str(i)]
                
                if self.quantizer.quantization_config.quant_method == QuantizationMethod.GPTQ:
                    self.quantizer.optimum_quantizer.bits = self.quantizer.quantization_config.bits
                    self.quantizer.optimum_quantizer.block_name_to_quantize = module._modules.keys()
                    
                self.quantizer._process_model_before_weight_loading(
                    module, **kwargs
                )
                # if hasattr(module, "post_init"):
                for name, m in module.named_modules():
                    if hasattr(m, "post_init"):
                        m.post_init()  
                
                self.quantizer._process_model_after_weight_loading(module, **kwargs)
                
                logger.info(f"{module.__class__.__name__}_{i} bitwidth: {16} -> {module.self_attn.q_proj.bits}")
                module.current_bit = module.self_attn.q_proj.bits
                module.quantized = True
            
        return model