import importlib
from typing import TYPE_CHECKING, Any, Optional, Dict, Tuple
from functools import partial
from packaging import version
from optimum.gptq.utils import get_layers
from torch import nn
import torch

from transformers.utils import logging
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

from logging import getLogger

from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils.quantization_config import QuantizationMethod, GPTQConfig

from optimum.utils import is_accelerate_available, is_auto_gptq_available, is_gptqmodel_available
from optimum.utils.modeling_utils import recurse_getattr
from optimum.gptq.data import get_dataset, prepare_dataset
from optimum.gptq.utils import (
    get_block_name_with_pattern,
    get_device,
    get_layers,
    get_preceding_modules,
    get_seqlen,
    nested_move_to,
)
from optimum.gptq.quantizer import ExllamaVersion

if is_accelerate_available():
    from accelerate import (
        cpu_offload_with_hook,
    )
    from accelerate.hooks import remove_hook_from_module

if is_auto_gptq_available():
    from auto_gptq.quantization import GPTQ

if is_gptqmodel_available():
    from gptqmodel.quantization import GPTQ

logger = getLogger(__name__)


def has_device_more_than_cpu():
    return torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())


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
            
def gptq_process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
    if self.pre_quantized:
        model = self.optimum_quantizer.post_init_model(model)
    else:
        if self.quantization_config.tokenizer is None:
            self.quantization_config.tokenizer = model.name_or_path

        self.optimum_quantizer.quantize_model(model=model, tokenizer=self.quantization_config.tokenizer, **kwargs)
        model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

def gptq_convert_model(self, model: nn.Module, **kwargs):
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
            # Determine if the original module had a bias that should be transferred
            original_bias_tensor = module.bias
            has_actual_bias = original_bias_tensor is not None and original_bias_tensor.numel() > 0

            new_module = nn.Linear(module.in_features, module.out_features, bias=has_actual_bias)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to('cuda').to(torch.bfloat16))
            
            if has_actual_bias:
                # Ensure bias is correctly shaped (e.g., [out_features]) and typed for nn.Linear
                new_module.bias.data = original_bias_tensor.detach().clone().squeeze().to(device='cuda', dtype=torch.bfloat16)

            current_bias_shape = new_module.bias.shape if new_module.bias is not None else "None"
            logger.debug(f'Dequantized module {name}: new weight shape: {new_module.weight.shape}, new bias shape: {current_bias_shape}')

            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name
            setattr(parent, module_name, new_module)
    return model.to('cuda').to(torch.bfloat16)

@torch.no_grad()
def gptq_quantize_model(self, model: nn.Module, tokenizer: Optional[Any] = None, **kwargs):
    """
    Quantizes the model using the dataset

    Args:
        model (`nn.Module`):
            The model to quantize
        tokenizer (Optional[`Any`], defaults to `None`):
            The tokenizer to use in order to prepare the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
    Returns:
        `nn.Module`: The quantized model
    """

    if not is_auto_gptq_available() and not is_gptqmodel_available():
        raise RuntimeError(
            "gptqmodel or auto-gptq is required in order to perform gptq quantzation: `pip install gptqmodel` or `pip install auto-gptq`. Please notice that auto-gptq will be deprecated in the future."
        )
    elif is_gptqmodel_available() and is_auto_gptq_available():
        logger.warning(
            "Detected gptqmodel and auto-gptq, will use gptqmodel. The auto_gptq will be deprecated in the future."
        )

    gptq_supports_cpu = (
        is_auto_gptq_available()
        and version.parse(importlib.metadata.version("auto-gptq")) > version.parse("0.4.2")
    ) or is_gptqmodel_available()

    if not gptq_supports_cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "No cuda gpu or cpu support using Intel/IPEX found. A gpu or cpu with Intel/IPEX is required for quantization."
        )

    if not self.sym and not is_gptqmodel_available():
        raise ValueError(
            "Asymmetric sym=False quantization is not supported with auto-gptq. Please use gptqmodel: `pip install gptqmodel`"
        )

    if self.checkpoint_format == "gptq_v2" and not is_gptqmodel_available():
        raise ValueError(
            "gptq_v2 format only supported with gptqmodel. Please install gptqmodel: `pip install gptqmodel`"
        )

    model.eval()

    # gptqmodel internal is gptq_v2 for asym support, gptq(v1) can only support sym=True
    if is_gptqmodel_available() and self.checkpoint_format != "gptq_v2":
        self.checkpoint_format = "gptq_v2"

    # For Transformer model
    has_config = False
    has_device_map = False
    if hasattr(model, "config"):
        has_config = True
        use_cache = model.config.use_cache
        model.config.use_cache = False
    
    module_index_to_quantize = kwargs.get("module_index_to_quantize", None)

    # If the model has a device_map, we don't move to model. We have already dispatched the hook that will do the work
    if hasattr(model, "hf_device_map"):
        devices = list(model.hf_device_map.values())
        has_device_map = True
        if "disk" in devices:
            raise ValueError("disk offload is not supported with GPTQ quantization")
        if "cpu" in devices or torch.device("cpu") in devices:
            if len(model.hf_device_map) > 1:
                logger.info("Cpu offload is not recommended. There might be some issues with the memory")
                hook = None
                for name, device in model.hf_device_map.items():
                    if device == "cpu":
                        module = recurse_getattr(model, name)
                        remove_hook_from_module(module, recurse=True)
                        module, hook = cpu_offload_with_hook(module, prev_module_hook=hook)
            else:
                has_device_map = False

    if hasattr(model, "dtype"):
        self.use_cuda_fp16 = model.dtype == torch.float16

    if self.model_seqlen is None:
        # We allow a max value of 4028 to avoid passing data with huge length to the model during the calibration step
        self.model_seqlen = min(4028, get_seqlen(model))

    device = get_device(model)

    # Step 1: Prepare the data
    if isinstance(self.dataset, list) and not isinstance(self.dataset[0], str):
        dataset = self.dataset
        logger.info("GPTQQuantizer dataset appears to be already tokenized. Skipping tokenization.")
    else:
        if isinstance(tokenizer, str):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError(
                    f"""We were not able to get the tokenizer using `AutoTokenizer.from_pretrained`
                    with the string that you have passed {tokenizer}. If you have a custom tokenizer, you can pass it as input.
                    For now, we only support quantization for text model. Support for vision, speech and multimodel will come later."""
                )
        if self.dataset is None:
            raise ValueError("You need to pass `dataset` in order to quantize your model")
        elif isinstance(self.dataset, str):
            dataset = get_dataset(self.dataset, tokenizer, seqlen=self.model_seqlen, split="train")
        elif isinstance(self.dataset, list):
            dataset = [tokenizer(data, return_tensors="pt") for data in self.dataset]
        else:
            raise ValueError(
                f"You need to pass a list of string, a list of tokenized data or a string for `dataset`. Found: {type(self.dataset)}."
            )

    dataset = prepare_dataset(dataset, pad_token_id=self.pad_token_id, batch_size=self.batch_size)

    # Step 2: get the input of the 1st block
    # To do that, we need to put the modules preceding the first block on the same device as the first bloc.
    # Then we run the model and it will stop at the first bloc as we added a prehook that raise an Exception after storing the inputs.

    layer_inputs = []
    layer_outputs = []
    layer_input_kwargs = []

    self.block_name_to_quantize = get_block_name_with_pattern(model)

    if self.module_name_preceding_first_block is None:
        self.module_name_preceding_first_block = get_preceding_modules(model, self.block_name_to_quantize)

    blocks = recurse_getattr(model, self.block_name_to_quantize)

    cur_layer_device = get_device(blocks[0])
    if not is_gptqmodel_available() and cur_layer_device.type == "cpu":
        cur_layer_device = 0

    if not has_device_map:
        # put modules from module_name_preceding_first_block on cuda or xpu or cpu
        to_device = cur_layer_device
        for module_name in self.module_name_preceding_first_block:
            module = recurse_getattr(model, module_name)
            if module is None:
                raise ValueError(f"Module {module_name} was not found in model")
            module = module.to(to_device)
        blocks[0] = blocks[0].to(to_device)

    def store_input_hook(_, input, *args):
        kwargs = args[0]
        if input is None:
            if "hidden_states" in kwargs:
                input = (nested_move_to(kwargs["hidden_states"], cur_layer_device),)
            else:
                raise ValueError("No input value found in the foward pass")
        layer_inputs.append(input)
        other_kwargs = {}
        for k, v in kwargs.items():  # make sure other arguments also be captured
            if k not in ["hidden_states"]:
                other_kwargs[k] = nested_move_to(v, cur_layer_device)
        layer_input_kwargs.append(other_kwargs)
        raise ValueError

    if self.cache_block_outputs:
        handle = blocks[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for data in dataset:
            for k, v in data.items():
                data[k] = nested_move_to(v, cur_layer_device)
            try:
                model(**data)
            except ValueError:
                pass
        handle.remove()

    if not has_device_map:
        blocks[0].to(device)
        for module_name in self.module_name_preceding_first_block:
            module = recurse_getattr(model, module_name)
            if module is None:
                raise ValueError(f"Module {module_name} was not found in model")

    torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

    # Step 3: Quantize the blocks
    quantizers = {}
    for i, block in enumerate(blocks):
        if (block.name == module_index_to_quantize) or (module_index_to_quantize is None):
            logger.info(f"Start quantizing block {self.block_name_to_quantize} {i}/{len(blocks)}")

            if not self.cache_block_outputs:
                handle = block.register_forward_pre_hook(store_input_hook, with_kwargs=True)
                for data in dataset:
                    for k, v in data.items():
                        data[k] = nested_move_to(v, cur_layer_device)
                    try:
                        model(**data)
                    except ValueError:
                        pass
                handle.remove()

            # move block to cuda if needed
            # in case we have offload modules, we need to put them on cuda because of GPTQ object
            if (not has_device_map or get_device(block) == torch.device("cpu")) and has_device_more_than_cpu():
                block = block.to(0)
            layers = get_layers(block)
            block_device = get_device(block)
            if not is_gptqmodel_available() and block_device.type == "cpu":
                block_device = 0
            if isinstance(self.modules_in_block_to_quantize, list) and len(self.modules_in_block_to_quantize) > 0:
                if self.true_sequential:
                    layers_name_list = self.modules_in_block_to_quantize
                else:
                    layers_name_list = [sum(self.modules_in_block_to_quantize, [])]
            else:
                if self.true_sequential:
                    # lazy sequential but works well
                    layers_name_list = [[key] for key in layers.keys()]
                else:
                    layers_name_list = [list(layers.keys())]
            for subset_name_list in tqdm(layers_name_list, leave=False, desc="Quantizing layers inside the block"):
                subset_layers = {name: layers[name] for name in subset_name_list}
                gptq = {}
                handles = []
                # add hook for each layer in subset_layers
                for name in subset_layers:
                    gptq[name] = GPTQ(subset_layers[name])
                    gptq[name].quantizer.configure(bits=self.bits, sym=self.sym, perchannel=True)

                    def add_batch(name):
                        def tmp(_, input, output):
                            gptq[name].add_batch(input[0].data, output.data)

                        return tmp

                    # because it adding a hook will replace the old one.
                    handles.append(subset_layers[name].register_forward_hook(add_batch(name)))
                # update Hessian for each layer in subset_layers thanks to the hook
                for j in range(len(dataset)):
                    # the args are already on the gpu
                    # don't need to store the output
                    layer_inputs[j] = nested_move_to(layer_inputs[j], block_device)
                    for k, v in layer_input_kwargs[j].items():
                        layer_input_kwargs[j][k] = nested_move_to(v, block_device)

                    block(*layer_inputs[j], **layer_input_kwargs[j])
                # remove hook
                for h in handles:
                    h.remove()
                for name in subset_name_list:
                    logger.info(f"Quantizing {name} in block {i}/{len(blocks)}...")
                    quant_outputs = gptq[name].fasterquant(
                        percdamp=self.damp_percent, group_size=self.group_size, actorder=self.desc_act
                    )
                    scale, zero, g_idx = quant_outputs[0], quant_outputs[1], quant_outputs[2]
                    quantizers[f"{self.block_name_to_quantize}.{i}.{name}"] = (
                        gptq[name].quantizer,
                        scale,
                        zero,
                        g_idx,
                    )
                    gptq[name].free()
                del subset_layers
            # we get the new output from the partial quantized block
            if self.cache_block_outputs:
                for j in range(len(dataset)):
                    layer_output = block(*layer_inputs[j], **layer_input_kwargs[j])
                    layer_outputs.append(layer_output)

                # put back to device
                if not has_device_map:
                    blocks[i] = block.to(device)
                del layers
                del layer_inputs
                layer_inputs, layer_outputs = layer_outputs, []
            else:
                del layers
                del layer_inputs
                layer_inputs = []
            torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            break

    if self.bits == 4:
        # device not on gpu
        if device.type != "cuda" or (has_device_map and any(d in devices for d in ["cpu", "disk", "hpu"])):
            if not self.disable_exllama and not is_gptqmodel_available():
                logger.warning(
                    "Found modules on cpu/disk. Using Exllama/Exllamav2 backend requires all the modules to be on GPU. Setting `disable_exllama=True`"
                )
                self.disable_exllama = True
        # act order and exllama
        elif self.desc_act and not self.disable_exllama and self.exllama_version == ExllamaVersion.ONE:
            logger.warning(
                "Using Exllama backend with act_order will reorder the weights offline, thus you will not be able to save the model with the right weights."
                "Setting `disable_exllama=True`. You should only use Exllama backend with act_order for inference. "
            )
            self.disable_exllama = True
        elif not self.disable_exllama and self.exllama_version == ExllamaVersion.TWO:
            logger.warning(
                "Using Exllamav2 backend will reorder the weights offline, thus you will not be able to save the model with the right weights."
                "Setting `disable_exllama=True`. You should only use Exllamav2 backend for inference. "
            )
            self.disable_exllama = True
    # Step 4: Pack the model at the end (Replacing the layers)
    self.pack_model(model=model, quantizers=quantizers)

    model.is_quantized = True
    model.quantization_method = QuantizationMethod.GPTQ
    if has_config:
        model.config.use_cache = use_cache
        model.config.quantization_config = self.to_dict()

    # Step 5: Any post-initialization that require device information, for example buffers initialization on device.
    model = self.post_init_model(model)

    torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    return model

def pack_model(
    self,
    model: nn.Module,
    quantizers: Dict[str, Tuple],
):
    """
    Pack the model by replacing the layers by quantized layers

    Args:
        model (`nn.Module`):
            The model to pack
        quantizers (`Dict[str,Tuple]`):
            A mapping of the layer name and the data needed to pack the layer
    """
    logger.info("Packing model...")
    layers = get_layers(model)
    layers = {n: layers[n] for n in quantizers}

    self.select_quant_linear(device_map=model.hf_device_map, pack=True)

    self._replace_by_quant_layers(model, quantizers)
    qlayers = get_layers(model, [self.quant_linear])
    for name in qlayers:
        if name not in quantizers:
            logger.debug(f"Layer {name} not found in quantizers.")
            continue
        logger.info(name)
        quantizers[name], scale, zero, g_idx = quantizers[name]
        # so far can only pack layer on CPU
        layer_device = qlayers[name].device
        qlayers[name].to("cpu")
        layers[name], scale, zero, g_idx = layers[name].to("cpu"), scale.to("cpu"), zero.to("cpu"), g_idx.to("cpu")
        qlayers[name].pack(layers[name], scale, zero, g_idx)
        qlayers[name].to(layer_device)

    logger.info("Model packed.")

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
    quantizer._process_model_after_weight_loading = partial(gptq_process_model_after_weight_loading, quantizer)
    quantizer.optimum_quantizer.convert_model = partial(gptq_convert_model, quantizer.optimum_quantizer)
    quantizer.optimum_quantizer.quantize_model = partial(gptq_quantize_model, quantizer.optimum_quantizer)
    quantizer.optimum_quantizer.pack_model = partial(pack_model, quantizer.optimum_quantizer)
    return quantizer