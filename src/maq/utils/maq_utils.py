import torch

from datasets import load_dataset
from tqdm.auto import tqdm

from transformers.utils import logging
from datasets.arrow_dataset import DatasetInfoMixin

logger = logging.get_logger(__name__)

def named_module_tensors(module, recurse=False):
    for named_parameter in module.named_parameters(recurse=recurse):
        name, val = named_parameter
        flag = True
        if hasattr(val,"_data") or hasattr(val,"_scale"):
            if hasattr(val,"_data"):
                yield name + "._data", val._data
            if hasattr(val,"_scale"):
                yield name + "._scale", val._scale
        else:
            yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
        yield named_buffer

def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    """
    import re
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8

def compute_model_sizes(model):
    """
    Compute the size of each submodule of a given model.
    """
    from collections import defaultdict
    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
        size = tensor.numel() * dtype_byte_size(tensor.dtype)
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes['']

def get_dataset(
    data,
    tokenizer,
    n_samples,
    remove_columns=[],
    max_seq_len=512,
    split="validation",
):
    data_return = []
    if isinstance(data, str):
        data = load_dataset(data, split=split)
        
    data = data.remove_columns(remove_columns)
    
    if isinstance(data, DatasetInfoMixin):
        for d in tqdm(data.select(range(n_samples))):
            data_encoded = tokenizer.encode(' '.join(list(d.values())))
            data_return.append(torch.tensor([data_encoded]))
    elif isinstance(data[0], str):
        data_return = tokenizer.encode(data)
    elif isinstance(data[0][0], int):
        data_return = data
    
    data_return = torch.cat(data_return, dim=1)
    
    return data_return[:, :data_return.shape[1] - data_return.shape[1]%max_seq_len].split(max_seq_len, dim=1) # (samples_len, max_seq_len)