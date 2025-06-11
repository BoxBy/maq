# MAQ: Weight Quantization Library

MAQ is a lightweight library for weight quantization in modern deep learning models. It provides flexible quantization methods, along with useful configuration and metric utilities.

## Features

- **Flexible Quantization Methods**:  
  It supports the following quantization libraries:
  * GPTQ ([`maq_gptq.py`](src/maq/utils/maq_gptq.py))
  * AWQ ([`maq_awq.py`](src/maq/utils/maq_awq.py)) – currently disabled due to 4-bit support limitations.  
    We plan to support all the methods in transformers.quantizers.

- **Memory-based Quantization Approach**:  
  Unlike traditional methods that quantize based primarily on bit-width, MAQ adapts quantization according to available memory. In this approach, each module's importance is computed, and modules with lower importance are prioritized for quantization. For an in-depth overview of the methodology, please refer to a related paper ([https://arxiv.org/abs/2406.17415]). ([https://arxiv.org/abs/2409.14381]) 

- **Configuration Tools**:  
  Easily fine-tune quantization parameters using the configuration utility ([`config.py`](src/maq/utils/config.py)).

- **Evaluation Metrics**:  
  Built-in support for quantization evaluation via the metrics module ([`metric.py`](src/maq/utils/metric.py)).

## Installation

Install MAQ from the project root:

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

## Usage

Here is a basic example to help you get started:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_MAQ import MaqQuantizationConfig, MaqQuantizer
from transformers.utils.quantization_config import GPTQConfig

# Specify the model to be quantized.
model_name = "meta-llama/Meta-Llama-3-8B"

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a GPTQ quantization configuration.
# Note: The 'bits' parameter here is not used by the quantization process.
gptqconfig = GPTQConfig(bits=4, dataset='wikitext2')

# Define the MAQ quantization configuration with memory limits and pruning as options.
quantization_config = MaqQuantizationConfig(
    memory_limit=0.2,
    tokenizer=tokenizer,
    dataset="pileval",
    quantization_config=gptqconfig,
    use_pruning=True
)

# Load the pre-trained model with low CPU memory usage and proper device mapping.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

# Initialize the MAQ quantizer using the configuration and apply quantization.
quantizer = MaqQuantizer(quantization_config, metric='lim')
quantizer.quantize_model(model)

# Save the quantized model to disk.
quantizer.save_model(model, f"{model_name}_MAQ")
```

Additional utilities and advanced usage examples can be found in the [`src/maq/utils/`](src/maq/utils/) directory.

## License

MAQ is licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for further details.

---

This repository was developed leveraging GitHub Copilot's o3-mini.
```
