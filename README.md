#### This repository was developed leveraging GitHub Copilot's o3-mini.
---
### [English](README.md) | [한국어](README-ko.md)
---
# MAQ: Weight Quantization Library

MAQ is a lightweight library for weight quantization in modern deep learning models. It provides flexible quantization methods, along with useful configuration and metric utilities.

## Features

- **Flexible Quantization Methods**:  
  It supports the following quantization libraries:
  * GPTQ ([`maq_gptq.py`](src/maq/utils/maq_gptq.py))
  * AWQ ([`maq_awq.py`](src/maq/utils/maq_awq.py)) – currently disabled due to 4-bit support limitations.  
  
  We plan to support all the methods in transformers.quantizers.

- **Memory-based Quantization Approach**:  
  Unlike traditional methods that quantize based primarily on bit-width, MAQ adapts quantization according to available memory. In this approach, each module's importance is computed, and modules with lower importance are prioritized for quantization. For an in-depth overview of the methodology, please refer to the related papers [Layer-Wise Quantization](https://arxiv.org/abs/2406.17415) and [Investigating Layer Importance in Large Language Models](https://arxiv.org/abs/2409.14381).

  Furthermore, MAQ is developed based on these two studies. It calculates the contribution of each hidden layer to the output and quantizes those layers with lower importance first. It was designed to be fully compatible with the huggingface-transformers library and to be user-friendly.

- **Evaluation Metrics**:  
  The metrics module ([`metric.py`](src/maq/utils/metric.py)) provides methods to compute importance metrics, which are essential for determining quantization priorities.

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
from maq import MaqQuantizationConfig, MaqQuantizer
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
  memory_limit=0.3, 
  tokenizer=tokenizer, 
  dataset="mit-han-lab/pile-val-backup", 
  remove_columns="meta", 
  quantization_config=gptqconfig,
  use_pruning=True, 
  n_samples=32 # If python is killed, try reducing it.
)

# Load the pre-trained model with proper device mapping.
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

Loading a saved model:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model_dir")
```

## License

MAQ is licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for further details.