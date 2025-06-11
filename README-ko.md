# MAQ: 가중치 양자화 라이브러리

MAQ는 최신 딥러닝 모델의 가중치 양자화를 위한 경량 라이브러리입니다. 이 라이브러리는 유연한 양자화 방법과 유용한 구성 및 메트릭 유틸리티를 제공합니다.

## 특징

- **유연한 양자화 방법**:  
  다음 양자화 라이브러리들을 지원합니다:
  * GPTQ ([`maq_gptq.py`](src/maq/utils/maq_gptq.py))
  * AWQ ([`maq_awq.py`](src/maq/utils/maq_awq.py)) – 현재 4비트 지원 한계로 인해 비활성화되어 있습니다.  
    추후 transformers.quantizers의 모든 방법을 지원할 계획입니다.

- **메모리 기반 양자화 접근법**:  
  전통적인 비트 폭 중심의 양자화와 달리, MAQ는 사용 가능한 메모리에 따라 양자화를 조정합니다. 이 방식에서는 각 모듈의 중요도를 산정하며, 중요도가 낮은 모듈이 우선적으로 양자화됩니다. 방법론에 관한 자세한 설명은 관련 논문 ([https://arxiv.org/abs/2406.17415])([https://arxiv.org/abs/2409.14381]) 을 참조하세요.

- **구성 도구**:  
  구성 유틸리티 ([`config.py`](src/maq/utils/config.py))를 사용하여 양자화 파라미터를 손쉽게 미세 조정할 수 있습니다.

- **평가 메트릭**:  
  메트릭 모듈 ([`metric.py`](src/maq/utils/metric.py))을 통해 양자화 평가를 내장 지원합니다.

## 설치

프로젝트 루트에서 MAQ를 설치하세요:

```bash
pip install .
```

개발을 위해 편집 모드로 설치하려면:

```bash
pip install -e .
```

## 사용 방법

다음은 시작하는 데 도움이 되는 기본 예제입니다:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_MAQ import MaqQuantizationConfig, MaqQuantizer
from transformers.utils.quantization_config import GPTQConfig

# 양자화할 모델을 지정합니다.
model_name = "meta-llama/Meta-Llama-3-8B"

# 토크나이저를 로드합니다.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPTQ 양자화 구성을 생성합니다.
# 참고: 여기서 'bits' 파라미터는 양자화 프로세스에서 사용되지 않습니다.
gptqconfig = GPTQConfig(bits=4, dataset='wikitext2')

# 메모리 제한 및 가지치기를 옵션으로 하는 MAQ 양자화 구성을 정의합니다.
quantization_config = MaqQuantizationConfig(
    memory_limit=0.2,
    tokenizer=tokenizer,
    dataset="pileval",
    quantization_config=gptqconfig,
    use_pruning=True
)

# CPU 메모리 사용량을 최소화하고 적절한 장치 매핑을 사용하여 사전 학습된 모델을 로드합니다.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

# 구성에 따라 MAQ 양자화기를 초기화하고 양자화를 적용합니다.
quantizer = MaqQuantizer(quantization_config, metric='lim')
quantizer.quantize_model(model)

# 양자화된 모델을 디스크에 저장합니다.
quantizer.save_model(model, f"{model_name}_MAQ")
```

추가 유틸리티와 고급 사용 예제는 [`src/maq/utils/`](src/maq/utils/) 디렉토리에서 확인할 수 있습니다.

## 기여

기여와 개선은 언제나 환영합니다! MAQ에 기여하는 방법에 대한 자세한 내용은 [CONTRIBUTING.md](./CONTRIBUTING.md)를 참조하세요.

## 라이선스

MAQ는 Apache License, Version 2.0에 따라 라이선스가 부여됩니다. 자세한 사항은 [LICENSE](./LICENSE) 파일을 확인하세요.

---

이 문서는 GitHub Copilot의 o3-mini를 사용하여 생성되었습니다.
```
```<!-- filepath: /mnt/d/BoxBy/workspace/Dynamic_Quantization/maq/README.md -->
```md
# MAQ: 가중치 양자화 라이브러리

MAQ는 최신 딥러닝 모델의 가중치 양자화를 위한 경량 라이브러리입니다. 이 라이브러리는 유연한 양자화 방법과 유용한 구성 및 메트릭 유틸리티를 제공합니다.

## 특징

- **유연한 양자화 방법**:  
  다음 양자화 라이브러리들을 지원합니다:
  * GPTQ ([`maq_gptq.py`](src/maq/utils/maq_gptq.py))
  * AWQ ([`maq_awq.py`](src/maq/utils/maq_awq.py)) – 현재 4비트 지원 한계로 인해 비활성화되어 있습니다.  
    추후 transformers.quantizers의 모든 방법을 지원할 계획입니다.

- **메모리 기반 양자화 접근법**:  
  전통적인 비트 폭 중심의 양자화와 달리, MAQ는 사용 가능한 메모리에 따라 양자화를 조정합니다. 이 방식에서는 각 모듈의 중요도를 산정하며, 중요도가 낮은 모듈이 우선적으로 양자화됩니다. 방법론에 관한 자세한 설명은 관련 논문 ([https://arxiv.org/abs/2406.17415])을 참조하세요.

- **구성 도구**:  
  구성 유틸리티 ([`config.py`](src/maq/utils/config.py))를 사용하여 양자화 파라미터를 손쉽게 미세 조정할 수 있습니다.

- **평가 메트릭**:  
  메트릭 모듈 ([`metric.py`](src/maq/utils/metric.py))을 통해 양자화 평가를 내장 지원합니다.

## 설치

프로젝트 루트에서 MAQ를 설치하세요:

```bash
pip install .
```

개발을 위해 편집 모드로 설치하려면:

```bash
pip install -e .
```

## 사용 방법

다음은 시작하는 데 도움이 되는 기본 예제입니다:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_MAQ import MaqQuantizationConfig, MaqQuantizer
from transformers.utils.quantization_config import GPTQConfig

# 양자화할 모델을 지정합니다.
model_name = "meta-llama/Meta-Llama-3-8B"

# 토크나이저를 로드합니다.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPTQ 양자화 구성을 생성합니다.
# 참고: 여기서 'bits' 파라미터는 양자화 프로세스에서 사용되지 않습니다.
gptqconfig = GPTQConfig(bits=4, dataset='wikitext2')

# 메모리 제한 및 가지치기를 옵션으로 하는 MAQ 양자화 구성을 정의합니다.
quantization_config = MaqQuantizationConfig(
    memory_limit=0.2,
    tokenizer=tokenizer,
    dataset="pileval",
    quantization_config=gptqconfig,
    use_pruning=True
)

# CPU 메모리 사용량을 최소화하고 적절한 장치 매핑을 사용하여 사전 학습된 모델을 로드합니다.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

# 구성에 따라 MAQ 양자화기를 초기화하고 양자화를 적용합니다.
quantizer = MaqQuantizer(quantization_config, metric='lim')
quantizer.quantize_model(model)

# 양자화된 모델을 디스크에 저장합니다.
quantizer.save_model(model, f"{model_name}_MAQ")
```

추가 유틸리티와 고급 사용 예제는 [`src/maq/utils/`](src/maq/utils/) 디렉토리에서 확인할 수 있습니다.

## 라이선스

MAQ는 Apache License, Version 2.0에 따라 라이선스가 부여됩니다. 자세한 사항은 [LICENSE](./LICENSE) 파일을 확인하세요.

---

이 레퍼지토리는 GitHub Copilot의 o3-mini를 활용하여 생성되었습니다.
