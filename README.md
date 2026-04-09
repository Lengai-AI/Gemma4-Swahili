# Gemma4 Swahili Fine-tuning (E2B & E24)

Fine-tuning **Gemma4 E2B (2B)** and **Gemma4 E24 (24B)** — Google's latest multimodal LLMs — specifically adapted for Swahili language instruction-following and conversation tasks.

## Overview

This project demonstrates supervised fine-tuning of the Gemma4 model family on Swahili language data, enabling high-quality conversational AI capabilities in Swahili. Fine-tuning is performed using LoRA (Low-Rank Adaptation) for parameter-efficient training, making it feasible on consumer and cloud GPUs. Two model scales are targeted:

| Variant | Parameters | Use Case                                        |
| ------- | ---------- | ----------------------------------------------- |
| **E2B** | ~2B        | Edge deployment, low-resource environments      |
| **E24** | ~24B       | High-quality inference, server-side deployment  |

## Models on Hugging Face

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-E2B%20Model-yellow)](https://huggingface.co/ngusadeep/gemma-4-2B-Swahili-llm)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-E24%20Model-orange)](https://huggingface.co/ngusadeep/gemma-4-24B-Swahili-llm)

- **E2B Model**: [ngusadeep/gemma-4-2B-Swahili-llm](https://huggingface.co/ngusadeep/gemma-4-2B-Swahili-llm)
- **E24 Model**: [ngusadeep/gemma-4-24B-Swahili-llm](https://huggingface.co/ngusadeep/gemma-4-24B-Swahili-llm)

## Features

- **Models**: Gemma4 E2B (2B parameters) and Gemma4 E24 (24B parameters)
- **Language**: Swahili (Kiswahili)
- **Task Type**: Instruction-following and conversation-based fine-tuning
- **Training Method**: LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
- **Dataset**: ~67,000 Swahili instruction-response pairs
- **Multimodal Base**: Gemma4 supports text and vision — fine-tuning targets text/Swahili NLP

## Technologies Used

- **Unsloth**: Fast and memory-efficient fine-tuning framework (2x speed, 60% less VRAM)
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained model loading and inference
- **LoRA/PEFT**: Parameter-efficient fine-tuning
- **KaggleHub**: Dataset management
- **TRL**: Training library for language models (SFTTrainer)
- **bitsandbytes**: 4-bit quantization for E24 training on consumer hardware

## Dataset

The project uses the **Swahili Instructions** dataset from Kaggle:
- **Source**: [Swahili Instructions Dataset](https://www.kaggle.com/datasets/alfaxadeyembe/swahili-instructions/data)
- **Size**: ~67,000 instruction-response pairs
- **Format**: JSON with `instruction`, `input`, `output`, and `id` fields
- **Preprocessing**: Converted to Gemma4 chat format using the `gemma4` chat template

## Installation

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers trl peft accelerate bitsandbytes kagglehub datasets
```

> For a stable pinned install:
```bash
pip install unsloth transformers==4.51.0 trl==0.15.2 peft==0.14.0 \
    accelerate bitsandbytes kagglehub datasets
```

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngusadeep/Gemma4-Swahili/blob/main/notebooks/Gemma4_E2B_E24_Swahili_Finetuning.ipynb)

Open and run the Jupyter notebook for training:
```bash
jupyter notebook notebooks/Gemma4_E2B_E24_Swahili_Finetuning.ipynb
```

### Choosing a Variant

- Start with **E2B** on a single A100 (40 GB) or T4 (16 GB with 4-bit).
- Use **E24** on an A100 80 GB or multi-GPU setup (or with 4-bit quantization on A100 40 GB).

## Inference

### Using Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Choose E2B or E24
model_name = "ngusadeep/gemma-4-2B-Swahili-llm"  # or gemma-4-24B-Swahili-llm

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Gemma4 prefers bfloat16
    device_map="auto",
)

# Prepare input using Gemma4 chat template
messages = [{"role": "user", "content": "Eleza nini maana ya uongozi."}]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Using Unsloth (Recommended)

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

# Load model — set load_in_4bit=True for E24 on 40 GB GPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ngusadeep/gemma-4-2B-Swahili-llm",  # or gemma-4-24B-Swahili-llm
    max_seq_length=2048,
    load_in_4bit=False,   # set True for E24 on limited VRAM
    dtype=None,           # auto-detects bfloat16
)

# Set up Gemma4 chat template
tokenizer = get_chat_template(tokenizer, chat_template="gemma4")

# Prepare and generate
messages = [{"role": "user", "content": "Eleza nini maana ya uongozi."}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
).removeprefix("<bos>")

model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
```

### Example Swahili Prompts

```python
"Eleza nini maana ya uongozi."          # Explanation / Uelekezaji
"Tunga hadithi fupi kuhusu safari."     # Creative writing / Ubunifu
"Ni nini tofauti kati ya mchana na usiku?"  # Q&A / Maswali na Majibu
"Andika sentensi tano kuhusu elimu."    # Instruction following / Kufuata maagizo
"Fanya muhtasari wa makala hii."        # Summarization / Muhtasari
"Tafsiri sentensi hii kwa Kiingereza."  # Translation / Tafsiri
```

### Recommended Generation Parameters

| Parameter | Value |
|-----------|-------|
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `top_k` | 64 |
| `max_new_tokens` | 256–512 |
| `dtype` | bfloat16 |

## Training Configuration

| Config | E2B | E24 |
|--------|-----|-----|
| **LoRA Rank** | 64 | 128 |
| **LoRA Alpha** | 64 | 128 |
| **Max Sequence Length** | 2048 | 2048 |
| **Batch Size (per device)** | 4 | 2 |
| **Gradient Accumulation** | 4 | 8 |
| **Learning Rate** | 5e-5 | 2e-5 |
| **Optimizer** | AdamW 8-bit | AdamW 8-bit |
| **Quantization** | None (bfloat16) | 4-bit NF4 |
| **GPU Requirement** | T4 16 GB / A100 40 GB | A100 40–80 GB |

## Results

After fine-tuning, both model variants demonstrate improved capability to:
- Understand and respond to Swahili instructions with cultural context
- Generate fluent, coherent Swahili text
- Follow complex multi-turn conversational patterns in Swahili
- Handle diverse task types: explanation, summarization, creative writing, Q&A, and translation

The **E24** variant produces notably more accurate and nuanced Swahili responses, while **E2B** provides a strong balance of quality and inference speed for resource-constrained deployments.

## Links

[![Hugging Face E2B](https://img.shields.io/badge/🤗%20Hugging%20Face-E2B%20Model-yellow)](https://huggingface.co/ngusadeep/gemma-4-2B-Swahili-llm)
[![Hugging Face E24](https://img.shields.io/badge/🤗%20Hugging%20Face-E24%20Model-orange)](https://huggingface.co/ngusadeep/gemma-4-24B-Swahili-llm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngusadeep/Gemma4-Swahili/blob/main/notebooks/Gemma4_E2B_E24_Swahili_Finetuning.ipynb)
[![Unsloth Docs](https://img.shields.io/badge/Unsloth-Docs-blue)](https://docs.unsloth.ai/)
[![Gemma4 Paper](https://img.shields.io/badge/Google-Gemma4-4285F4)](https://ai.google.dev/gemma)

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google DeepMind for the Gemma4 model family
- Unsloth team for the memory-efficient fine-tuning framework
- Kaggle dataset contributors for the Swahili Instructions dataset
- The Swahili NLP community
