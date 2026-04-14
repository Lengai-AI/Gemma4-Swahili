# Gemma4 Swahili Fine-tuning (E2B & E24)

![cover](assets/md_jop19q_d41bc9ee4117b91c384381940b55c085a1486ae3.jpg)

Fine-tuning **Gemma4 E2B (2B)** and **Gemma4 E24 (24B)** for Swahili language instruction-following using LoRA + Unsloth.

[![Hugging Face E2B](https://img.shields.io/badge/🤗-E2B%20Model-yellow)](https://huggingface.co/ngusadeep/gemma-4-2B-Swahili-llm)
[![Hugging Face E24](https://img.shields.io/badge/🤗-E24%20Model-orange)](https://huggingface.co/ngusadeep/gemma-4-24B-Swahili-llm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngusadeep/Gemma4-Swahili/blob/main/finetuning-nb/Gemma4_E2B_E24_Swahili_Finetuning.ipynb)

## Models

| Variant | Parameters | HuggingFace |
| ------- | ---------- | ----------- |
| **E2B** | ~2B | [ngusadeep/gemma-4-2B-Swahili-llm](https://huggingface.co/ngusadeep/gemma-4-2B-Swahili-llm) |
| **E24** | ~24B | [ngusadeep/gemma-4-24B-Swahili-llm](https://huggingface.co/ngusadeep/gemma-4-24B-Swahili-llm) |

## Dataset

Translated from [`mlabonne/FineTome-100k`](https://huggingface.co/datasets/mlabonne/FineTome-100k) → **17,982 Swahili instruction-response pairs** via GPT-4o-mini Batch API. Dataset curated and published by **[Lengai AI Lab](https://huggingface.co/lengai-lab)**.

| Format | Dataset |
| ------ | ------- |
| Alpaca | [lengai-ai/Swahili-FineTome-Dataset](https://huggingface.co/datasets/lengai-ai/Swahili-FineTome-Dataset) |
| ShareGPT | [lengai-ai/Swahili-FineTome-Dataset-sharegpt](https://huggingface.co/datasets/lengai-ai/Swahili-FineTome-Dataset-sharegpt) |

## Training Configuration

| Config | E2B | E24 |
| ------ | --- | --- |
| LoRA Rank | 64 | 128 |
| Max Seq Length | 2048 | 2048 |
| Batch Size | 4 | 2 |
| Learning Rate | 5e-5 | 2e-5 |
| Quantization | bfloat16 | 4-bit NF4 |
| GPU | T4 16 GB / A100 40 GB | A100 40–80 GB |

## Inference

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "ngusadeep/gemma-4-2B-Swahili-llm",
    max_seq_length = 2048,
    load_in_4bit   = False,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma4")

messages = [{"role": "user", "content": "Eleza nini maana ya uongozi."}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
).removeprefix("<bos>")

model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens = 256,
    temperature    = 1.0,
    top_p          = 0.95,
    streamer       = TextStreamer(tokenizer, skip_prompt=True),
)
```

## Citation

If you use these models or the dataset, please cite:

```bibtex
@misc{gemma4_swahili_2026,
  author    = {Samwel, Ngusa},
  title     = {Gemma4 Swahili: Fine-tuning Gemma4 E2B \& E24 for Swahili},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/ngusadeep/gemma-4-2B-Swahili-llm}
}

@dataset{finetome_20k_sw_2026,
  author    = {Samwel, Ngusa},
  title     = {FineTome-20k-sw: A Swahili Instruction Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/lengai-ai/Swahili-FineTome-Dataset}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google DeepMind for Gemma4
- [Unsloth](https://docs.unsloth.ai/) for the fine-tuning framework
- [lengai-ai/Swahili-FineTome-Dataset](https://huggingface.co/datasets/lengai-ai/Swahili-FineTome-Dataset) — Swahili instruction dataset
- [Lengai AI Lab](https://huggingface.co/lengai-ai) — Swahili LLM Research
