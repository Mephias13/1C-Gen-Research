#!/usr/bin/env python3
"""Экспорт обученного LoRA (lora_bsl/) в GGUF q4_k_m для Ollama.
Запуск из WSL venv: ~/bsl-train/bin/python export_gguf.py
LoRA-дир и выход берём из env (по умолчанию — рядом).
"""
import os
from unsloth import FastLanguageModel

LORA = os.environ.get("BSL_LORA", os.path.expanduser("~/lora_bsl"))
OUT = os.environ.get("BSL_GGUF_OUT", "/mnt/c/ClaudeProjects/bsl-pipeline/gguf_bsl")
QUANT = os.environ.get("BSL_GGUF_QUANT", "q4_k_m")

# unsloth читает base из adapter_config.json LoRA-дира; веса базы уже в кэше HF.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA, max_seq_length=1024, dtype=None, load_in_4bit=True,
)
model.save_pretrained_gguf(OUT, tokenizer, quantization_method=QUANT)
print(f"GGUF ({QUANT}) -> {OUT}")
