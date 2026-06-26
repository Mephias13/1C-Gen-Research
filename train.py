#!/usr/bin/env python3
"""QLoRA-дообучение исполнителя BSL на data/dataset.jsonl (unsloth, 4-bit, под 8 ГБ).
Готов к запуску, КОГДА собрана среда (см. requirements ниже) и датасет вырос до сотен+ примеров.

Среда: NVIDIA GPU + CUDA, Python; pip install:
    pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" trl datasets
(на Windows unsloth капризен — проще WSL2; см. Task 4 / README.)

База (локед-решение): Qwen2.5-Coder-7B. На 8 ГБ 7B QLoRA впритык — если OOM, BSLGEN_BASE=unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit.
После обучения: экспорт в GGUF -> ollama create bsl-exec -f Modelfile -> подменить модель в bslgen (BSLGEN_MODEL=bsl-exec).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset.jsonl"
BASE = os.environ.get("BSLGEN_BASE", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
MAX_SEQ = int(os.environ.get("BSLGEN_MAXSEQ", "1024"))   # короткие точечные пары -> бьём память
EPOCHS = float(os.environ.get("BSLGEN_EPOCHS", "3"))
OUT = os.environ.get("BSLGEN_OUT", "lora_bsl")


def main() -> None:
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    if not DATA.exists():
        raise SystemExit(f"нет датасета {DATA} — сначала python build_dataset.py")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=MAX_SEQ, dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    ds = load_dataset("json", data_files=str(DATA), split="train")
    ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)})

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=MAX_SEQ,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            warmup_steps=5, num_train_epochs=EPOCHS, learning_rate=2e-4,
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
            logging_steps=1, seed=42, output_dir="outputs",
        ),
    )
    trainer.train()

    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"LoRA сохранён в {OUT}/")
    # Экспорт в GGUF для Ollama (раскомментировать когда нужно):
    # model.save_pretrained_gguf("gguf_bsl", tokenizer, quantization_method="q4_k_m")
    # -> Modelfile:  FROM ./gguf_bsl/unsloth.Q4_K_M.gguf   ->  ollama create bsl-exec -f Modelfile


if __name__ == "__main__":
    main()
