#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import load_config
from rl.data import clean_completion_for_protocol, format_protocol_prompt
from rl.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Qwen3 SFT。")
    parser.add_argument("--config", default="configs/sft.yaml", help="配置文件路径")
    parser.add_argument("--train-dataset", default=None, help="覆盖训练集路径")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--epochs", type=float, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="覆盖梯度累积步数")
    return parser.parse_args()


def _build_prompt(row: dict[str, str]) -> str:
    return format_protocol_prompt(row["question"])


def _tokenize_prompt_completion(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokenization 与 prompt+completion 前缀不一致，无法安全构造 response-only labels。")

    input_ids = full_ids[:max_length]
    attention_mask = [1] * len(input_ids)
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(input_ids))
    for index in range(prompt_len):
        labels[index] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_train_dataset(path: Path, tokenizer, max_length: int) -> Dataset:
    rows = read_jsonl(path)
    tokenized_rows = []
    for row in rows:
        prompt = _build_prompt(row)
        completion = clean_completion_for_protocol(row["solution"])
        tokenized = _tokenize_prompt_completion(tokenizer, prompt, completion, max_length)
        tokenized_rows.append(tokenized)
    return Dataset.from_list(tokenized_rows)


def preview_samples(path: Path) -> None:
    rows = read_jsonl(path)[:3]
    for row in rows:
        print("=" * 80)
        print(f"id: {row['id']}")
        print(row["question"][:300])
        print("--- prompt ---")
        print(_build_prompt(row))
        print("--- completion tail ---")
        cleaned_completion = clean_completion_for_protocol(row["solution"])
        print("\n".join(cleaned_completion.splitlines()[-4:]))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_dataset_path = Path(args.train_dataset) if args.train_dataset else cfg.train_dataset
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    epochs = args.epochs if args.epochs is not None else cfg.epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else cfg.gradient_accumulation_steps
    )

    preview_samples(train_dataset_path)

    import unsloth  # noqa: F401
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    train_dataset = build_train_dataset(train_dataset_path, tokenizer, cfg.max_seq_length)

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token or tokenizer.eos_token

    training_args = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=1,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to="none",
        max_length=cfg.max_seq_length,
        eos_token=eos_token,
        pad_token=pad_token,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
