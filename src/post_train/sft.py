from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset

from post_train.data import clean_solution_text
from post_train.io import ensure_parent, read_jsonl
from post_train.prompts import build_sft_prompt


def _tokenize_prompt_completion(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt 与 prompt+completion 的 tokenizer 前缀不一致，无法构造 response-only labels。")

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


def build_train_dataset(path: str | Path, tokenizer: Any, *, max_length: int) -> Dataset:
    rows = read_jsonl(path)
    tokenized_rows: list[dict[str, list[int]]] = []
    for row in rows:
        prompt = build_sft_prompt(str(row["question"]))
        completion = clean_solution_text(str(row["solution"]))
        tokenized_rows.append(
            _tokenize_prompt_completion(
                tokenizer,
                prompt=prompt,
                completion=completion,
                max_length=max_length,
            )
        )
    return Dataset.from_list(tokenized_rows)


def preview_training_samples(path: str | Path, *, count: int = 3) -> str:
    rows = read_jsonl(path)[:count]
    parts: list[str] = []
    for row in rows:
        completion = clean_solution_text(str(row["solution"]))
        parts.append(
            "\n".join(
                [
                    "=" * 80,
                    f"id: {row['id']}",
                    str(row["question"])[:300],
                    "--- prompt ---",
                    build_sft_prompt(str(row["question"])),
                    "--- completion tail ---",
                    "\n".join(completion.splitlines()[-4:]),
                ]
            )
        )
    return "\n".join(parts)


def train_sft(cfg) -> Path:
    import unsloth  # noqa: F401
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    train_dataset = build_train_dataset(cfg.train_dataset, tokenizer, max_length=cfg.max_seq_length)

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=1,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to="none",
        max_length=cfg.max_seq_length,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )
    trainer.train()
    output_dir = ensure_parent(cfg.output_dir / "placeholder.txt").parent
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
