from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset
import torch
from torch.nn import functional as F

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
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(input_ids))
    for index in range(prompt_len):
        labels[index] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "length": len(input_ids),
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


def _compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_mode: str,
    profit_enabled: bool,
    profit_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_nll = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    gold_prob = torch.exp(-token_nll.detach())

    completion_lengths = valid_mask.sum(dim=-1)
    effective_sample_mask = completion_lengths.gt(0)
    effective_sample_count = int(effective_sample_mask.sum().item())
    batch_max_completion = int(completion_lengths.max().item()) if effective_sample_count > 0 else 0
    avg_completion_tokens = (
        float(completion_lengths[effective_sample_mask].float().mean().item())
        if effective_sample_count > 0
        else 0.0
    )

    if loss_mode == "opsft":
        if effective_sample_count == 0 or batch_max_completion == 0:
            loss = logits.sum() * 0.0
        else:
            sample_sums = (token_nll * valid_mask.to(token_nll.dtype)).sum(dim=-1)
            normalized = sample_sums[effective_sample_mask] / float(batch_max_completion)
            loss = normalized.mean()
        return loss, {
            "valid_tokens": float(valid_mask.sum().item()),
            "kept_tokens": float(valid_mask.sum().item()),
            "filtered_tokens": 0.0,
            "empty_batches": 1.0 if effective_sample_count == 0 else 0.0,
            "avg_gold_prob": 0.0,
            "opsft_batch_max_completion_tokens": float(batch_max_completion),
            "opsft_avg_completion_tokens": avg_completion_tokens,
            "opsft_effective_sample_count": float(effective_sample_count),
            "opsft_zero_completion_batches": 1.0 if effective_sample_count == 0 else 0.0,
        }

    keep_mask = valid_mask
    if profit_enabled:
        keep_mask = valid_mask & gold_prob.ge(profit_threshold)

    kept_count = int(keep_mask.sum().item())
    valid_count = int(valid_mask.sum().item())
    if kept_count == 0:
        loss = logits.sum() * 0.0
    else:
        loss = (token_nll * keep_mask.to(token_nll.dtype)).sum() / keep_mask.sum().to(token_nll.dtype)

    kept_gold_prob = gold_prob[keep_mask]
    avg_gold_prob = float(kept_gold_prob.mean().item()) if kept_gold_prob.numel() else 0.0
    return loss, {
        "valid_tokens": float(valid_count),
        "kept_tokens": float(kept_count),
        "filtered_tokens": float(valid_count - kept_count),
        "empty_batches": 1.0 if valid_count > 0 and kept_count == 0 else 0.0,
        "avg_gold_prob": avg_gold_prob,
        "opsft_batch_max_completion_tokens": 0.0,
        "opsft_avg_completion_tokens": 0.0,
        "opsft_effective_sample_count": 0.0,
        "opsft_zero_completion_batches": 0.0,
    }


def _extract_model_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_inputs = {"input_ids": inputs["input_ids"]}
    if "attention_mask" in inputs:
        model_inputs["attention_mask"] = inputs["attention_mask"]
    return model_inputs


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

    class ProfitSFTTrainer(SFTTrainer):
        def __init__(self, *args, loss_mode: str, profit_enabled: bool, profit_threshold: float, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.loss_mode = loss_mode
            self.profit_enabled = profit_enabled
            self.profit_threshold = profit_threshold
            self._reset_profit_stats()

        def _reset_profit_stats(self) -> None:
            self._profit_stats = {
                "valid_tokens": 0.0,
                "kept_tokens": 0.0,
                "filtered_tokens": 0.0,
                "empty_batches": 0.0,
                "avg_gold_prob_weighted_sum": 0.0,
                "opsft_batch_max_completion_tokens_sum": 0.0,
                "opsft_avg_completion_tokens_sum": 0.0,
                "opsft_effective_sample_count_sum": 0.0,
                "opsft_zero_completion_batches": 0.0,
                "step_count": 0.0,
            }

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            model_inputs = _extract_model_inputs(inputs)
            outputs = model(**model_inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss, metrics = _compute_sft_loss(
                logits,
                labels,
                loss_mode=self.loss_mode,
                profit_enabled=self.profit_enabled,
                profit_threshold=self.profit_threshold,
            )
            self._profit_stats["valid_tokens"] += metrics["valid_tokens"]
            self._profit_stats["kept_tokens"] += metrics["kept_tokens"]
            self._profit_stats["filtered_tokens"] += metrics["filtered_tokens"]
            self._profit_stats["empty_batches"] += metrics["empty_batches"]
            self._profit_stats["avg_gold_prob_weighted_sum"] += metrics["avg_gold_prob"] * metrics["kept_tokens"]
            self._profit_stats["opsft_batch_max_completion_tokens_sum"] += metrics["opsft_batch_max_completion_tokens"]
            self._profit_stats["opsft_avg_completion_tokens_sum"] += metrics["opsft_avg_completion_tokens"]
            self._profit_stats["opsft_effective_sample_count_sum"] += metrics["opsft_effective_sample_count"]
            self._profit_stats["opsft_zero_completion_batches"] += metrics["opsft_zero_completion_batches"]
            self._profit_stats["step_count"] += 1.0
            return (loss, outputs) if return_outputs else loss

        def log(self, logs: dict[str, float], start_time=None) -> None:
            if self.loss_mode == "opsft" and self._profit_stats["step_count"] > 0:
                logs = dict(logs)
                step_count = self._profit_stats["step_count"]
                logs["opsft_batch_max_completion_tokens"] = (
                    self._profit_stats["opsft_batch_max_completion_tokens_sum"] / step_count
                )
                logs["opsft_avg_completion_tokens"] = (
                    self._profit_stats["opsft_avg_completion_tokens_sum"] / step_count
                )
                logs["opsft_effective_sample_count"] = (
                    self._profit_stats["opsft_effective_sample_count_sum"] / step_count
                )
                logs["opsft_zero_completion_batch_count"] = self._profit_stats["opsft_zero_completion_batches"]
            elif self.profit_enabled and self._profit_stats["valid_tokens"] > 0:
                logs = dict(logs)
                valid_tokens = self._profit_stats["valid_tokens"]
                kept_tokens = self._profit_stats["kept_tokens"]
                logs["profit_kept_ratio"] = kept_tokens / valid_tokens
                logs["profit_filtered_ratio"] = self._profit_stats["filtered_tokens"] / valid_tokens
                logs["profit_retention_ratio"] = logs["profit_kept_ratio"]
                logs["profit_avg_gold_prob"] = (
                    self._profit_stats["avg_gold_prob_weighted_sum"] / kept_tokens
                    if kept_tokens > 0
                    else 0.0
                )
                logs["profit_empty_batch_count"] = self._profit_stats["empty_batches"]
            self._reset_profit_stats()
            return super().log(logs, start_time=start_time)

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
        logging_steps=cfg.logging_steps,
        seed=cfg.seed,
        report_to="none",
        max_length=cfg.max_seq_length,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        group_by_length=cfg.group_by_length,
        length_column_name="length",
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
    )

    trainer = ProfitSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        loss_mode=cfg.loss_mode,
        profit_enabled=cfg.profit_enabled,
        profit_threshold=cfg.profit_threshold,
    )
    trainer.train()
    output_dir = ensure_parent(cfg.output_dir / "placeholder.txt").parent
    if cfg.export_final_model:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    return output_dir
