from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Dataset
import torch
from torch.nn import functional as F

from post_train.data import clean_solution_text
from post_train.io import ensure_parent, read_jsonl
from post_train.prompts import build_math_prompt, build_sft_prompt, render_chat_prompt
from post_train.tracking import create_trainer_callback


def _tokenize_prompt_completion(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    max_length: int,
    append_eos: bool = True,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos and eos_token_id is not None and (not completion_ids or completion_ids[-1] != eos_token_id):
        completion_ids = [*completion_ids, int(eos_token_id)]
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


def build_train_dataset(
    path: str | Path,
    tokenizer: Any,
    *,
    max_length: int,
    prompt_style: str = "default",
    use_chat_template: bool = False,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
) -> Dataset:
    rows = read_jsonl(path)
    tokenized_rows: list[dict[str, list[int]]] = []
    for row in rows:
        prompt = build_math_prompt(str(row["question"]), style=prompt_style)  # type: ignore[arg-type]
        if use_chat_template:
            prompt = render_chat_prompt(
                tokenizer,
                prompt,
                system_prompt=system_prompt,
                assistant_prefill=assistant_prefill,
            )
        elif assistant_prefill:
            prompt += assistant_prefill
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


def _dense_lightning_teacher_fields(row: dict[str, Any], *, max_length: int, distill_top_k: int) -> dict[str, Any]:
    input_ids = list(row["input_ids"])
    response_mask = [int(value) for value in row["response_mask"]]
    if len(input_ids) != len(response_mask):
        raise ValueError(f"input_ids 与 response_mask 长度不一致：id={row.get('id')}")
    if len(input_ids) > max_length:
        raise ValueError(f"Lightning-OPD 样本超过 max_seq_length：id={row.get('id')} len={len(input_ids)}")

    response_positions = [index for index, value in enumerate(response_mask) if value]
    token_logprobs = list(row["teacher_token_logprobs"])
    if len(response_positions) != len(token_logprobs):
        raise ValueError(f"teacher_token_logprobs 与 response token 数不一致：id={row.get('id')}")

    topk_ids = row.get("teacher_topk_token_ids", [])
    topk_logprobs = row.get("teacher_topk_logprobs", [])
    stored_top_k = len(topk_ids[0]) if topk_ids else 0
    if distill_top_k > 1 and stored_top_k < distill_top_k:
        raise ValueError(
            f"distill_top_k={distill_top_k} 大于离线保存 topK={stored_top_k}：id={row.get('id')}，需要重新生成 teacher logits。"
        )
    if topk_ids and (len(topk_ids) != len(response_positions) or len(topk_logprobs) != len(response_positions)):
        raise ValueError(f"teacher_topk_* 与 response token 数不一致：id={row.get('id')}")

    labels = [-100 if value == 0 else token_id for token_id, value in zip(input_ids, response_mask, strict=True)]
    teacher_token_logprobs = [0.0] * len(input_ids)
    teacher_topk_token_ids = [[0] * stored_top_k for _ in input_ids]
    teacher_topk_logprobs = [[0.0] * stored_top_k for _ in input_ids]
    teacher_topk_mask = [[0] * stored_top_k for _ in input_ids]
    for response_index, pos in enumerate(response_positions):
        teacher_token_logprobs[pos] = float(token_logprobs[response_index])
        if stored_top_k:
            teacher_topk_token_ids[pos] = [int(value) for value in topk_ids[response_index]]
            teacher_topk_logprobs[pos] = [float(value) for value in topk_logprobs[response_index]]
            teacher_topk_mask[pos] = [1] * stored_top_k

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "response_mask": response_mask,
        "teacher_token_logprobs": teacher_token_logprobs,
        "teacher_topk_token_ids": teacher_topk_token_ids,
        "teacher_topk_logprobs": teacher_topk_logprobs,
        "teacher_topk_mask": teacher_topk_mask,
        "length": len(input_ids),
    }


def build_lightning_opd_dataset(path: str | Path, *, max_length: int, distill_top_k: int) -> Dataset:
    rows = read_jsonl(path)
    tokenized_rows = [
        _dense_lightning_teacher_fields(row, max_length=max_length, distill_top_k=distill_top_k)
        for row in rows
    ]
    return Dataset.from_list(tokenized_rows)


class LightningOPDDataCollator:
    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        stored_top_k = max(
            (len(token_topk) for feature in features for token_topk in feature.get("teacher_topk_token_ids", [])),
            default=0,
        )

        batch: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "response_mask": [],
            "teacher_token_logprobs": [],
            "teacher_topk_token_ids": [],
            "teacher_topk_logprobs": [],
            "teacher_topk_mask": [],
        }
        for feature in features:
            length = len(feature["input_ids"])
            pad_len = max_len - length
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
            batch["response_mask"].append(feature["response_mask"] + [0] * pad_len)
            batch["teacher_token_logprobs"].append(feature["teacher_token_logprobs"] + [0.0] * pad_len)

            topk_ids = [list(values) + [0] * (stored_top_k - len(values)) for values in feature["teacher_topk_token_ids"]]
            topk_logprobs = [
                list(values) + [0.0] * (stored_top_k - len(values)) for values in feature["teacher_topk_logprobs"]
            ]
            topk_mask = [list(values) + [0] * (stored_top_k - len(values)) for values in feature["teacher_topk_mask"]]
            batch["teacher_topk_token_ids"].append(topk_ids + [[0] * stored_top_k for _ in range(pad_len)])
            batch["teacher_topk_logprobs"].append(topk_logprobs + [[0.0] * stored_top_k for _ in range(pad_len)])
            batch["teacher_topk_mask"].append(topk_mask + [[0] * stored_top_k for _ in range(pad_len)])

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "response_mask": torch.tensor(batch["response_mask"], dtype=torch.bool),
            "teacher_token_logprobs": torch.tensor(batch["teacher_token_logprobs"], dtype=torch.float32),
            "teacher_topk_token_ids": torch.tensor(batch["teacher_topk_token_ids"], dtype=torch.long),
            "teacher_topk_logprobs": torch.tensor(batch["teacher_topk_logprobs"], dtype=torch.float32),
            "teacher_topk_mask": torch.tensor(batch["teacher_topk_mask"], dtype=torch.bool),
        }


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
            "dft_loss_scale": 0.0,
            "opsft_batch_max_completion_tokens": float(batch_max_completion),
            "opsft_avg_completion_tokens": avg_completion_tokens,
            "opsft_effective_sample_count": float(effective_sample_count),
            "opsft_zero_completion_batches": 1.0 if effective_sample_count == 0 else 0.0,
        }

    valid_count = int(valid_mask.sum().item())
    if loss_mode == "dft":
        if valid_count == 0:
            loss = logits.sum() * 0.0
            avg_gold_prob = 0.0
        else:
            token_loss = gold_prob * token_nll
            loss = (token_loss * valid_mask.to(token_loss.dtype)).sum() / valid_mask.sum().to(token_loss.dtype)
            avg_gold_prob = float(gold_prob[valid_mask].mean().item())
        return loss, {
            "valid_tokens": float(valid_count),
            "kept_tokens": float(valid_count),
            "filtered_tokens": 0.0,
            "empty_batches": 1.0 if valid_count == 0 else 0.0,
            "avg_gold_prob": avg_gold_prob,
            "dft_loss_scale": avg_gold_prob,
            "opsft_batch_max_completion_tokens": 0.0,
            "opsft_avg_completion_tokens": 0.0,
            "opsft_effective_sample_count": 0.0,
            "opsft_zero_completion_batches": 0.0,
        }

    keep_mask = valid_mask
    if profit_enabled:
        keep_mask = valid_mask & gold_prob.ge(profit_threshold)

    kept_count = int(keep_mask.sum().item())
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
        "dft_loss_scale": 0.0,
        "opsft_batch_max_completion_tokens": 0.0,
        "opsft_avg_completion_tokens": 0.0,
        "opsft_effective_sample_count": 0.0,
        "opsft_zero_completion_batches": 0.0,
    }


def _compute_lightning_opd_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_token_logprobs: torch.Tensor,
    teacher_topk_token_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_mask: torch.Tensor,
    *,
    distill_top_k: int,
    topk_kd_weight: float,
    opd_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_input_ids = input_ids[..., 1:].contiguous()
    shift_response_mask = response_mask[..., 1:].contiguous()
    shift_teacher_logprobs = teacher_token_logprobs[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    student_token_logprobs = log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
    valid_mask = shift_response_mask
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        zero = logits.sum() * 0.0
        return zero, {
            "valid_tokens": 0.0,
            "opd_loss": 0.0,
            "topk_kd_loss": 0.0,
            "avg_teacher_logprob": 0.0,
            "avg_student_logprob": 0.0,
        }

    advantage = shift_teacher_logprobs - student_token_logprobs.detach()
    opd_token_loss = -advantage * student_token_logprobs
    opd_loss = (opd_token_loss * valid_mask.to(opd_token_loss.dtype)).sum() / valid_mask.sum().to(opd_token_loss.dtype)

    topk_kd_loss = logits.sum() * 0.0
    if topk_kd_weight > 0.0 and distill_top_k > 1:
        stored_top_k = teacher_topk_token_ids.shape[-1]
        if distill_top_k > stored_top_k:
            raise ValueError(f"distill_top_k={distill_top_k} 大于 batch 中离线保存 topK={stored_top_k}。")
        topk_ids = teacher_topk_token_ids[..., 1:, :distill_top_k].contiguous()
        topk_logprobs = teacher_topk_logprobs[..., 1:, :distill_top_k].contiguous()
        topk_mask = teacher_topk_mask[..., 1:, :distill_top_k].contiguous() & valid_mask.unsqueeze(-1)
        masked_teacher_logprobs = topk_logprobs.masked_fill(~topk_mask, torch.finfo(topk_logprobs.dtype).min)
        teacher_probs = torch.softmax(masked_teacher_logprobs, dim=-1).masked_fill(~topk_mask, 0.0)
        student_topk_logprobs = log_probs.gather(dim=-1, index=topk_ids.clamp_min(0))
        kd_per_token = -(teacher_probs * student_topk_logprobs).sum(dim=-1)
        kd_token_mask = topk_mask.any(dim=-1)
        if bool(kd_token_mask.any().item()):
            topk_kd_loss = (kd_per_token * kd_token_mask.to(kd_per_token.dtype)).sum() / kd_token_mask.sum().to(
                kd_per_token.dtype
            )

    loss = opd_weight * opd_loss + topk_kd_weight * topk_kd_loss
    return loss, {
        "valid_tokens": float(valid_count),
        "opd_loss": float(opd_loss.detach().item()),
        "topk_kd_loss": float(topk_kd_loss.detach().item()),
        "avg_teacher_logprob": float(shift_teacher_logprobs[valid_mask].detach().mean().item()),
        "avg_student_logprob": float(student_token_logprobs[valid_mask].detach().mean().item()),
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
        completion = clean_solution_text(str(row.get("solution", row.get("response", ""))))
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


class ProfitStatsMixin:
    def _init_profit_controls(
        self,
        *,
        loss_mode: str,
        profit_enabled: bool,
        profit_threshold: float,
        distill_top_k: int,
        topk_kd_weight: float,
        opd_weight: float,
    ) -> None:
        self.loss_mode = loss_mode
        self.profit_enabled = profit_enabled
        self.profit_threshold = profit_threshold
        self.distill_top_k = distill_top_k
        self.topk_kd_weight = topk_kd_weight
        self.opd_weight = opd_weight
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
            "lightning_opd_valid_tokens": 0.0,
            "lightning_opd_loss_sum": 0.0,
            "lightning_opd_topk_kd_loss_sum": 0.0,
            "lightning_opd_teacher_logprob_sum": 0.0,
            "lightning_opd_student_logprob_sum": 0.0,
            "step_count": 0.0,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        model_inputs = _extract_model_inputs(inputs)
        outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if self.loss_mode == "lightning_opd":
            loss, metrics = _compute_lightning_opd_loss(
                logits,
                inputs["input_ids"],
                inputs["response_mask"],
                inputs["teacher_token_logprobs"],
                inputs["teacher_topk_token_ids"],
                inputs["teacher_topk_logprobs"],
                inputs["teacher_topk_mask"],
                distill_top_k=self.distill_top_k,
                topk_kd_weight=self.topk_kd_weight,
                opd_weight=self.opd_weight,
            )
            self._profit_stats["lightning_opd_valid_tokens"] += metrics["valid_tokens"]
            self._profit_stats["lightning_opd_loss_sum"] += metrics["opd_loss"]
            self._profit_stats["lightning_opd_topk_kd_loss_sum"] += metrics["topk_kd_loss"]
            self._profit_stats["lightning_opd_teacher_logprob_sum"] += metrics["avg_teacher_logprob"]
            self._profit_stats["lightning_opd_student_logprob_sum"] += metrics["avg_student_logprob"]
        else:
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
        if self.loss_mode == "lightning_opd" and self._profit_stats["step_count"] > 0:
            logs = dict(logs)
            step_count = self._profit_stats["step_count"]
            logs["lightning_opd_valid_tokens"] = self._profit_stats["lightning_opd_valid_tokens"]
            logs["lightning_opd_opd_loss"] = self._profit_stats["lightning_opd_loss_sum"] / step_count
            logs["lightning_opd_topk_kd_loss"] = self._profit_stats["lightning_opd_topk_kd_loss_sum"] / step_count
            logs["lightning_opd_avg_teacher_logprob"] = (
                self._profit_stats["lightning_opd_teacher_logprob_sum"] / step_count
            )
            logs["lightning_opd_avg_student_logprob"] = (
                self._profit_stats["lightning_opd_student_logprob_sum"] / step_count
            )
            logs["lightning_opd_distill_top_k"] = float(self.distill_top_k)
        elif self.loss_mode == "dft" and self._profit_stats["valid_tokens"] > 0:
            logs = dict(logs)
            kept_tokens = self._profit_stats["kept_tokens"]
            logs["dft_valid_tokens"] = self._profit_stats["valid_tokens"]
            logs["dft_avg_gold_prob"] = (
                self._profit_stats["avg_gold_prob_weighted_sum"] / kept_tokens if kept_tokens > 0 else 0.0
            )
        elif self.loss_mode == "opsft" and self._profit_stats["step_count"] > 0:
            logs = dict(logs)
            step_count = self._profit_stats["step_count"]
            logs["opsft_batch_max_completion_tokens"] = (
                self._profit_stats["opsft_batch_max_completion_tokens_sum"] / step_count
            )
            logs["opsft_avg_completion_tokens"] = self._profit_stats["opsft_avg_completion_tokens_sum"] / step_count
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
                self._profit_stats["avg_gold_prob_weighted_sum"] / kept_tokens if kept_tokens > 0 else 0.0
            )
            logs["profit_empty_batch_count"] = self._profit_stats["empty_batches"]
        self._reset_profit_stats()
        return super().log(logs, start_time=start_time)


def _build_training_args(cfg, tokenizer: Any):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        max_grad_norm=cfg.max_grad_norm,
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
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )


def _train_sft_unsloth(cfg) -> Path:
    import unsloth  # noqa: F401
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    class ProfitSFTTrainer(ProfitStatsMixin, SFTTrainer):
        def __init__(
            self,
            *args,
            loss_mode: str,
            profit_enabled: bool,
            profit_threshold: float,
            distill_top_k: int,
            topk_kd_weight: float,
            opd_weight: float,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._init_profit_controls(
                loss_mode=loss_mode,
                profit_enabled=profit_enabled,
                profit_threshold=profit_threshold,
                distill_top_k=distill_top_k,
                topk_kd_weight=topk_kd_weight,
                opd_weight=opd_weight,
            )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    if cfg.loss_mode == "lightning_opd":
        train_dataset = build_lightning_opd_dataset(
            cfg.train_dataset,
            max_length=cfg.max_seq_length,
            distill_top_k=cfg.distill_top_k,
        )
        data_collator = LightningOPDDataCollator(pad_token_id=tokenizer.pad_token_id)
    else:
        train_dataset = build_train_dataset(
            cfg.train_dataset,
            tokenizer,
            max_length=cfg.max_seq_length,
            prompt_style=cfg.prompt_style,
            use_chat_template=cfg.use_chat_template,
            system_prompt=cfg.system_prompt,
            assistant_prefill=cfg.assistant_prefill,
        )
        data_collator = None

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

    training_args = _build_training_args(cfg, tokenizer)

    trainer = ProfitSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
        loss_mode=cfg.loss_mode,
        profit_enabled=cfg.profit_enabled,
        profit_threshold=cfg.profit_threshold,
        distill_top_k=cfg.distill_top_k,
        topk_kd_weight=cfg.topk_kd_weight,
        opd_weight=cfg.opd_weight,
    )
    mlflow_callback = create_trainer_callback()
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)
    trainer.train()
    output_dir = ensure_parent(cfg.output_dir / "placeholder.txt").parent
    if cfg.export_final_model:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _train_sft_trl_peft(cfg) -> Path:
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer

    class ProfitSFTTrainer(ProfitStatsMixin, SFTTrainer):
        def __init__(
            self,
            *args,
            loss_mode: str,
            profit_enabled: bool,
            profit_threshold: float,
            distill_top_k: int,
            topk_kd_weight: float,
            opd_weight: float,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._init_profit_controls(
                loss_mode=loss_mode,
                profit_enabled=profit_enabled,
                profit_threshold=profit_threshold,
                distill_top_k=distill_top_k,
                topk_kd_weight=topk_kd_weight,
                opd_weight=opd_weight,
            )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    model_path = Path(cfg.model_name)
    is_adapter_checkpoint = (model_path / "adapter_config.json").exists()
    base_model_name = cfg.model_name
    if is_adapter_checkpoint:
        peft_source_config = PeftConfig.from_pretrained(cfg.model_name)
        base_model_name = cfg.adapter_base_model or peft_source_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )
    model = prepare_model_for_kbit_training(model)
    if is_adapter_checkpoint:
        model = PeftModel.from_pretrained(model, cfg.model_name, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)

    if cfg.loss_mode == "lightning_opd":
        train_dataset = build_lightning_opd_dataset(
            cfg.train_dataset,
            max_length=cfg.max_seq_length,
            distill_top_k=cfg.distill_top_k,
        )
        data_collator = LightningOPDDataCollator(pad_token_id=tokenizer.pad_token_id)
    else:
        train_dataset = build_train_dataset(
            cfg.train_dataset,
            tokenizer,
            max_length=cfg.max_seq_length,
            prompt_style=cfg.prompt_style,
            use_chat_template=cfg.use_chat_template,
            system_prompt=cfg.system_prompt,
            assistant_prefill=cfg.assistant_prefill,
        )
        data_collator = None
    training_args = _build_training_args(cfg, tokenizer)
    trainer = ProfitSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
        loss_mode=cfg.loss_mode,
        profit_enabled=cfg.profit_enabled,
        profit_threshold=cfg.profit_threshold,
        distill_top_k=cfg.distill_top_k,
        topk_kd_weight=cfg.topk_kd_weight,
        opd_weight=cfg.opd_weight,
    )
    mlflow_callback = create_trainer_callback()
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)
    trainer.train()
    output_dir = ensure_parent(cfg.output_dir / "placeholder.txt").parent
    if cfg.export_final_model:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    return output_dir


def train_sft(cfg) -> Path:
    if cfg.backend == "trl_peft":
        return _train_sft_trl_peft(cfg)
    return _train_sft_unsloth(cfg)
