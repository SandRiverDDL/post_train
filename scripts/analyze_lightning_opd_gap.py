#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.io import ensure_parent, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 Lightning OPD 数据中 teacher/student token logprob gap。")
    parser.add_argument("--dataset", required=True, help="Lightning OPD train.jsonl")
    parser.add_argument("--student-model", required=True, help="OPD 前 student 模型或 LoRA adapter 目录")
    parser.add_argument("--adapter-base-model", default=None, help="student 是 LoRA adapter 时使用的 base model")
    parser.add_argument("--output", required=True, help="输出 report.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="使用 4bit 加载，省显存但 logprob 会有量化误差")
    parser.add_argument("--logprob-chunk-size", type=int, default=256, help="按 token chunk 计算 logprob，避免大显存峰值")
    parser.add_argument("--overlap-top-k", type=int, default=16, help="统计 teacher/student topK 交集，0 表示关闭")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_student_model(args: argparse.Namespace):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = Path(args.student_model)
    is_adapter = (model_path / "adapter_config.json").exists()
    base_model = args.student_model
    if is_adapter:
        peft_cfg = PeftConfig.from_pretrained(args.student_model)
        base_model = args.adapter_base_model or peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    dtype = _dtype(args.dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if is_adapter:
        model = PeftModel.from_pretrained(model, args.student_model, is_trainable=False)
    if not args.load_in_4bit:
        model.to(args.device)
    model.eval()
    return model, tokenizer


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[index])


def _collate(rows: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    response_mask: list[list[int]] = []
    teacher_token_logprobs: list[list[float]] = []
    teacher_topk_token_ids: list[list[list[int]]] = []
    teacher_topk_mask: list[list[list[int]]] = []
    stored_top_k = max(
        (len(values) for row in rows for values in row.get("teacher_topk_token_ids", [])),
        default=0,
    )
    for row in rows:
        ids = list(row["input_ids"])
        mask = list(row["response_mask"])
        teacher_values = list(row["teacher_token_logprobs"])
        topk_values = list(row.get("teacher_topk_token_ids", []))
        response_positions = [idx for idx, value in enumerate(mask) if value]
        if len(response_positions) != len(teacher_values):
            raise ValueError(f"teacher_token_logprobs 与 response token 数不一致：id={row.get('id')}")
        if topk_values and len(response_positions) != len(topk_values):
            raise ValueError(f"teacher_topk_token_ids 与 response token 数不一致：id={row.get('id')}")
        dense_teacher = [0.0] * len(ids)
        dense_topk = [[0] * stored_top_k for _ in ids]
        dense_topk_mask = [[0] * stored_top_k for _ in ids]
        for response_index, pos in enumerate(response_positions):
            dense_teacher[pos] = float(teacher_values[response_index])
            if topk_values:
                token_topk = [int(value) for value in topk_values[response_index]]
                dense_topk[pos] = token_topk + [0] * (stored_top_k - len(token_topk))
                dense_topk_mask[pos] = [1] * len(token_topk) + [0] * (stored_top_k - len(token_topk))
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        response_mask.append(mask + [0] * pad_len)
        teacher_token_logprobs.append(dense_teacher + [0.0] * pad_len)
        teacher_topk_token_ids.append(dense_topk + [[0] * stored_top_k for _ in range(pad_len)])
        teacher_topk_mask.append(dense_topk_mask + [[0] * stored_top_k for _ in range(pad_len)])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
        "teacher_token_logprobs": torch.tensor(teacher_token_logprobs, dtype=torch.float32),
        "teacher_topk_token_ids": torch.tensor(teacher_topk_token_ids, dtype=torch.long),
        "teacher_topk_mask": torch.tensor(teacher_topk_mask, dtype=torch.bool),
    }


def _gather_token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        chunk_logits = logits[:, start:end, :].float()
        chunk_targets = target_ids[:, start:end]
        target_logits = chunk_logits.gather(dim=-1, index=chunk_targets.unsqueeze(-1)).squeeze(-1)
        parts.append(target_logits - torch.logsumexp(chunk_logits, dim=-1))
        del chunk_logits, chunk_targets, target_logits
    return torch.cat(parts, dim=1)


def _topk_overlap_counts(
    logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int,
    chunk_size: int,
) -> tuple[int, int, int, list[float]]:
    if top_k <= 0 or teacher_topk_ids.shape[-1] == 0:
        return 0, 0, 0, []
    k = min(top_k, teacher_topk_ids.shape[-1], logits.shape[-1])
    overlap_sum = 0
    full_overlap = 0
    token_count = 0
    ratios: list[float] = []
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        chunk_logits = logits[:, start:end, :]
        student_topk = torch.topk(chunk_logits, k=k, dim=-1).indices
        teacher_ids = teacher_topk_ids[:, start:end, :k]
        teacher_mask = teacher_topk_mask[:, start:end, :k]
        token_mask = valid_mask[:, start:end] & teacher_mask.any(dim=-1)
        if not bool(token_mask.any().item()):
            continue
        matches = (student_topk.unsqueeze(-1) == teacher_ids.unsqueeze(-2)) & teacher_mask.unsqueeze(-2)
        overlap = matches.any(dim=-1).sum(dim=-1)
        valid_overlap = overlap[token_mask]
        overlap_sum += int(valid_overlap.sum().item())
        full_overlap += int((valid_overlap == k).sum().item())
        token_count += int(token_mask.sum().item())
        ratios.extend((valid_overlap.float() / float(k)).detach().cpu().tolist())
    return overlap_sum, full_overlap, token_count, ratios


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("dataset 为空。")

    model, tokenizer = _load_student_model(args)
    device = next(model.parameters()).device

    positive = 0
    negative = 0
    zero = 0
    total = 0
    deltas: list[float] = []
    teacher_values_all: list[float] = []
    student_values_all: list[float] = []
    sample_means: list[float] = []
    overlap_sum = 0
    overlap_full_count = 0
    overlap_token_count = 0
    overlap_ratios: list[float] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch = _collate(batch_rows, pad_token_id=int(tokenizer.pad_token_id))
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_input_ids = batch["input_ids"][..., 1:].contiguous()
        shift_mask = batch["response_mask"][..., 1:].contiguous()
        shift_teacher = batch["teacher_token_logprobs"][..., 1:].contiguous()
        shift_teacher_topk_ids = batch["teacher_topk_token_ids"][..., 1:, :].contiguous()
        shift_teacher_topk_mask = batch["teacher_topk_mask"][..., 1:, :].contiguous()
        student_logprobs = _gather_token_logprobs(
            shift_logits,
            shift_input_ids,
            chunk_size=args.logprob_chunk_size,
        )
        batch_overlap_sum, batch_full_count, batch_overlap_tokens, batch_overlap_ratios = _topk_overlap_counts(
            shift_logits,
            shift_teacher_topk_ids,
            shift_teacher_topk_mask,
            shift_mask,
            top_k=args.overlap_top_k,
            chunk_size=args.logprob_chunk_size,
        )
        overlap_sum += batch_overlap_sum
        overlap_full_count += batch_full_count
        overlap_token_count += batch_overlap_tokens
        overlap_ratios.extend(batch_overlap_ratios)
        delta = shift_teacher - student_logprobs
        valid_delta = delta[shift_mask].detach().cpu().tolist()
        valid_teacher = shift_teacher[shift_mask].detach().cpu().tolist()
        valid_student = student_logprobs[shift_mask].detach().cpu().tolist()
        deltas.extend(float(value) for value in valid_delta)
        teacher_values_all.extend(float(value) for value in valid_teacher)
        student_values_all.extend(float(value) for value in valid_student)
        total += len(valid_delta)
        positive += sum(1 for value in valid_delta if value > 0)
        negative += sum(1 for value in valid_delta if value < 0)
        zero += sum(1 for value in valid_delta if value == 0)
        for row_index in range(len(batch_rows)):
            row_delta = delta[row_index][shift_mask[row_index]].detach().cpu().tolist()
            if row_delta:
                sample_means.append(float(sum(row_delta) / len(row_delta)))
        if (start // args.batch_size + 1) % 10 == 0:
            print(f"processed={min(start + args.batch_size, len(rows))}/{len(rows)} tokens={total}")

    report = {
        "dataset": args.dataset,
        "student_model": args.student_model,
        "adapter_base_model": args.adapter_base_model,
        "num_samples": len(rows),
        "num_tokens": total,
        "teacher_gt_student_count": positive,
        "teacher_lt_student_count": negative,
        "teacher_eq_student_count": zero,
        "positive_ratio": positive / total if total else 0.0,
        "negative_ratio": negative / total if total else 0.0,
        "zero_ratio": zero / total if total else 0.0,
        "delta_mean": float(statistics.fmean(deltas)) if deltas else 0.0,
        "delta_median": float(statistics.median(deltas)) if deltas else 0.0,
        "delta_p10": _percentile(deltas, 0.10),
        "delta_p90": _percentile(deltas, 0.90),
        "teacher_logprob_mean": float(statistics.fmean(teacher_values_all)) if teacher_values_all else 0.0,
        "student_logprob_mean": float(statistics.fmean(student_values_all)) if student_values_all else 0.0,
        "per_sample_delta_mean_p10": _percentile(sample_means, 0.10),
        "per_sample_delta_mean_p50": _percentile(sample_means, 0.50),
        "per_sample_delta_mean_p90": _percentile(sample_means, 0.90),
        "overlap_top_k": args.overlap_top_k,
        "overlap_token_count": overlap_token_count,
        "overlap_intersection_mean": overlap_sum / overlap_token_count if overlap_token_count else 0.0,
        "overlap_ratio_mean": float(statistics.fmean(overlap_ratios)) if overlap_ratios else 0.0,
        "overlap_ratio_p10": _percentile(overlap_ratios, 0.10),
        "overlap_ratio_p50": _percentile(overlap_ratios, 0.50),
        "overlap_ratio_p90": _percentile(overlap_ratios, 0.90),
        "overlap_full_ratio": overlap_full_count / overlap_token_count if overlap_token_count else 0.0,
    }
    output = ensure_parent(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
