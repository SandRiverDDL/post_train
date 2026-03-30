from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from post_train.answers import are_equivalent
from post_train.config import Stage1RSRCandidatesConfig, Stage1RSRSelectConfig
from post_train.data import load_dataset_rows, make_sft_record, summarize_sft_dataset
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.prompts import build_sft_prompt
from post_train.stage2_data import validate_sft_record

SOURCE_ORDER = ("large", "small", "short", "long")
SOURCE_DATASET_FIELDS = {
    "short": ("short_dataset", "short_split"),
    "long": ("long_dataset", "long_split"),
    "small": ("small_teacher_dataset", "small_teacher_split"),
    "large": ("large_teacher_dataset", "large_teacher_split"),
}
QUESTION_KEYS = ("problem", "question", "query", "prompt")
RSR_DIAGNOSTIC_EVERY = 10


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def normalize_problem_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _first_present(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _stable_problem_id(problem: str) -> str:
    digest = hashlib.sha1(problem.encode("utf-8")).hexdigest()
    return f"rsr-{digest[:16]}"


def _length_stats(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0}
    ordered = sorted(lengths)

    def percentile(ratio: float) -> int:
        index = int(round((len(ordered) - 1) * ratio))
        return int(ordered[index])

    return {
        "min": int(ordered[0]),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "max": int(ordered[-1]),
    }


def _build_tokenized_example(
    tokenizer: Any,
    *,
    question: str,
    solution: str,
    max_length: int,
) -> dict[str, Any]:
    prompt = build_sft_prompt(question)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(solution, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))
    completion_positions = list(range(prompt_len, len(input_ids)))
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "prompt_tokens": len(prompt_ids),
        "solution_tokens": len(completion_ids),
        "scored_solution_tokens": len(completion_positions),
        "truncated": len(prompt_ids) + len(completion_ids) > max_length,
    }


def _group_rows_by_length(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (-len(row["input_ids"]), str(row.get("id", "")), str(row.get("meta", {}).get("source_name", ""))),
    )


def _compute_rsr_from_logits(
    logits: torch.Tensor,
    input_ids: list[int],
    completion_positions: list[int],
    *,
    rank_clip_r: int,
) -> dict[str, float | int]:
    if not completion_positions:
        return {
            "resp_token_length": 0,
            "avg_rank_clip": 0.0,
            "avg_surprisal": 0.0,
            "rank_surprisal_ratio": 0.0,
        }

    logit_positions = [position - 1 for position in completion_positions if position > 0]
    target_ids = [input_ids[position] for position in completion_positions if position > 0]
    valid_logits = logits[logit_positions]
    targets = torch.tensor(target_ids, dtype=torch.long, device=valid_logits.device)

    logits_fp32 = valid_logits.float()
    log_z = torch.logsumexp(logits_fp32, dim=-1)
    target_logits = logits_fp32.gather(1, targets[:, None]).squeeze(1)
    nll = log_z - target_logits

    top_values = torch.topk(valid_logits, k=min(rank_clip_r, valid_logits.shape[-1]), dim=-1).values
    rank = 1 + (top_values.float() > target_logits[:, None]).sum(dim=-1)
    rank = torch.clamp(rank, max=rank_clip_r)

    sum_nll = float(nll.sum().item())
    token_length = len(target_ids)
    avg_surprisal = sum_nll / token_length
    avg_rank = float(rank.float().mean().item())
    ratio = float(rank.sum().item()) / max(sum_nll, 1.0e-12)
    return {
        "resp_token_length": token_length,
        "avg_rank_clip": avg_rank,
        "avg_surprisal": avg_surprisal,
        "rank_surprisal_ratio": ratio,
    }


def _format_cuda_gb(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 3):.2f}G"


def _build_rsr_diagnostic_line(
    *,
    batch_index: int,
    total_batches: int,
    batch_lengths: list[int],
    allocated_bytes: int | None,
    reserved_bytes: int | None,
    peak_allocated_bytes: int | None,
) -> str:
    if not batch_lengths:
        lengths_preview = "-"
        min_len = 0
        max_len = 0
        avg_len = 0.0
    else:
        preview = batch_lengths[:6]
        preview_suffix = "..." if len(batch_lengths) > 6 else ""
        lengths_preview = ",".join(str(length) for length in preview) + preview_suffix
        min_len = min(batch_lengths)
        max_len = max(batch_lengths)
        avg_len = sum(batch_lengths) / len(batch_lengths)
    return (
        f"RSR batch {batch_index}/{total_batches} | "
        f"size={len(batch_lengths)} | "
        f"len[min/avg/max]={min_len}/{avg_len:.1f}/{max_len} | "
        f"lens=[{lengths_preview}] | "
        f"alloc={_format_cuda_gb(allocated_bytes)} | "
        f"reserved={_format_cuda_gb(reserved_bytes)} | "
        f"peak={_format_cuda_gb(peak_allocated_bytes)}"
    )


def _emit_rsr_diagnostic(
    *,
    batch_index: int,
    total_batches: int,
    batch_lengths: list[int],
    device: torch.device,
) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        allocated_bytes = torch.cuda.memory_allocated(device_index)
        reserved_bytes = torch.cuda.memory_reserved(device_index)
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device_index)
    else:
        allocated_bytes = None
        reserved_bytes = None
        peak_allocated_bytes = None
    line = _build_rsr_diagnostic_line(
        batch_index=batch_index,
        total_batches=total_batches,
        batch_lengths=batch_lengths,
        allocated_bytes=allocated_bytes,
        reserved_bytes=reserved_bytes,
        peak_allocated_bytes=peak_allocated_bytes,
    )
    sys.stdout.write("\r" + line)
    sys.stdout.flush()


@torch.inference_mode()
def score_rsr_trajectories(
    trajectory_rows: list[dict[str, Any]],
    cfg: Stage1RSRCandidatesConfig,
    *,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    if not trajectory_rows:
        return []

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    model, loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.student_model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )
    tokenizer = tokenizer or loaded_tokenizer
    model.eval()

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized_rows: list[dict[str, Any]] = []
    for row in trajectory_rows:
        if {"input_ids", "attention_mask", "prompt_tokens", "solution_tokens", "scored_solution_tokens", "truncated"} <= set(row.keys()):
            encoded = {
                "input_ids": list(row["input_ids"]),
                "attention_mask": list(row["attention_mask"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "solution_tokens": int(row["solution_tokens"]),
                "scored_solution_tokens": int(row["scored_solution_tokens"]),
                "truncated": bool(row["truncated"]),
            }
        else:
            encoded = _build_tokenized_example(
                tokenizer,
                question=str(row["question"]),
                solution=str(row["solution"]),
                max_length=cfg.max_seq_length,
            )
        copied = dict(row)
        copied["prompt_tokens"] = encoded["prompt_tokens"]
        copied["solution_tokens"] = encoded["solution_tokens"]
        copied["scored_solution_tokens"] = encoded["scored_solution_tokens"]
        copied["truncated"] = encoded["truncated"]
        copied["input_ids"] = encoded["input_ids"]
        copied["attention_mask"] = encoded["attention_mask"]
        copied["completion_positions"] = list(range(encoded["prompt_tokens"], len(encoded["input_ids"])))
        tokenized_rows.append(copied)

    tokenized_rows = _group_rows_by_length(tokenized_rows)

    scored_rows: list[dict[str, Any]] = []
    total_batches = (len(tokenized_rows) + cfg.batch_size - 1) // cfg.batch_size
    for start in tqdm(
        range(0, len(tokenized_rows), cfg.batch_size),
        desc="RSR scoring",
    ):
        batch = tokenized_rows[start : start + cfg.batch_size]
        batch_number = start // cfg.batch_size + 1
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids_batch: list[list[int]] = []
        attention_mask_batch: list[list[int]] = []
        for row in batch:
            pad_length = max_len - len(row["input_ids"])
            input_ids_batch.append(row["input_ids"] + [pad_token_id] * pad_length)
            attention_mask_batch.append(row["attention_mask"] + [0] * pad_length)

        model_inputs = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long, device=model.device),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long, device=model.device),
            "use_cache": False,
        }
        outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if batch_number % RSR_DIAGNOSTIC_EVERY == 0 or batch_number == total_batches:
            _emit_rsr_diagnostic(
                batch_index=batch_number,
                total_batches=total_batches,
                batch_lengths=[len(row["input_ids"]) for row in batch],
                device=model.device,
            )

        for batch_index, row in enumerate(batch):
            seq_len = len(row["input_ids"])
            metrics = _compute_rsr_from_logits(
                logits[batch_index, :seq_len, :],
                row["input_ids"],
                row["completion_positions"],
                rank_clip_r=cfg.rank_clip_r,
            )
            scored_row = dict(row)
            scored_row.update(metrics)
            del scored_row["input_ids"]
            del scored_row["attention_mask"]
            del scored_row["completion_positions"]
            scored_rows.append(scored_row)
    if total_batches > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return scored_rows


def _source_rows_from_config(cfg: Stage1RSRCandidatesConfig) -> dict[str, list[dict[str, Any]]]:
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for source_name, (dataset_field, split_field) in SOURCE_DATASET_FIELDS.items():
        dataset_name = getattr(cfg, dataset_field)
        split_name = getattr(cfg, split_field)
        source_rows[source_name] = load_dataset_rows(
            dataset_name,
            split=split_name,
            cache_dir=cfg.cache_dir,
        )
    return source_rows


def _build_problem_index(
    source_rows: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, int]], dict[str, str]]:
    source_index: dict[str, dict[str, dict[str, Any]]] = {}
    source_stats: dict[str, dict[str, int]] = {}
    problem_display: dict[str, str] = {}

    for source_name, rows in source_rows.items():
        normalized_map: dict[str, dict[str, Any]] = {}
        duplicate_count = 0
        for row in rows:
            question = _first_present(row, QUESTION_KEYS)
            normalized_problem = normalize_problem_text(question)
            if not normalized_problem:
                continue
            problem_display.setdefault(normalized_problem, question)
            if normalized_problem in normalized_map:
                duplicate_count += 1
                continue
            normalized_map[normalized_problem] = row
        source_index[source_name] = normalized_map
        source_stats[source_name] = {
            "raw_rows": len(rows),
            "unique_problem_rows": len(normalized_map),
            "duplicate_problem_rows": duplicate_count,
        }
    return source_index, source_stats, problem_display


def prepare_stage1_rsr_candidates(
    cfg: Stage1RSRCandidatesConfig,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        cfg.tokenizer_name or cfg.student_model_name,
        trust_remote_code=True,
        local_files_only=False,
    )

    source_rows = _source_rows_from_config(cfg)
    source_index, source_stats, problem_display = _build_problem_index(source_rows)
    all_problems = set()
    for rows in source_index.values():
        all_problems.update(rows.keys())

    unmatched_preview_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    invalid_counts = {source_name: Counter() for source_name in SOURCE_ORDER}
    coverage_counter: Counter[int] = Counter()
    combo_counter: Counter[tuple[str, ...]] = Counter()
    truncated_total = 0
    dropped_truncated = 0

    for normalized_problem in sorted(all_problems):
        available_sources = tuple(
            source_name for source_name in SOURCE_ORDER if normalized_problem in source_index.get(source_name, {})
        )
        coverage_counter[len(available_sources)] += 1
        combo_counter[available_sources] += 1

        for source_name in available_sources:
            dataset_field, _ = SOURCE_DATASET_FIELDS[source_name]
            dataset_name = getattr(cfg, dataset_field)
            raw_row = source_index[source_name][normalized_problem]
            record = make_sft_record(raw_row, 0, source=dataset_name)
            record.setdefault("meta", {})
            record["meta"]["source_dataset"] = dataset_name
            record["meta"]["source_name"] = source_name
            record["meta"]["normalized_problem"] = normalized_problem
            ok, reason = validate_sft_record(record)
            if not ok:
                invalid_counts[source_name][reason] += 1
                continue
            encoded = _build_tokenized_example(
                tokenizer,
                question=str(record["question"]),
                solution=str(record["solution"]),
                max_length=cfg.max_seq_length,
            )
            if encoded["truncated"]:
                truncated_total += 1
                if cfg.drop_truncated:
                    dropped_truncated += 1
                    continue
            record["meta"]["solution_tokens"] = encoded["solution_tokens"]
            record["prompt_tokens"] = encoded["prompt_tokens"]
            record["solution_tokens"] = encoded["solution_tokens"]
            record["scored_solution_tokens"] = encoded["scored_solution_tokens"]
            record["truncated"] = encoded["truncated"]
            record["input_ids"] = encoded["input_ids"]
            record["attention_mask"] = encoded["attention_mask"]
            trajectory_rows.append(record)

        if len(unmatched_preview_rows) < cfg.unmatched_preview_count and len(available_sources) < len(SOURCE_ORDER):
            missing_sources = [source_name for source_name in SOURCE_ORDER if source_name not in available_sources]
            unmatched_preview_rows.append(
                {
                    "problem": problem_display.get(normalized_problem, normalized_problem),
                    "normalized_problem": normalized_problem,
                    "available_sources": list(available_sources),
                    "missing_sources": missing_sources,
                }
            )

    scored_rows = score_rsr_trajectories(trajectory_rows, cfg, tokenizer=tokenizer)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        normalized_problem = str(row.get("meta", {}).get("normalized_problem", ""))
        grouped[normalized_problem].append(row)

    candidate_rows: list[dict[str, Any]] = []
    filtered_inconsistent_answers = 0
    for normalized_problem, rows in grouped.items():
        if not rows:
            continue
        answer_groups = [str(row["final_answer"]).strip() for row in rows if str(row["final_answer"]).strip()]
        if answer_groups and any(
            not are_equivalent(answer_groups[0], candidate_answer) for candidate_answer in answer_groups[1:]
        ):
            filtered_inconsistent_answers += 1
            continue

        sorted_rows = sorted(
            rows,
            key=lambda row: (
                float(row["rank_surprisal_ratio"]),
                int(row["meta"].get("solution_tokens", 0)),
                SOURCE_ORDER.index(str(row["meta"].get("source_name"))),
            ),
        )
        trajectories: list[dict[str, Any]] = []
        for row in sorted_rows:
            meta = dict(row.get("meta", {}))
            trajectories.append(
                {
                    "source_name": str(meta.get("source_name", "")),
                    "source_dataset": str(meta.get("source_dataset", "")),
                    "solution": str(row["solution"]),
                    "final_answer": str(row["final_answer"]),
                    "solution_tokens": int(meta.get("solution_tokens", 0)),
                    "prompt_tokens": int(row.get("prompt_tokens", 0)),
                    "scored_solution_tokens": int(row.get("scored_solution_tokens", 0)),
                    "truncated": bool(row.get("truncated", False)),
                    "avg_rank_clip": float(row["avg_rank_clip"]),
                    "avg_surprisal": float(row["avg_surprisal"]),
                    "rank_surprisal_ratio": float(row["rank_surprisal_ratio"]),
                }
            )
        best = trajectories[0]
        candidate_rows.append(
            {
                "id": _stable_problem_id(normalized_problem),
                "question": problem_display.get(normalized_problem, normalized_problem),
                "final_answer": str(best["final_answer"]),
                "normalized_problem": normalized_problem,
                "available_sources": [trajectory["source_name"] for trajectory in trajectories],
                "trajectory_count": len(trajectories),
                "best_source_by_rsr": str(best["source_name"]),
                "best_rsr": float(best["rank_surprisal_ratio"]),
                "trajectories": trajectories,
                "meta": {
                    "source": "rsr_candidates",
                },
            }
        )

    candidate_rows.sort(
        key=lambda row: (
            -int(row["trajectory_count"]),
            float(row["best_rsr"]),
            str(row["id"]),
        )
    )

    source_metric_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        for trajectory in row["trajectories"]:
            source_metric_rows[str(trajectory["source_name"])].append(trajectory)

    report = {
        "config": {
            "student_model_name": cfg.student_model_name,
            "tokenizer_name": cfg.tokenizer_name or cfg.student_model_name,
            "batch_size": cfg.batch_size,
            "max_seq_length": cfg.max_seq_length,
            "rank_clip_r": cfg.rank_clip_r,
            "drop_truncated": cfg.drop_truncated,
            "load_in_4bit": cfg.load_in_4bit,
        },
        "filters": {
            "truncated_before_scoring": truncated_total,
            "dropped_truncated": dropped_truncated,
            "dropped_truncated_ratio": (
                dropped_truncated / truncated_total if truncated_total else 0.0
            ),
        },
        "sources": {
            source_name: {
                **source_stats[source_name],
                "invalid_rows": int(sum(invalid_counts[source_name].values())),
                "invalid_reasons": dict(sorted(invalid_counts[source_name].items())),
                "scored_rows": len(source_metric_rows.get(source_name, [])),
                "solution_tokens": _length_stats(
                    [int(row["solution_tokens"]) for row in source_metric_rows.get(source_name, [])]
                ),
                "rsr": {
                    "min": min((float(row["rank_surprisal_ratio"]) for row in source_metric_rows.get(source_name, [])), default=0.0),
                    "p50": _length_stats(
                        [int(round(float(row["rank_surprisal_ratio"]) * 10000)) for row in source_metric_rows.get(source_name, [])]
                    )["p50"]
                    / 10000.0,
                    "p90": _length_stats(
                        [int(round(float(row["rank_surprisal_ratio"]) * 10000)) for row in source_metric_rows.get(source_name, [])]
                    )["p90"]
                    / 10000.0,
                    "max": max((float(row["rank_surprisal_ratio"]) for row in source_metric_rows.get(source_name, [])), default=0.0),
                },
            }
            for source_name in SOURCE_ORDER
        },
        "coverage": {
            "trajectory_count_distribution": dict(sorted(coverage_counter.items())),
            "source_combo_distribution": {
                "+".join(combo): count for combo, count in sorted(combo_counter.items(), key=lambda item: (-len(item[0]), item[0]))
            },
            "candidate_rows": len(candidate_rows),
            "filtered_inconsistent_answers": filtered_inconsistent_answers,
        },
        "dataset_summary": {
            "candidate_rows": len(candidate_rows),
            "best_source_counts": dict(
                sorted(Counter(str(row["best_source_by_rsr"]) for row in candidate_rows).items())
            ),
            "trajectory_count_distribution": dict(
                sorted(Counter(int(row["trajectory_count"]) for row in candidate_rows).items())
            ),
        },
    }

    output_path = write_jsonl(cfg.output_path, candidate_rows)
    report_path = _write_json(cfg.report_path, report)
    unmatched_preview_path = write_jsonl(cfg.unmatched_preview_path, unmatched_preview_rows)
    return {
        "output_path": output_path,
        "report_path": report_path,
        "unmatched_preview_path": unmatched_preview_path,
        "report": report,
    }


def _select_best_trajectory(
    row: dict[str, Any],
    *,
    min_solution_tokens: int,
    max_solution_tokens: int | None,
    allowed_sources: set[str] | None,
    drop_truncated_trajectories: bool,
    prefer_lower_rsr: bool,
) -> dict[str, Any] | None:
    trajectories = list(row.get("trajectories", []))
    eligible: list[dict[str, Any]] = []
    for trajectory in trajectories:
        source_name = str(trajectory.get("source_name", ""))
        if allowed_sources is not None and source_name not in allowed_sources:
            continue
        if drop_truncated_trajectories and bool(trajectory.get("truncated", False)):
            continue
        solution_tokens = int(trajectory.get("solution_tokens", 0))
        if solution_tokens < min_solution_tokens:
            continue
        if max_solution_tokens is not None and solution_tokens >= max_solution_tokens:
            continue
        eligible.append(trajectory)
    if not eligible:
        return None
    sorted_eligible = sorted(
        eligible,
        key=lambda trajectory: (
            float(trajectory["rank_surprisal_ratio"]) if prefer_lower_rsr else -float(trajectory["rank_surprisal_ratio"]),
            int(trajectory["solution_tokens"]),
            SOURCE_ORDER.index(str(trajectory["source_name"])),
        ),
    )
    return sorted_eligible[0]


def select_stage1_rsr_dataset(cfg: Stage1RSRSelectConfig) -> dict[str, Any]:
    candidate_rows = read_jsonl(cfg.input_path)
    allowed_sources = set(cfg.allowed_sources) if cfg.allowed_sources else None

    bucketed_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    filtered_length_only = 0
    dropped_truncated_trajectory_count = 0
    dropped_truncated_by_source: Counter[str] = Counter()
    for row in candidate_rows:
        if cfg.drop_truncated_trajectories:
            for trajectory in row.get("trajectories", []):
                if bool(trajectory.get("truncated", False)):
                    dropped_truncated_trajectory_count += 1
                    dropped_truncated_by_source[str(trajectory.get("source_name", ""))] += 1
        selected = _select_best_trajectory(
            row,
            min_solution_tokens=cfg.min_solution_tokens,
            max_solution_tokens=cfg.max_solution_tokens,
            allowed_sources=allowed_sources,
            drop_truncated_trajectories=cfg.drop_truncated_trajectories,
            prefer_lower_rsr=cfg.prefer_lower_rsr,
        )
        if selected is None:
            filtered_length_only += 1
            continue
        copied = dict(row)
        copied["selected_trajectory"] = selected
        copied["eligible_trajectory_count"] = sum(
            1
            for trajectory in row.get("trajectories", [])
            if allowed_sources is None or str(trajectory.get("source_name", "")) in allowed_sources
            if (not cfg.drop_truncated_trajectories or not bool(trajectory.get("truncated", False)))
            if int(trajectory.get("solution_tokens", 0)) >= cfg.min_solution_tokens
            and (cfg.max_solution_tokens is None or int(trajectory.get("solution_tokens", 0)) < cfg.max_solution_tokens)
        )
        bucketed_rows[int(copied["eligible_trajectory_count"])].append(copied)

    ordered_rows: list[dict[str, Any]] = []
    for trajectory_count in sorted(bucketed_rows.keys(), reverse=True):
        sorted_bucket = sorted(
            bucketed_rows[trajectory_count],
            key=lambda row: (
                -int(row["eligible_trajectory_count"]),
                float(row["selected_trajectory"]["rank_surprisal_ratio"])
                if cfg.prefer_lower_rsr
                else -float(row["selected_trajectory"]["rank_surprisal_ratio"]),
                int(row["selected_trajectory"]["solution_tokens"]),
                SOURCE_ORDER.index(str(row["selected_trajectory"]["source_name"])),
                str(row["id"]),
            ),
        )
        ordered_rows.extend(sorted_bucket)

    if cfg.sample_size > len(ordered_rows):
        raise ValueError(f"样本不足：需要 {cfg.sample_size} 条，筛选后只有 {len(ordered_rows)} 条。")

    selected_rows = ordered_rows[: cfg.sample_size]
    final_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        chosen = row["selected_trajectory"]
        final_rows.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "solution": str(chosen["solution"]),
                "final_answer": str(chosen["final_answer"]),
                "meta": {
                    "source": str(chosen["source_dataset"]),
                    "source_name": str(chosen["source_name"]),
                    "rsr": float(chosen["rank_surprisal_ratio"]),
                    "trajectory_count": int(row["trajectory_count"]),
                    "eligible_trajectory_count": int(row["eligible_trajectory_count"]),
                    "selected_from_rsr_candidates": True,
                },
            }
        )

    report = {
        "config": {
            "input_path": str(cfg.input_path),
            "sample_size": cfg.sample_size,
            "min_solution_tokens": cfg.min_solution_tokens,
            "max_solution_tokens": cfg.max_solution_tokens,
            "allowed_sources": sorted(allowed_sources) if allowed_sources is not None else None,
            "drop_truncated_trajectories": cfg.drop_truncated_trajectories,
            "prefer_lower_rsr": cfg.prefer_lower_rsr,
            "seed": cfg.seed,
        },
        "filters": {
            "filtered_no_eligible_trajectory": filtered_length_only,
            "dropped_truncated_trajectory_count": dropped_truncated_trajectory_count,
            "dropped_truncated_by_source": dict(sorted(dropped_truncated_by_source.items())),
        },
        "selection": {
            "available_after_filter": len(ordered_rows),
            "selected_rows": len(final_rows),
            "trajectory_count_counts": dict(
                sorted(Counter(int(row["meta"]["trajectory_count"]) for row in final_rows).items())
            ),
            "eligible_trajectory_count_counts": dict(
                sorted(Counter(int(row["meta"]["eligible_trajectory_count"]) for row in final_rows).items())
            ),
            "chosen_source_counts": dict(
                sorted(Counter(str(row["meta"]["source_name"]) for row in final_rows).items())
            ),
            "chosen_source_ratio": {
                source_name: count / len(final_rows)
                for source_name, count in sorted(Counter(str(row["meta"]["source_name"]) for row in final_rows).items())
            },
            "selected_rsr_stats": {
                "min": min((float(row["meta"]["rsr"]) for row in final_rows), default=0.0),
                "max": max((float(row["meta"]["rsr"]) for row in final_rows), default=0.0),
                "avg": (
                    sum(float(row["meta"]["rsr"]) for row in final_rows) / len(final_rows)
                    if final_rows
                    else 0.0
                ),
            },
        },
        "dataset_summary": summarize_sft_dataset(final_rows),
    }
    output_path = write_jsonl(cfg.output_path, final_rows)
    report_path = _write_json(cfg.report_path, report)
    return {
        "output_path": output_path,
        "report_path": report_path,
        "report": report,
    }
