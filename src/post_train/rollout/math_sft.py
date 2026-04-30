from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

from post_train.answers import evaluate_prediction
from post_train.data import load_dataset_rows, make_sft_record, sample_rows
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.prompts import build_eval_prompt
from post_train.schemas import SFTRecord

DEFAULT_MATH_DATASET = "EleutherAI/hendrycks_math"
DEFAULT_HENDRYCKS_MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
DEFAULT_MIX_LONG_PATH = Path("data/stage1/mix_long/train.jsonl")


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def parse_level(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
    return None


def _output_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def build_math_query_pool(
    *,
    dataset_name: str = DEFAULT_MATH_DATASET,
    split: str = "train",
    sample_size: int = 1500,
    seed: int = 42,
    levels: list[int] | None = None,
    cache_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    config_names: list[str] = []
    if dataset_name == "EleutherAI/hendrycks_math":
        config_names = list(DEFAULT_HENDRYCKS_MATH_CONFIGS)
        for config_name in config_names:
            for row in load_dataset_rows(dataset_name, split=split, config_name=config_name, cache_dir=cache_dir):
                copied = dict(row)
                copied["__config_name__"] = config_name
                raw_rows.append(copied)
    else:
        raw_rows = load_dataset_rows(dataset_name, split=split, cache_dir=cache_dir)
    allowed_levels = set(levels or [])
    query_rows: list[dict[str, Any]] = []
    level_distribution: dict[int, int] = {}
    filtered_missing_level = 0
    filtered_level = 0

    for index, row in enumerate(raw_rows):
        level = parse_level(row.get("level"))
        if level is None:
            filtered_missing_level += 1
            continue
        level_distribution[level] = level_distribution.get(level, 0) + 1
        if allowed_levels and level not in allowed_levels:
            filtered_level += 1
            continue

        record = make_sft_record(row, index, source=dataset_name)
        meta = dict(record.get("meta", {}))
        meta.update(
            {
                "source_dataset": dataset_name,
                "source_split": split,
                "source_index": index,
                "source_config": str(row.get("__config_name__", "")),
                "level": level,
                "type": str(row.get("type", "")),
            }
        )
        query_rows.append(
            {
                "id": str(row.get("unique_id") or row.get("id") or record["id"]),
                "question": record["question"],
                "final_answer": record["final_answer"],
                "meta": meta,
            }
        )

    if sample_size > len(query_rows):
        raise ValueError(f"MATH 可用样本不足：需要 {sample_size} 条，过滤后只有 {len(query_rows)} 条。")

    sampled_rows = sample_rows(query_rows, sample_size=sample_size, seed=seed)
    selected_level_distribution: dict[int, int] = {}
    for row in sampled_rows:
        level = int(row["meta"]["level"])
        selected_level_distribution[level] = selected_level_distribution.get(level, 0) + 1

    return sampled_rows, {
        "dataset_name": dataset_name,
        "split": split,
        "config_names": config_names or None,
        "raw_rows": len(raw_rows),
        "eligible_rows": len(query_rows),
        "sampled_queries": len(sampled_rows),
        "levels": sorted(allowed_levels) if allowed_levels else None,
        "filtered_missing_level": filtered_missing_level,
        "filtered_level": filtered_level,
        "level_distribution": dict(sorted(level_distribution.items())),
        "selected_level_distribution": dict(sorted(selected_level_distribution.items())),
        "seed": seed,
    }


def sample_math_responses(
    query_rows: list[dict[str, Any]],
    *,
    model: str,
    responses_per_prompt: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import LLM, SamplingParams

    prompts = [build_eval_prompt(str(row["question"])) for row in query_rows]
    llm = LLM(
        model=model,
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=responses_per_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=["</s>", "<|im_end|>"],
    )

    started_at = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    total_seconds = time.perf_counter() - started_at

    raw_samples: list[dict[str, Any]] = []
    response_count = 0
    correct_count = 0
    boxed_count = 0
    parse_ok_count = 0
    output_token_counts: list[int] = []

    for row, prompt, output in zip(query_rows, prompts, outputs, strict=True):
        responses: list[dict[str, Any]] = []
        for rollout_index, candidate in enumerate(output.outputs):
            text = candidate.text
            verdict = evaluate_prediction(text, str(row.get("final_answer", "")), require_boxed=True)
            output_tokens = _output_tokens(text)
            response_count += 1
            correct_count += int(bool(verdict["correct"]))
            boxed_count += int(bool(verdict["boxed"]))
            parse_ok_count += int(bool(verdict["parse_ok"]))
            output_token_counts.append(output_tokens)
            responses.append(
                {
                    "rollout_index": rollout_index,
                    "text": text,
                    "predicted_answer": str(verdict.get("parsed_answer", "")),
                    "boxed": bool(verdict["boxed"]),
                    "parse_ok": bool(verdict["parse_ok"]),
                    "correct": bool(verdict["correct"]),
                    "output_tokens": output_tokens,
                }
            )
        raw_samples.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "final_answer": str(row.get("final_answer", "")),
                "prompt": prompt,
                "responses": responses,
                "meta": dict(row.get("meta", {})),
            }
        )

    avg_output_tokens = sum(output_token_counts) / len(output_token_counts) if output_token_counts else 0.0
    return raw_samples, {
        "model": model,
        "prompt_count": len(query_rows),
        "responses_per_prompt": responses_per_prompt,
        "response_count": response_count,
        "total_seconds": total_seconds,
        "responses_per_second": response_count / total_seconds if total_seconds > 0 else 0.0,
        "boxed_count": boxed_count,
        "parse_ok_count": parse_ok_count,
        "correct_count": correct_count,
        "boxed_rate": boxed_count / response_count if response_count else 0.0,
        "parse_ok_rate": parse_ok_count / response_count if response_count else 0.0,
        "correct_rate": correct_count / response_count if response_count else 0.0,
        "avg_output_tokens": avg_output_tokens,
    }


def build_rejection_sampled_sft_rows(
    raw_samples: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
    generation_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sample in raw_samples:
        for response in sample.get("responses", []):
            if not (bool(response.get("boxed")) and bool(response.get("parse_ok")) and bool(response.get("correct"))):
                continue
            meta = dict(sample.get("meta", {}))
            meta.update(
                {
                    "source": "math_rollout_rejection",
                    "source_question_id": str(sample.get("id", "")),
                    "generation_model": generation_model,
                    "rollout_index": int(response.get("rollout_index", 0)),
                    "output_tokens": int(response.get("output_tokens", 0)),
                    "selector_name": "boxed_parse_correct",
                }
            )
            candidates.append(
                SFTRecord(
                    id=f"{sample.get('id', '')}::rollout_{response.get('rollout_index', 0)}",
                    question=str(sample.get("question", "")),
                    solution=str(response.get("text", "")),
                    final_answer=str(sample.get("final_answer", "")),
                    meta=meta,
                ).model_dump()
            )

    if target_count > len(candidates):
        raise ValueError(f"拒绝采样样本不足：目标 {target_count} 条，合格轨迹只有 {len(candidates)} 条。")

    retained_rows = sample_rows(candidates, sample_size=target_count, seed=seed)
    return retained_rows, {
        "selector_name": "boxed_parse_correct",
        "candidate_count": len(candidates),
        "target_count": target_count,
        "retained_count": len(retained_rows),
        "retention_rate_from_candidates": len(retained_rows) / len(candidates) if candidates else 0.0,
        "seed": seed,
    }


def sample_mix_long_rows(
    *,
    path: str | Path,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(path)
    sampled = sample_rows(rows, sample_size=sample_size, seed=seed)
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(sampled):
        copied = dict(row)
        meta = dict(copied.get("meta", {}))
        meta.setdefault("source", "mix_long")
        meta["mix_long_sample_index"] = index
        copied["meta"] = meta
        normalized_rows.append(SFTRecord.model_validate(copied).model_dump())
    return normalized_rows, {
        "path": str(path),
        "raw_rows": len(rows),
        "sample_size": sample_size,
        "seed": seed,
    }


def prepare_math_rollout_sft_dataset(
    *,
    model: str,
    output_dir: str | Path,
    sample_size: int = 1500,
    responses_per_prompt: int = 8,
    retained_count: int = 1000,
    mix_long_sample_size: int = 1000,
    mix_long_path: str | Path = DEFAULT_MIX_LONG_PATH,
    dataset_name: str = DEFAULT_MATH_DATASET,
    split: str = "train",
    levels: list[int] | None = None,
    seed: int = 42,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.8,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    base = Path(output_dir)
    query_path = base / "query_pool.jsonl"
    raw_samples_path = base / "raw_samples.jsonl"
    retained_path = base / "train.rollout_rejection.jsonl"
    mix_long_path_out = base / "train.mix_long_random.jsonl"
    combined_path = base / "train.combined.jsonl"
    report_path = base / "report.json"

    query_rows, query_report = build_math_query_pool(
        dataset_name=dataset_name,
        split=split,
        sample_size=sample_size,
        seed=seed,
        levels=levels,
        cache_dir=cache_dir,
    )
    write_jsonl(query_path, query_rows)

    raw_samples, sampling_report = sample_math_responses(
        query_rows,
        model=model,
        responses_per_prompt=responses_per_prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    write_jsonl(raw_samples_path, raw_samples)

    retained_rows, rejection_report = build_rejection_sampled_sft_rows(
        raw_samples,
        target_count=retained_count,
        seed=seed + 1,
        generation_model=model,
    )
    write_jsonl(retained_path, retained_rows)

    mix_long_rows, mix_long_report = sample_mix_long_rows(
        path=mix_long_path,
        sample_size=mix_long_sample_size,
        seed=seed + 2,
    )
    write_jsonl(mix_long_path_out, mix_long_rows)

    combined_rows = list(retained_rows) + list(mix_long_rows)
    combined_rows = sample_rows(combined_rows, sample_size=len(combined_rows), seed=seed + 3)
    write_jsonl(combined_path, combined_rows)

    report = {
        "model": model,
        "query_pool": query_report,
        "sampling": sampling_report,
        "rejection_sampling": rejection_report,
        "mix_long_sampling": mix_long_report,
        "outputs": {
            "query_pool": str(query_path),
            "raw_samples": str(raw_samples_path),
            "rollout_rejection_train": str(retained_path),
            "mix_long_random_train": str(mix_long_path_out),
            "combined_train": str(combined_path),
        },
        "combined": {
            "rollout_rejection_count": len(retained_rows),
            "mix_long_count": len(mix_long_rows),
            "total": len(combined_rows),
        },
    }
    _write_json(report_path, report)
    report["outputs"]["report"] = str(report_path)
    return report
