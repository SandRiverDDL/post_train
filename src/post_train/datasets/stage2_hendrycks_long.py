from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from post_train.config import Stage2HendrycksLongDataConfig
from post_train.data import load_dataset_rows, make_sft_record, sample_rows, summarize_sft_dataset
from post_train.io import ensure_parent, read_jsonl, write_jsonl

QUESTION_KEYS = ("problem", "question", "query", "prompt")


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def _first_present(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_problem_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact


def _extract_level(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
    return None


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


def _target_long_count(total_size: int, *, long_ratio: int, short_ratio: int) -> int:
    ratio_sum = long_ratio + short_ratio
    return round(total_size * long_ratio / ratio_sum)


def prepare_stage2_hendrycks_long_dataset(cfg: Stage2HendrycksLongDataConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        trust_remote_code=True,
        local_files_only=False,
    )

    short_rows = read_jsonl(cfg.short_dataset_path)
    hendrycks_rows: list[dict[str, Any]] = []
    for config_name in cfg.hendrycks_config_names:
        rows = load_dataset_rows(
            cfg.hendrycks_dataset,
            split=cfg.hendrycks_split,
            config_name=config_name,
            cache_dir=cfg.cache_dir,
        )
        for row in rows:
            copied = dict(row)
            copied["__hendrycks_config_name__"] = config_name
            hendrycks_rows.append(copied)
    long_rows = load_dataset_rows(
        cfg.long_cot_dataset,
        split=cfg.long_cot_split,
        config_name=cfg.long_cot_config_name,
        cache_dir=cfg.cache_dir,
    )

    long_index: dict[str, dict[str, Any]] = {}
    duplicate_long_problem_count = 0
    for row in long_rows:
        normalized_problem = normalize_problem_text(_first_present(row, QUESTION_KEYS))
        if not normalized_problem:
            continue
        if normalized_problem in long_index:
            duplicate_long_problem_count += 1
            continue
        long_index[normalized_problem] = row

    levels_set = {int(level) for level in cfg.levels}
    filtered_hendrycks_rows: list[dict[str, Any]] = []
    for row in hendrycks_rows:
        level = _extract_level(row.get("level"))
        if level in levels_set:
            copied = dict(row)
            copied["__level__"] = level
            copied["__normalized_problem__"] = normalize_problem_text(_first_present(row, QUESTION_KEYS))
            filtered_hendrycks_rows.append(copied)

    matched_long_rows: list[dict[str, Any]] = []
    unmatched_preview_rows: list[dict[str, Any]] = []
    filtered_too_long = 0
    matched_solution_lengths: list[int] = []
    for index, row in enumerate(filtered_hendrycks_rows):
        normalized_problem = str(row.get("__normalized_problem__", ""))
        if not normalized_problem:
            continue
        matched_row = long_index.get(normalized_problem)
        if matched_row is None:
            if len(unmatched_preview_rows) < cfg.unmatched_preview_count:
                unmatched_preview_rows.append(
                    {
                        "problem": _first_present(row, QUESTION_KEYS),
                        "normalized_problem": normalized_problem,
                        "level": row.get("__level__"),
                        "hendrycks_config_name": row.get("__hendrycks_config_name__"),
                    }
                )
            continue

        record = make_sft_record(matched_row, index, source=cfg.long_cot_dataset)
        record.setdefault("meta", {})
        record["meta"]["source_dataset"] = cfg.long_cot_dataset
        record["meta"]["matched_level"] = int(row["__level__"])
        record["meta"]["normalized_problem"] = normalized_problem
        solution = str(record.get("solution", "") or "")
        solution_tokens = len(tokenizer(solution, add_special_tokens=False)["input_ids"])
        if solution_tokens > cfg.max_solution_tokens:
            filtered_too_long += 1
            continue
        record["meta"]["solution_tokens"] = solution_tokens
        matched_solution_lengths.append(solution_tokens)
        matched_long_rows.append(record)

    long_target = _target_long_count(
        cfg.total_size,
        long_ratio=cfg.long_ratio,
        short_ratio=cfg.short_ratio,
    )
    short_target = cfg.total_size - long_target
    if long_target > len(matched_long_rows):
        raise ValueError(
            f"long 样本不足：需要 {long_target} 条，匹配并过滤后只有 {len(matched_long_rows)} 条。"
        )

    sampled_long_rows = sample_rows(matched_long_rows, sample_size=long_target, seed=cfg.seed)
    sampled_long_problem_keys = {
        normalize_problem_text(_first_present(row, QUESTION_KEYS))
        for row in sampled_long_rows
    }

    eligible_short_rows: list[dict[str, Any]] = []
    short_overlap_excluded = 0
    for row in short_rows:
        normalized_problem = normalize_problem_text(_first_present(row, QUESTION_KEYS))
        if normalized_problem in sampled_long_problem_keys:
            short_overlap_excluded += 1
            continue
        eligible_short_rows.append(row)
    if short_target > len(eligible_short_rows):
        raise ValueError(
            f"short 样本不足：需要 {short_target} 条，排除重叠后只有 {len(eligible_short_rows)} 条。"
        )

    sampled_short_rows = sample_rows(eligible_short_rows, sample_size=short_target, seed=cfg.seed + 1)
    final_rows = list(sampled_long_rows) + list(sampled_short_rows)
    random.Random(cfg.seed).shuffle(final_rows)

    report = {
        "config": {
            "short_dataset_path": str(cfg.short_dataset_path),
            "hendrycks_dataset": cfg.hendrycks_dataset,
            "hendrycks_split": cfg.hendrycks_split,
            "hendrycks_config_name": cfg.hendrycks_config_name,
            "hendrycks_config_names": list(cfg.hendrycks_config_names),
            "long_cot_dataset": cfg.long_cot_dataset,
            "long_cot_split": cfg.long_cot_split,
            "long_cot_config_name": cfg.long_cot_config_name,
            "seed": cfg.seed,
            "total_size": cfg.total_size,
            "long_ratio": cfg.long_ratio,
            "short_ratio": cfg.short_ratio,
            "levels": sorted(levels_set),
            "max_solution_tokens": cfg.max_solution_tokens,
            "unmatched_preview_count": cfg.unmatched_preview_count,
        },
        "sources": {
            "short": {
                "raw_rows": len(short_rows),
                "eligible_rows": len(eligible_short_rows),
                "selected_rows": len(sampled_short_rows),
                "overlap_excluded_rows": short_overlap_excluded,
            },
            "hendrycks_filter": {
                "raw_rows": len(hendrycks_rows),
                "level_filtered_rows": len(filtered_hendrycks_rows),
                "matched_problem_rows": len(matched_long_rows),
                "matched_problem_rate": (
                    len(matched_long_rows) / len(filtered_hendrycks_rows)
                    if filtered_hendrycks_rows
                    else 0.0
                ),
                "duplicate_long_problem_count": duplicate_long_problem_count,
            },
            "matched_long": {
                "source_dataset": cfg.long_cot_dataset,
                "available_rows": len(matched_long_rows),
                "selected_rows": len(sampled_long_rows),
            },
            "long_cot": {
                "raw_rows": len(long_rows),
            },
        },
        "filters": {
            "filtered_too_long_solution": filtered_too_long,
            "unmatched_rows": max(len(filtered_hendrycks_rows) - len(matched_long_rows) - filtered_too_long, 0),
        },
        "solution_tokens": {
            "matched_long": _length_stats(matched_solution_lengths),
        },
        "dataset_summary": summarize_sft_dataset(final_rows),
    }

    output_path = write_jsonl(cfg.output_path, final_rows)
    report_path = _write_json(cfg.report_path, report)
    unmatched_preview_path = write_jsonl(cfg.unmatched_preview_path, unmatched_preview_rows)
    return {
        "output_path": output_path,
        "report_path": report_path,
        "unmatched_preview_path": unmatched_preview_path,
        "report": report,
    }
