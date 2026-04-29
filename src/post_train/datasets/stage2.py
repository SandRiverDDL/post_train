from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from post_train.answers import (
    ANSWER_LINE_RE,
    are_equivalent,
    extract_final_answer,
    extract_relaxed_final_answer,
    has_boxed_final_answer,
)
from post_train.config import Stage2DataConfig
from post_train.data import clean_solution_text, load_dataset_rows, sample_rows
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.schemas import SFTRecord

BOXED_MARKER = r"\boxed{"


def _normalize_question(row: dict[str, Any]) -> str:
    clean_problem = str(row.get("clean_problem", "") or "").strip()
    if clean_problem:
        return clean_problem
    return str(row.get("problem", "") or "").strip()


def _is_trailing_boxed_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if BOXED_MARKER in stripped:
        return True
    return bool(re.search(r"\\\(\s*\\boxed\{", stripped))


def strip_stage2_answer_tail(solution: str) -> str:
    lines = solution.rstrip().splitlines()
    while lines:
        candidate = lines[-1].strip()
        if not candidate:
            lines.pop()
            continue
        if ANSWER_LINE_RE.match(candidate):
            lines.pop()
            continue
        if _is_trailing_boxed_line(candidate):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def normalize_stage2_solution(raw_solution: str, final_answer: str | None) -> tuple[str, dict[str, int | str]]:
    solution = clean_solution_text(raw_solution)
    answer = (final_answer or extract_relaxed_final_answer(solution) or "").strip()
    boxed_count_before = solution.count(BOXED_MARKER)
    if not answer:
        return solution, {
            "action": "missing_answer",
            "boxed_count_before": boxed_count_before,
            "boxed_count_after": boxed_count_before,
        }

    parsed_answer = extract_final_answer(solution)
    if parsed_answer and are_equivalent(parsed_answer, answer) and boxed_count_before == 1:
        normalized = solution.strip()
        return normalized, {
            "action": "already_valid_boxed",
            "boxed_count_before": boxed_count_before,
            "boxed_count_after": normalized.count(BOXED_MARKER),
        }

    suffix = f"\\boxed{{{answer}}}"
    if not solution:
        return suffix, {
            "action": "appended_boxed",
            "boxed_count_before": boxed_count_before,
            "boxed_count_after": 1,
        }

    stripped = strip_stage2_answer_tail(solution)
    if not stripped:
        return suffix, {
            "action": "rewritten_tail",
            "boxed_count_before": boxed_count_before,
            "boxed_count_after": 1,
        }

    normalized = f"{stripped}\n\n{suffix}"
    action = "appended_boxed" if stripped == solution.strip() and boxed_count_before == 0 else "rewritten_tail"
    return normalized, {
        "action": action,
        "boxed_count_before": boxed_count_before,
        "boxed_count_after": normalized.count(BOXED_MARKER),
    }


def make_math220k_record(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    question_type = str(row.get("question_type", "") or "").strip()
    if question_type == "MCQ":
        return None

    question = _normalize_question(row)
    raw_solution = clean_solution_text(str(row.get("solution", "") or ""))
    final_answer = str(row.get("answer", "") or "").strip()
    if not final_answer:
        final_answer = extract_relaxed_final_answer(raw_solution) or ""

    solution, normalization_meta = normalize_stage2_solution(raw_solution, final_answer)

    normalized_final_answer = extract_relaxed_final_answer(solution) or final_answer
    record = SFTRecord(
        id=f"math220k-{index}",
        question=question,
        solution=solution,
        final_answer=normalized_final_answer,
        meta={
            "source": "math220k",
            "source_dataset": "qingy2024/OpenR1-Math-220k-Cleaned",
            "problem_type": str(row.get("problem_type", "") or ""),
            "question_type": question_type,
            "normalization_action": normalization_meta["action"],
            "boxed_count_before": normalization_meta["boxed_count_before"],
            "boxed_count_after": normalization_meta["boxed_count_after"],
        },
    )
    return record.model_dump()


def validate_sft_record(row: dict[str, Any]) -> tuple[bool, str]:
    question = str(row.get("question", "") or "").strip()
    solution = str(row.get("solution", "") or "").strip()
    final_answer = str(row.get("final_answer", "") or "").strip()
    if not question:
        return False, "empty_question"
    if not solution:
        return False, "empty_solution"
    if not final_answer:
        return False, "empty_final_answer"
    if not has_boxed_final_answer(solution):
        return False, "missing_boxed"
    parsed_answer = extract_final_answer(solution)
    if not parsed_answer:
        return False, "parse_failed"
    if not are_equivalent(parsed_answer, final_answer):
        return False, "answer_mismatch"
    return True, "ok"


def token_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def assign_selection_pool(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    short_max_tokens: int,
    long_max_tokens: int,
) -> str | None:
    source = str(row.get("meta", {}).get("source", ""))
    if source == "stage1":
        return "stage1_random"

    solution = str(row.get("solution", "") or "")
    solution_tokens = token_length(tokenizer, solution)
    if solution_tokens < short_max_tokens:
        return "math220k_short"
    if solution_tokens < long_max_tokens:
        return "math220k_long"
    return None


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def prepare_stage2_dataset(cfg: Stage2DataConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        trust_remote_code=True,
        local_files_only=False,
    )

    report: dict[str, Any] = {
        "config": {
            "total_size": cfg.total_size,
            "stage1_random_quota": cfg.stage1_random_quota,
            "math220k_short_quota": cfg.math220k_short_quota,
            "math220k_long_quota": cfg.math220k_long_quota,
            "short_max_tokens": cfg.short_max_tokens,
            "long_max_tokens": cfg.long_max_tokens,
            "exact_dedup": cfg.exact_dedup,
            "near_dedup": cfg.near_dedup,
            "decontam": cfg.decontam,
        },
        "sources": {},
        "filters": {},
        "pools": {},
        "normalization": {
            "already_valid_boxed_count": 0,
            "appended_boxed_count": 0,
            "rewritten_tail_count": 0,
            "double_boxed_detected_count": 0,
            "double_boxed_after_normalization_count": 0,
        },
    }

    stage1_rows = [SFTRecord.model_validate(row).model_dump() for row in read_jsonl(cfg.stage1_input)]
    for row in stage1_rows:
        row.setdefault("meta", {})
        row["meta"]["source"] = "stage1"
        row["meta"]["source_dataset"] = str(cfg.stage1_input)
    report["sources"]["stage1"] = {"raw": len(stage1_rows)}

    math220k_raw_rows = load_dataset_rows(
        cfg.math220k_dataset,
        split=cfg.math220k_split,
        cache_dir=cfg.cache_dir,
    )
    report["sources"]["math220k"] = {"raw": len(math220k_raw_rows)}

    math220k_rows: list[dict[str, Any]] = []
    filtered_mcq = 0
    for index, row in enumerate(math220k_raw_rows):
        normalized = make_math220k_record(row, index)
        if normalized is None:
            filtered_mcq += 1
            continue
        normalization_action = str(normalized.get("meta", {}).get("normalization_action", ""))
        if normalization_action == "already_valid_boxed":
            report["normalization"]["already_valid_boxed_count"] += 1
        elif normalization_action == "appended_boxed":
            report["normalization"]["appended_boxed_count"] += 1
        elif normalization_action == "rewritten_tail":
            report["normalization"]["rewritten_tail_count"] += 1
        if int(normalized.get("meta", {}).get("boxed_count_before", 0)) >= 2:
            report["normalization"]["double_boxed_detected_count"] += 1
        if int(normalized.get("meta", {}).get("boxed_count_after", 0)) >= 2:
            report["normalization"]["double_boxed_after_normalization_count"] += 1
        math220k_rows.append(normalized)
    report["sources"]["math220k"]["filtered_mcq"] = filtered_mcq
    report["sources"]["math220k"]["normalized"] = len(math220k_rows)

    valid_stage1_rows: list[dict[str, Any]] = []
    valid_math220k_rows: list[dict[str, Any]] = []
    filter_counts = {
        "empty_question": 0,
        "empty_solution": 0,
        "empty_final_answer": 0,
        "missing_boxed": 0,
        "parse_failed": 0,
        "answer_mismatch": 0,
    }

    for row in stage1_rows:
        ok, reason = validate_sft_record(row)
        if ok:
            valid_stage1_rows.append(row)
        else:
            filter_counts[reason] += 1

    for row in math220k_rows:
        ok, reason = validate_sft_record(row)
        if ok:
            valid_math220k_rows.append(row)
        else:
            filter_counts[reason] += 1

    report["filters"] = filter_counts
    report["sources"]["stage1"]["valid"] = len(valid_stage1_rows)
    report["sources"]["math220k"]["valid"] = len(valid_math220k_rows)

    stage1_pool: list[dict[str, Any]] = []
    math220k_short_pool: list[dict[str, Any]] = []
    math220k_long_pool: list[dict[str, Any]] = []

    for row in valid_stage1_rows:
        row["meta"]["selection_pool"] = "stage1_random"
        stage1_pool.append(row)

    for row in valid_math220k_rows:
        selection_pool = assign_selection_pool(
            row,
            tokenizer=tokenizer,
            short_max_tokens=cfg.short_max_tokens,
            long_max_tokens=cfg.long_max_tokens,
        )
        if selection_pool is None:
            continue
        row["meta"]["selection_pool"] = selection_pool
        if selection_pool == "math220k_short":
            math220k_short_pool.append(row)
        else:
            math220k_long_pool.append(row)

    report["pools"] = {
        "stage1_random": {"available": len(stage1_pool), "quota": cfg.stage1_random_quota},
        "math220k_short": {"available": len(math220k_short_pool), "quota": cfg.math220k_short_quota},
        "math220k_long": {"available": len(math220k_long_pool), "quota": cfg.math220k_long_quota},
    }

    sampled_stage1 = sample_rows(stage1_pool, sample_size=cfg.stage1_random_quota, seed=cfg.seed)
    sampled_short = sample_rows(math220k_short_pool, sample_size=cfg.math220k_short_quota, seed=cfg.seed + 1)
    sampled_long = sample_rows(math220k_long_pool, sample_size=cfg.math220k_long_quota, seed=cfg.seed + 2)

    final_rows = sampled_stage1 + sampled_short + sampled_long
    random.Random(cfg.seed).shuffle(final_rows)
    if len(final_rows) != cfg.total_size:
        raise ValueError("stage2 最终样本数与配置 total_size 不一致。")

    write_jsonl(cfg.output_path, final_rows)
    _write_json(cfg.report_path, report)

    if cfg.save_intermediate:
        write_jsonl(cfg.intermediate_dir / "stage1.valid.jsonl", valid_stage1_rows)
        write_jsonl(cfg.intermediate_dir / "math220k.valid.jsonl", valid_math220k_rows)
        write_jsonl(cfg.intermediate_dir / "math220k.short.jsonl", math220k_short_pool)
        write_jsonl(cfg.intermediate_dir / "math220k.long.jsonl", math220k_long_pool)

    return {
        "output_path": str(cfg.output_path),
        "report_path": str(cfg.report_path),
        "report": report,
        "total_rows": len(final_rows),
    }
