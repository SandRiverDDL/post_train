from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from post_train.answers import (
    are_equivalent,
    ensure_boxed_final_answer,
    extract_final_answer,
    extract_relaxed_final_answer,
    has_boxed_final_answer,
)
from post_train.schemas import SFTRecord

QUESTION_KEYS = ("problem", "question", "query", "prompt")
SOLUTION_KEYS = ("solution", "response", "completion")
FINAL_ANSWER_KEYS = ("final_answer", "answer")
ID_KEYS = ("id", "problem_id", "uuid")
GSM8K_ANSWER_RE = re.compile(r"####\s*(.+)$", re.DOTALL)


def _first_present(row: dict[str, Any], candidates: tuple[str, ...], default: str = "") -> str:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _row_id(row: dict[str, Any], fallback_index: int) -> str:
    for key in ID_KEYS:
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(fallback_index)


def _extract_source_final_answer(source: str, row: dict[str, Any], fallback: str) -> str:
    if source == "gsm8k":
        answer = row.get("answer")
        if isinstance(answer, str):
            match = GSM8K_ANSWER_RE.search(answer)
            if match:
                return match.group(1).strip()
    return fallback


def clean_solution_text(solution: str) -> str:
    if not solution.strip():
        return solution

    patterns = [
        re.compile(r"^\s*#{1,6}\s*solution\.?\s*$", re.IGNORECASE),
        re.compile(r"^\s*solution(?:\s+\d+)?\.?\s*$", re.IGNORECASE),
        re.compile(r"^\s*answer\.?\s*$", re.IGNORECASE),
        re.compile(r"^\s*soln\.?\s*$", re.IGNORECASE),
    ]
    lines = solution.splitlines()
    start = 0
    while start < len(lines):
        line = lines[start].strip()
        if not line:
            start += 1
            continue
        if any(pattern.match(line) for pattern in patterns):
            start += 1
            continue
        break
    cleaned = "\n".join(lines[start:]).strip()
    return cleaned or solution.strip()


def make_sft_record(row: dict[str, Any], index: int, *, source: str) -> dict[str, Any]:
    question = _first_present(row, QUESTION_KEYS)
    raw_solution = clean_solution_text(_first_present(row, SOLUTION_KEYS))
    final_answer = _first_present(row, FINAL_ANSWER_KEYS)
    final_answer = _extract_source_final_answer(source, row, final_answer)
    if not final_answer:
        final_answer = extract_relaxed_final_answer(raw_solution) or ""

    if raw_solution:
        solution = ensure_boxed_final_answer(raw_solution, final_answer)
    elif final_answer:
        solution = f"\\boxed{{{final_answer}}}"
    else:
        solution = ""

    normalized_final_answer = extract_relaxed_final_answer(solution) or final_answer
    record = SFTRecord(
        id=_row_id(row, index),
        question=question,
        solution=solution,
        final_answer=normalized_final_answer,
        meta={"source": source},
    )
    return record.model_dump()


def make_eval_record(row: dict[str, Any], index: int, *, source: str) -> dict[str, Any]:
    record = make_sft_record(row, index, source=source)
    return {
        "id": record["id"],
        "question": record["question"],
        "final_answer": record["final_answer"],
        "meta": record.get("meta", {}),
    }


def split_train_dev(
    rows: Iterable[dict[str, Any]],
    *,
    train_size: int,
    dev_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = list(rows)
    required = train_size + dev_size
    if required > len(records):
        raise ValueError(f"样本不足：需要 {required} 条，实际只有 {len(records)} 条。")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_rows = shuffled[:train_size]
    dev_rows = shuffled[train_size : train_size + dev_size]
    return train_rows, dev_rows


def summarize_sft_dataset(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    total = len(records)
    boxed = sum(has_boxed_final_answer(str(row.get("solution", ""))) for row in records)
    empty_final_answer = sum(not str(row.get("final_answer", "")).strip() for row in records)
    parse_success = 0
    consistent_final_answer = 0

    for row in records:
        solution = str(row.get("solution", ""))
        final_answer = str(row.get("final_answer", "")).strip()
        parsed_answer = extract_final_answer(solution)
        if parsed_answer:
            parse_success += 1
        if parsed_answer and final_answer and are_equivalent(parsed_answer, final_answer):
            consistent_final_answer += 1

    boxed_rate = boxed / total if total else 0.0
    parse_success_rate = parse_success / total if total else 0.0
    consistent_rate = consistent_final_answer / total if total else 0.0
    empty_rate = empty_final_answer / total if total else 0.0
    return {
        "total": total,
        "boxed": boxed,
        "boxed_rate": boxed_rate,
        "parse_success": parse_success,
        "parse_success_rate": parse_success_rate,
        "consistent_final_answer": consistent_final_answer,
        "consistent_rate": consistent_rate,
        "empty_final_answer": empty_final_answer,
        "empty_rate": empty_rate,
    }


def load_dataset_rows(
    dataset_name: str,
    *,
    split: str,
    config_name: str | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    return [dict(dataset[index]) for index in range(len(dataset))]


def prepare_stage1_artifacts(
    *,
    dataset_name: str,
    split: str,
    train_size: int,
    dev_size: int,
    seed: int,
    cache_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_dataset_rows(dataset_name, split=split, cache_dir=cache_dir)
    sft_rows = [make_sft_record(row, index, source=dataset_name) for index, row in enumerate(rows)]
    return split_train_dev(sft_rows, train_size=train_size, dev_size=dev_size, seed=seed)


def prepare_benchmark_artifact(
    *,
    dataset_name: str,
    split: str,
    source: str,
    config_name: str | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    rows = load_dataset_rows(dataset_name, split=split, config_name=config_name, cache_dir=cache_dir)
    return [make_eval_record(row, index, source=source) for index, row in enumerate(rows)]


def preview_rows(rows: Iterable[dict[str, Any]], *, count: int = 3) -> str:
    previews: list[str] = []
    for row in list(rows)[:count]:
        solution = str(row.get("solution", f"\\boxed{{{row.get('final_answer', '')}}}"))
        previews.append(
            "\n".join(
                [
                    "=" * 80,
                    f"id: {row.get('id', '')}",
                    str(row.get("question", ""))[:300],
                    "--- solution tail ---",
                    "\n".join(solution.splitlines()[-4:]),
                ]
            )
        )
    return "\n".join(previews)


def read_json_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows
