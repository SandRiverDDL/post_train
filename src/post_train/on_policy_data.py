from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from post_train.data import load_dataset_rows, make_eval_record, sample_rows
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.schemas import SFTRecord
from post_train.simpo_data import load_excluded_ids, sample_candidate_responses

WHITESPACE_RE = re.compile(r"\s+")


class InsufficientRetainedSamplesError(ValueError):
    """当 retained 样本不足时抛出。"""


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def normalize_question(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def load_excluded_questions(paths: list[str | Path]) -> set[str]:
    excluded_questions: set[str] = set()
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            continue
        for row in read_jsonl(candidate):
            question = normalize_question(str(row.get("question", "")))
            if question:
                excluded_questions.add(question)
    return excluded_questions


def load_query_candidates(cfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = load_dataset_rows(
        cfg.query_dataset,
        split=cfg.query_split,
        config_name=cfg.query_config_name,
        cache_dir=cfg.cache_dir,
    )
    eval_rows = [
        make_eval_record(row, index, source=cfg.query_source)
        for index, row in enumerate(raw_rows)
    ]
    excluded_ids = load_excluded_ids(cfg.exclude_paths, source_name=cfg.query_source)
    excluded_questions = load_excluded_questions(cfg.exclude_paths)
    available_rows = [
        row
        for row in eval_rows
        if str(row.get("id", "")) not in excluded_ids
        and normalize_question(str(row.get("question", ""))) not in excluded_questions
    ]
    return available_rows, {
        "raw_rows": len(raw_rows),
        "excluded_ids": len(excluded_ids),
        "excluded_questions": len(excluded_questions),
        "available_rows": len(available_rows),
        "sampled_queries": 0,
        "effective_query_count": 0,
    }


def build_query_pool(cfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available_rows, report = load_query_candidates(cfg)
    effective_query_count = min(cfg.query_count, len(available_rows))
    if effective_query_count == 0:
        raise ValueError("query 池为空，无法构造 on-policy query。")
    query_rows = sample_rows(available_rows, sample_size=effective_query_count, seed=cfg.seed)
    report["sampled_queries"] = len(query_rows)
    report["effective_query_count"] = effective_query_count
    return query_rows, report


def _empty_selection_report(raw_sample_count: int) -> dict[str, Any]:
    return {
        "raw_sample_count": raw_sample_count,
        "questions_with_any_valid": 0,
        "questions_without_valid": 0,
        "kept": 0,
        "multiple_valid_candidates": 0,
        "filtered_empty_text": 0,
        "filtered_parse_fail": 0,
        "filtered_incorrect": 0,
        "filtered_too_long": 0,
        "retained_ratio": 0.0,
        "avg_kept_output_tokens": 0.0,
        "p50_kept_output_tokens": 0,
        "p90_kept_output_tokens": 0,
        "max_completion_tokens": 0,
        "min_retained_count": 0,
    }


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def build_retained_sft_dataset(
    raw_samples: list[dict[str, Any]],
    cfg,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retained_rows: list[dict[str, Any]] = []
    report = _empty_selection_report(len(raw_samples))
    report["max_completion_tokens"] = cfg.max_completion_tokens
    report["min_retained_count"] = cfg.min_retained_count
    kept_lengths: list[int] = []

    for sample in raw_samples:
        valid_candidates: list[dict[str, Any]] = []
        for response in sample.get("responses", []):
            text = str(response.get("text", ""))
            if not text.strip():
                report["filtered_empty_text"] += 1
                continue
            if not bool(response.get("parse_ok")):
                report["filtered_parse_fail"] += 1
                continue
            if not bool(response.get("correct")):
                report["filtered_incorrect"] += 1
                continue
            output_tokens = int(response.get("output_tokens", 0))
            if output_tokens > cfg.max_completion_tokens:
                report["filtered_too_long"] += 1
                continue
            valid_candidates.append(response)

        if not valid_candidates:
            report["questions_without_valid"] += 1
            continue

        report["questions_with_any_valid"] += 1
        if len(valid_candidates) >= 2:
            report["multiple_valid_candidates"] += 1

        chosen = min(
            enumerate(valid_candidates),
            key=lambda item: (int(item[1].get("output_tokens", 0)), item[0]),
        )[1]
        output_tokens = int(chosen.get("output_tokens", 0))
        kept_lengths.append(output_tokens)

        record = SFTRecord(
            id=str(sample.get("id", "")),
            question=str(sample.get("question", "")),
            solution=str(chosen.get("text", "")),
            final_answer=str(sample.get("final_answer", "")),
            meta={
                "source": str(sample.get("meta", {}).get("source", "")),
                "source_dataset": cfg.query_dataset,
                "source_question_id": str(sample.get("id", "")),
                "generation_model": cfg.generation_model,
                "round_name": cfg.round_name,
                "output_tokens": output_tokens,
                "responses_per_query": cfg.responses_per_query,
            },
        )
        retained_rows.append(record.model_dump())

    report["kept"] = len(retained_rows)
    report["retained_ratio"] = (len(retained_rows) / len(raw_samples)) if raw_samples else 0.0
    if kept_lengths:
        ordered = sorted(kept_lengths)
        report["avg_kept_output_tokens"] = sum(ordered) / len(ordered)
        report["p50_kept_output_tokens"] = _percentile(ordered, 0.5)
        report["p90_kept_output_tokens"] = _percentile(ordered, 0.9)
    return retained_rows, report


def prepare_on_policy_sft_dataset(cfg) -> dict[str, Any]:
    return prepare_on_policy_sft_dataset_from_queries(cfg)


def prepare_on_policy_sft_dataset_from_queries(
    cfg,
    *,
    query_rows: list[dict[str, Any]] | None = None,
    query_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if query_rows is None:
        query_rows, query_report = build_query_pool(cfg)
    else:
        query_rows = list(query_rows)
        if not query_rows:
            raise ValueError("传入的 query_rows 为空，无法构造 on-policy 数据。")
        query_report = dict(query_report or {})
        query_report.setdefault("sampled_queries", len(query_rows))
        query_report.setdefault("effective_query_count", len(query_rows))
    write_jsonl(cfg.query_output_path, query_rows)

    raw_samples, sampling_report = sample_candidate_responses(query_rows, cfg)
    write_jsonl(cfg.raw_samples_output_path, raw_samples)

    retained_rows, retained_report = build_retained_sft_dataset(raw_samples, cfg)
    report = {
        "round_name": cfg.round_name,
        "query_pool": query_report,
        "sampling": sampling_report,
        "retained": retained_report,
    }
    report_path = _write_json(cfg.report_path, report)

    if len(retained_rows) < cfg.min_retained_count:
        raise InsufficientRetainedSamplesError(
            f"retained 样本不足：需要至少 {cfg.min_retained_count} 条，实际只有 {len(retained_rows)} 条。"
        )

    retained_output_path = write_jsonl(cfg.retained_output_path, retained_rows)
    return {
        "query_output_path": str(cfg.query_output_path),
        "raw_samples_output_path": str(cfg.raw_samples_output_path),
        "retained_output_path": str(retained_output_path),
        "report_path": str(report_path),
        "report": report,
    }
