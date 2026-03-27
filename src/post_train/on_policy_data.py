from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from post_train.data import load_dataset_rows, make_eval_record, read_json_rows, sample_rows
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.schemas import SFTRecord
from post_train.simpo_data import load_excluded_ids, sample_candidate_responses

WHITESPACE_RE = re.compile(r"\s+")
SELECTOR_ANY_CORRECT = "any_correct_shortest"
SELECTOR_MIXED_ONLY = "mixed_only_shortest"
SUPPORTED_SELECTORS = (SELECTOR_ANY_CORRECT, SELECTOR_MIXED_ONLY)


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
    if cfg.query_file_path is not None:
        raw_rows = read_json_rows(cfg.query_file_path)
    else:
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
        "registry_size": len(available_rows),
        "available_rows": len(available_rows),
        "sampled_queries": 0,
        "effective_query_count": 0,
        "query_limit": cfg.query_limit,
    }


def build_query_pool(cfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available_rows, report = load_query_candidates(cfg)
    if cfg.query_limit is None:
        effective_query_count = len(available_rows)
    else:
        effective_query_count = min(cfg.query_limit, len(available_rows))
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
        "questions_with_mixed_outcome": 0,
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


def summarize_raw_samples(raw_samples: list[dict[str, Any]], *, responses_per_query: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(raw_samples),
        "responses_per_query": responses_per_query,
        "all_correct": 0,
        "mixed": 0,
        "all_wrong": 0,
    }
    for correct_count in range(responses_per_query + 1):
        summary[f"correct_{correct_count}_of_{responses_per_query}"] = 0

    for sample in raw_samples:
        response_summary = dict(sample.get("response_summary", {}))
        correct_count = int(response_summary.get("correct_count", 0))
        response_count = int(response_summary.get("response_count", responses_per_query))
        key = f"correct_{correct_count}_of_{response_count}"
        summary.setdefault(key, 0)
        summary[key] = int(summary.get(key, 0)) + 1
        if correct_count == 0:
            summary["all_wrong"] += 1
        elif response_count > 0 and correct_count == response_count:
            summary["all_correct"] += 1
        else:
            summary["mixed"] += 1
    return summary


def annotate_raw_samples(raw_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated_rows: list[dict[str, Any]] = []
    for sample in raw_samples:
        responses = list(sample.get("responses", []))
        correct_count = sum(1 for row in responses if bool(row.get("correct")))
        response_count = len(responses)
        response_summary = {
            "response_count": response_count,
            "correct_count": correct_count,
            "has_any_correct": correct_count > 0,
            "has_mixed_outcome": 0 < correct_count < response_count,
        }
        enriched_sample = dict(sample)
        enriched_sample["response_summary"] = response_summary
        annotated_rows.append(enriched_sample)
    return annotated_rows


def _selector_output_path(base_path: str | Path, selector_name: str, *, primary_selector: str) -> Path:
    output_path = Path(base_path)
    if selector_name == primary_selector:
        return output_path
    suffix = "".join(output_path.suffixes)
    stem = output_path.name[: -len(suffix)] if suffix else output_path.name
    if suffix:
        filename = f"{stem}.{selector_name}{suffix}"
    else:
        filename = f"{output_path.name}.{selector_name}"
    return output_path.with_name(filename)


def _select_candidates(
    sample: dict[str, Any],
    report: dict[str, Any],
    *,
    cfg,
    selector_name: str,
) -> list[dict[str, Any]]:
    valid_candidates: list[dict[str, Any]] = []
    incorrect_count = 0
    for response in sample.get("responses", []):
        text = str(response.get("text", ""))
        if not text.strip():
            report["filtered_empty_text"] += 1
            incorrect_count += 1
            continue
        if not bool(response.get("parse_ok")):
            report["filtered_parse_fail"] += 1
            incorrect_count += 1
            continue
        if not bool(response.get("correct")):
            report["filtered_incorrect"] += 1
            incorrect_count += 1
            continue
        output_tokens = int(response.get("output_tokens", 0))
        if output_tokens > cfg.max_completion_tokens:
            report["filtered_too_long"] += 1
            continue
        valid_candidates.append(response)

    if valid_candidates:
        report["questions_with_any_valid"] += 1
        if len(valid_candidates) >= 2:
            report["multiple_valid_candidates"] += 1
    else:
        report["questions_without_valid"] += 1

    if selector_name == SELECTOR_ANY_CORRECT:
        return valid_candidates

    if valid_candidates and incorrect_count > 0:
        report["questions_with_mixed_outcome"] += 1
        return valid_candidates
    if valid_candidates:
        report["questions_without_valid"] += 1
    return []


def build_retained_sft_dataset(
    raw_samples: list[dict[str, Any]],
    cfg,
    *,
    selector_name: str = SELECTOR_ANY_CORRECT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retained_rows: list[dict[str, Any]] = []
    report = _empty_selection_report(len(raw_samples))
    report["max_completion_tokens"] = cfg.max_completion_tokens
    report["min_retained_count"] = cfg.min_retained_count
    report["selector_name"] = selector_name
    kept_lengths: list[int] = []

    for sample in raw_samples:
        valid_candidates = _select_candidates(sample, report, cfg=cfg, selector_name=selector_name)
        if not valid_candidates:
            continue

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
                "source_dataset": str(cfg.query_file_path or cfg.query_dataset),
                "source_question_id": str(sample.get("id", "")),
                "generation_model": cfg.generation_model,
                "round_name": cfg.round_name,
                "output_tokens": output_tokens,
                "responses_per_query": cfg.responses_per_query,
                "selector_name": selector_name,
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
    raw_samples = annotate_raw_samples(raw_samples)
    write_jsonl(cfg.raw_samples_output_path, raw_samples)

    selector_rows: dict[str, list[dict[str, Any]]] = {}
    selector_reports: dict[str, dict[str, Any]] = {}
    retained_output_paths: dict[str, str] = {}
    for selector_name in SUPPORTED_SELECTORS:
        retained_rows, retained_report = build_retained_sft_dataset(
            raw_samples,
            cfg,
            selector_name=selector_name,
        )
        output_path = _selector_output_path(
            cfg.retained_output_path,
            selector_name,
            primary_selector=cfg.primary_selector,
        )
        write_jsonl(output_path, retained_rows)
        selector_rows[selector_name] = retained_rows
        selector_reports[selector_name] = retained_report
        retained_output_paths[selector_name] = str(output_path)

    raw_sample_report = summarize_raw_samples(
        raw_samples,
        responses_per_query=cfg.responses_per_query,
    )
    primary_retained_report = selector_reports[cfg.primary_selector]
    report = {
        "round_name": cfg.round_name,
        "query_pool": query_report,
        "sampling": sampling_report,
        "raw_samples": raw_sample_report,
        "retained": primary_retained_report,
        "retained_selectors": selector_reports,
        "primary_selector": cfg.primary_selector,
    }
    report_path = _write_json(cfg.report_path, report)

    primary_rows = selector_rows[cfg.primary_selector]
    if len(primary_rows) < cfg.min_retained_count:
        raise InsufficientRetainedSamplesError(
            f"retained 样本不足：selector={cfg.primary_selector}，需要至少 {cfg.min_retained_count} 条，实际只有 {len(primary_rows)} 条。"
        )
    return {
        "query_output_path": str(cfg.query_output_path),
        "raw_samples_output_path": str(cfg.raw_samples_output_path),
        "retained_output_path": retained_output_paths[cfg.primary_selector],
        "retained_output_paths": retained_output_paths,
        "report_path": str(report_path),
        "report": report,
    }
