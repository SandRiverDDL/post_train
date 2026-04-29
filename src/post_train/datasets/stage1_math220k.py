from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_train.config import Stage1Math220kDataConfig
from post_train.data import load_dataset_rows, sample_rows, summarize_sft_dataset
from post_train.io import ensure_parent, write_jsonl
from post_train.datasets.stage2 import make_math220k_record, token_length, validate_sft_record


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


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


def prepare_stage1_math220k_dataset(cfg: Stage1Math220kDataConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        trust_remote_code=True,
        local_files_only=False,
    )

    raw_rows = load_dataset_rows(
        cfg.dataset_name,
        split=cfg.dataset_split,
        cache_dir=cfg.cache_dir,
    )

    normalized_rows: list[dict[str, Any]] = []
    filtered_mcq = 0
    for index, raw_row in enumerate(raw_rows):
        normalized = make_math220k_record(raw_row, index)
        if normalized is None:
            filtered_mcq += 1
            continue
        normalized_rows.append(normalized)

    filter_counts = {
        "empty_question": 0,
        "empty_solution": 0,
        "empty_final_answer": 0,
        "missing_boxed": 0,
        "parse_failed": 0,
        "answer_mismatch": 0,
        "too_short": 0,
        "too_long": 0,
    }
    eligible_rows: list[dict[str, Any]] = []
    eligible_lengths: list[int] = []
    for row in normalized_rows:
        ok, reason = validate_sft_record(row)
        if not ok:
            filter_counts[reason] += 1
            continue
        solution_tokens = token_length(tokenizer, str(row.get("solution", "") or ""))
        if solution_tokens < cfg.min_solution_tokens:
            filter_counts["too_short"] += 1
            continue
        if solution_tokens >= cfg.max_solution_tokens:
            filter_counts["too_long"] += 1
            continue
        row.setdefault("meta", {})
        row["meta"]["solution_tokens"] = solution_tokens
        eligible_rows.append(row)
        eligible_lengths.append(solution_tokens)

    if cfg.sample_size > len(eligible_rows):
        raise ValueError(f"样本不足：需要 {cfg.sample_size} 条，长度筛选后只有 {len(eligible_rows)} 条。")

    sampled_rows = sample_rows(eligible_rows, sample_size=cfg.sample_size, seed=cfg.seed)
    report = {
        "config": {
            "dataset_name": cfg.dataset_name,
            "dataset_split": cfg.dataset_split,
            "seed": cfg.seed,
            "sample_size": cfg.sample_size,
            "min_solution_tokens": cfg.min_solution_tokens,
            "max_solution_tokens": cfg.max_solution_tokens,
        },
        "sources": {
            "math220k": {
                "raw_rows": len(raw_rows),
                "filtered_mcq": filtered_mcq,
                "normalized_rows": len(normalized_rows),
                "eligible_rows": len(eligible_rows),
                "selected_rows": len(sampled_rows),
            }
        },
        "filters": filter_counts,
        "solution_tokens": {
            "eligible_rows": _length_stats(eligible_lengths),
        },
        "dataset_summary": summarize_sft_dataset(sampled_rows),
    }

    output_path = write_jsonl(cfg.output_path, sampled_rows)
    report_path = _write_json(cfg.report_path, report)
    return {
        "output_path": output_path,
        "report_path": report_path,
        "report": report,
    }
