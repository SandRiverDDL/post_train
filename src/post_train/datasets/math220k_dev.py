from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from post_train.data import load_dataset_rows, sample_rows
from post_train.io import ensure_parent, write_jsonl
from post_train.datasets.stage2 import make_math220k_record, token_length, validate_sft_record

DATASET_NAME = "qingy2024/OpenR1-Math-220k-Cleaned"
DEFAULT_TOKENIZER_NAME = "/home/fsw/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"

PROFILE_SPECS: dict[str, list[tuple[str, int, int, int]]] = {
    "main150": [
        ("lt512", 0, 512, 70),
        ("tok512_768", 512, 768, 35),
        ("tok768_1380", 768, 1380, 45),
    ],
    "short150": [
        ("lt512", 0, 512, 150),
    ],
}


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def profile_names() -> list[str]:
    return sorted(PROFILE_SPECS)


def profile_spec(profile: str) -> list[tuple[str, int, int, int]]:
    if profile not in PROFILE_SPECS:
        supported = ", ".join(profile_names())
        raise ValueError(f"不支持的 profile: {profile}。可选值：{supported}")
    return PROFILE_SPECS[profile]


def default_output_path(profile: str) -> Path:
    return Path(f"data/eval/math220k_dev_{profile}.jsonl")


def default_report_path(profile: str) -> Path:
    return Path(f"data/eval/math220k_dev_{profile}.report.json")


def _bucket_name(solution_tokens: int, *, profile: str) -> str | None:
    for name, lower, upper, _quota in profile_spec(profile):
        if lower <= solution_tokens < upper:
            return name
    return None


def _to_eval_row(row: dict[str, Any], *, bucket: str, solution_tokens: int) -> dict[str, Any]:
    meta = dict(row.get("meta", {}))
    meta["token_bucket"] = bucket
    meta["solution_tokens"] = solution_tokens
    return {
        "id": str(row.get("id", "")),
        "question": str(row.get("question", "")),
        "final_answer": str(row.get("final_answer", "")),
        "meta": meta,
    }


def prepare_math220k_dev(
    *,
    profile: str,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    seed: int = 42,
    dataset_name: str = DATASET_NAME,
    split: str = "train",
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=False,
    )

    raw_rows = load_dataset_rows(dataset_name, split=split, cache_dir=cache_dir)
    normalized_rows: list[dict[str, Any]] = []
    filtered_mcq = 0
    valid_rows = 0
    for index, raw_row in enumerate(raw_rows):
        normalized = make_math220k_record(raw_row, index)
        if normalized is None:
            filtered_mcq += 1
            continue
        normalized_rows.append(normalized)
        ok, _reason = validate_sft_record(normalized)
        if ok:
            valid_rows += 1

    filter_counts = {
        "empty_question": 0,
        "empty_solution": 0,
        "empty_final_answer": 0,
        "missing_boxed": 0,
        "parse_failed": 0,
        "answer_mismatch": 0,
    }
    bucket_rows: dict[str, list[dict[str, Any]]] = {name: [] for name, *_ in profile_spec(profile)}
    for index, raw_row in enumerate(raw_rows):
        normalized = make_math220k_record(raw_row, index)
        if normalized is None:
            continue
        ok, reason = validate_sft_record(normalized)
        if not ok:
            filter_counts[reason] += 1
            continue
        solution_tokens = token_length(tokenizer, str(normalized.get("solution", "")))
        bucket = _bucket_name(solution_tokens, profile=profile)
        if bucket is None:
            continue
        bucket_rows[bucket].append(
            _to_eval_row(normalized, bucket=bucket, solution_tokens=solution_tokens)
        )

    sampled_rows: list[dict[str, Any]] = []
    pool_report: dict[str, Any] = {}
    for offset, (bucket, _lower, _upper, quota) in enumerate(profile_spec(profile)):
        sampled = sample_rows(bucket_rows[bucket], sample_size=quota, seed=seed + offset)
        sampled_rows.extend(sampled)
        pool_report[bucket] = {"available": len(bucket_rows[bucket]), "quota": quota}

    random.Random(seed).shuffle(sampled_rows)
    final_output = ensure_parent(output_path or default_output_path(profile))
    final_report = ensure_parent(report_path or default_report_path(profile))
    write_jsonl(final_output, sampled_rows)

    report = {
        "profile": profile,
        "seed": seed,
        "dataset_name": dataset_name,
        "split": split,
        "raw_rows": len(raw_rows),
        "filtered_mcq": filtered_mcq,
        "normalized_rows": len(normalized_rows),
        "valid_rows": valid_rows,
        "filters": filter_counts,
        "pools": pool_report,
        "output_size": len(sampled_rows),
    }
    _write_json(final_report, report)
    return {
        "output_path": str(final_output),
        "report_path": str(final_report),
        "report": report,
    }
