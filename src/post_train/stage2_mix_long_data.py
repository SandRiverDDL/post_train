from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from post_train.config import Stage2MixLongDataConfig
from post_train.data import load_dataset_rows, make_sft_record, summarize_sft_dataset
from post_train.io import ensure_parent, write_jsonl


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


def prepare_stage2_mix_long_dataset(cfg: Stage2MixLongDataConfig) -> dict[str, Any]:
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
    kept_rows: list[dict[str, Any]] = []
    raw_solution_lengths: list[int] = []
    kept_solution_lengths: list[int] = []
    filtered_too_long = 0

    for index, row in enumerate(raw_rows):
        record = make_sft_record(row, index, source=cfg.dataset_name)
        record.setdefault("meta", {})
        record["meta"]["source_dataset"] = cfg.dataset_name
        solution = str(record.get("solution", "") or "")
        solution_tokens = len(tokenizer(solution, add_special_tokens=False)["input_ids"])
        raw_solution_lengths.append(solution_tokens)
        if solution_tokens > cfg.max_solution_tokens:
            filtered_too_long += 1
            continue
        record["meta"]["solution_tokens"] = solution_tokens
        kept_solution_lengths.append(solution_tokens)
        kept_rows.append(record)

    filtered_kept_rows = list(kept_rows)
    if cfg.sample_size is not None:
        if cfg.sample_size > len(filtered_kept_rows):
            raise ValueError(
                f"过滤后样本不足：请求 sample_size={cfg.sample_size}，实际只有 {len(filtered_kept_rows)} 条。"
            )
        sampled_rows = list(filtered_kept_rows)
        random.Random(cfg.seed).shuffle(sampled_rows)
        filtered_kept_rows = sampled_rows[: cfg.sample_size]

    report = {
        "config": {
            "dataset_name": cfg.dataset_name,
            "dataset_split": cfg.dataset_split,
            "seed": cfg.seed,
            "max_solution_tokens": cfg.max_solution_tokens,
            "sample_size": cfg.sample_size,
        },
        "source": {
            "raw_rows": len(raw_rows),
            "kept_rows": len(kept_rows),
            "sampled_rows": len(filtered_kept_rows),
        },
        "filters": {
            "filtered_too_long_solution": filtered_too_long,
        },
        "solution_tokens": {
            "raw": _length_stats(raw_solution_lengths),
            "kept": _length_stats(kept_solution_lengths),
        },
        "dataset_summary": summarize_sft_dataset(filtered_kept_rows),
    }

    output_path = write_jsonl(cfg.output_path, filtered_kept_rows)
    report_path = _write_json(cfg.report_path, report)
    return {
        "output_path": output_path,
        "report_path": report_path,
        "report": report,
    }
