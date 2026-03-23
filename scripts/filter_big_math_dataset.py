#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.answers import ensure_boxed_final_answer
from rl.data import (
    BIG_MATH_ALLOWED_SOURCES,
    BIG_MATH_EXCLUDED_SOURCES,
    canonical_problem_type,
    count_latex_markers,
    domain_paths_text,
    has_allowed_big_math_domain,
    has_excluded_big_math_domain,
    is_allowed_short_answer,
    looks_like_proof_prompt,
    select_middle_by_solve_rate,
)
from rl.io import ensure_parent, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="过滤 Big-Math quintile_2 候选集。")
    parser.add_argument("--dataset", default="open-r1/Big-Math-RL-Verified-Processed", help="HF 数据集名")
    parser.add_argument("--config-name", default="quintile_2", help="HF config 名")
    parser.add_argument("--split", default="train", help="数据集 split")
    parser.add_argument("--cache-dir", default=None, help="datasets cache 目录")
    parser.add_argument("--output", default="data/grpo/big_math_quintile2_filtered.jsonl", help="输出 JSONL")
    parser.add_argument("--summary-output", default=None, help="过滤摘要 JSON 路径")
    return parser.parse_args()


def default_summary_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.stem}_summary.json")


def build_big_math_record(row: dict[str, object], index: int, *, dataset_name: str, dataset_config: str) -> dict[str, object]:
    question = str(row.get("prompt", "")).strip()
    final_answer = str(row.get("solution", "")).strip()
    domain = row.get("domain", [])
    domain_text = domain_paths_text(domain)
    return {
        "id": f"big_math:{dataset_config}:{index}",
        "question": question,
        "final_answer": final_answer,
        "solution": ensure_boxed_final_answer(final_answer, final_answer),
        "source": str(row.get("source", "")).strip(),
        "problem_type": canonical_problem_type(domain_text),
        "domain": domain,
        "solve_rate": float(row.get("llama8b_solve_rate", 0.0)),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
    }


def filter_big_math_rows(rows: list[dict[str, object]], *, dataset_name: str, dataset_config: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    kept: list[dict[str, object]] = []
    dropped = Counter()

    for index, row in enumerate(rows):
        source = str(row.get("source", "")).strip()
        if source in BIG_MATH_EXCLUDED_SOURCES or source not in BIG_MATH_ALLOWED_SOURCES:
            dropped["source_excluded"] += 1
            continue

        domain = row.get("domain", [])
        if has_excluded_big_math_domain(domain):
            dropped["domain_excluded"] += 1
            continue
        if not has_allowed_big_math_domain(domain):
            dropped["domain_not_allowed"] += 1
            continue

        question = str(row.get("prompt", "")).strip()
        if not question or len(question) > 650:
            dropped["question_too_long_or_empty"] += 1
            continue
        if count_latex_markers(question) > 6:
            dropped["too_many_latex_markers"] += 1
            continue
        if looks_like_proof_prompt(question):
            dropped["proof_like_prompt"] += 1
            continue

        final_answer = str(row.get("solution", "")).strip()
        if not is_allowed_short_answer(final_answer):
            dropped["answer_shape_excluded"] += 1
            continue

        kept.append(build_big_math_record(row, index, dataset_name=dataset_name, dataset_config=dataset_config))

    return kept, dict(dropped)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.config_name, split=args.split, cache_dir=args.cache_dir)
    rows = [dict(dataset[index]) for index in range(len(dataset))]
    filtered_rows, dropped_by_reason = filter_big_math_rows(rows, dataset_name=args.dataset, dataset_config=args.config_name)
    selected_rows = select_middle_by_solve_rate(filtered_rows, rate_key="solve_rate")

    write_jsonl(args.output, selected_rows)
    summary_path = ensure_parent(args.summary_output or default_summary_path(args.output))
    summary = {
        "dataset_name": args.dataset,
        "dataset_config": args.config_name,
        "input_count": len(rows),
        "hard_filtered_count": len(filtered_rows),
        "kept_count": len(selected_rows),
        "dropped_by_reason": dropped_by_reason,
        "source_counts": dict(Counter(str(row["source"]) for row in selected_rows)),
        "problem_type_counts": dict(Counter(str(row["problem_type"]) for row in selected_rows)),
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(selected_rows)} rows to {args.output}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
