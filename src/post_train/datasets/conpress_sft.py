from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from post_train.answers import evaluate_prediction, extract_relaxed_final_answer
from post_train.data import summarize_sft_dataset
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.schemas import SFTRecord

QUESTION_HEADER_RE = re.compile(r"^\s*(?:\*\*)?Question\s+\d+\s*[:：]\s*(?:\*\*)?\s*", re.IGNORECASE)
TRAILING_SEPARATOR_RE = re.compile(r"\s*-{3,}\s*$")


def _iter_parsed_paths(roots: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        base = Path(root)
        paths.extend(sorted(base.glob("shard*/parsed_outputs.jsonl")))
    return paths


def _clean_solution_block(block: str) -> str:
    cleaned = QUESTION_HEADER_RE.sub("", block.strip(), count=1)
    cleaned = TRAILING_SEPARATOR_RE.sub("", cleaned).strip()
    return cleaned or block.strip()


def _load_expected_by_id(query_pool: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(query_pool)}


def build_conpress_sft_dataset(
    *,
    parsed_roots: list[str | Path],
    query_pool: str | Path,
    output_path: str | Path,
    source_name: str = "conpress_qwen3_4b_nt_correct",
) -> dict[str, Any]:
    expected_by_id = _load_expected_by_id(query_pool)
    candidates_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path_count = 0
    question_attempt_count = 0
    judged_correct_count = 0

    for path in _iter_parsed_paths(parsed_roots):
        path_count += 1
        for pack in read_jsonl(path):
            for question in pack.get("questions", []):
                question_id = str(question.get("id", ""))
                if question_id not in expected_by_id:
                    continue
                question_attempt_count += 1
                expected_answer = str(expected_by_id[question_id].get("final_answer", ""))
                block = str(question.get("block", ""))
                judgment = evaluate_prediction(block, expected_answer, require_boxed=True)
                if not judgment["correct"]:
                    continue
                judged_correct_count += 1
                solution = _clean_solution_block(block)
                final_answer = extract_relaxed_final_answer(solution) or expected_answer
                candidates_by_id[question_id].append(
                    {
                        "id": question_id,
                        "question": expected_by_id[question_id]["question"],
                        "solution": solution,
                        "final_answer": final_answer,
                        "meta": {
                            **dict(expected_by_id[question_id].get("meta", {})),
                            "source": source_name,
                            "teacher_rollout_path": str(path),
                            "pack_id": pack.get("pack_id"),
                            "sample_index": pack.get("sample_index"),
                            "question_index": question.get("question_index"),
                            "output_tokens": question.get("output_tokens"),
                            "parsed_answer": judgment.get("parsed_answer", ""),
                        },
                    }
                )

    rows: list[dict[str, Any]] = []
    for question_id in sorted(candidates_by_id, key=lambda value: int(value) if value.isdigit() else value):
        candidates = candidates_by_id[question_id]
        # 同题多条正确轨迹时选最短，减少把冗长推理蒸进 student。
        selected = min(candidates, key=lambda row: (len(str(row["solution"])), str(row["meta"].get("teacher_rollout_path"))))
        rows.append(SFTRecord(**selected).model_dump())

    write_jsonl(output_path, rows)
    output_path = Path(output_path)

    level_counts = Counter(str(row.get("meta", {}).get("level", "unknown")) for row in rows)
    type_counts = Counter(str(row.get("meta", {}).get("type", "unknown")) for row in rows)
    solution_lengths = [len(str(row["solution"])) for row in rows]
    report = {
        "output_path": str(output_path),
        "query_pool": str(query_pool),
        "parsed_roots": [str(root) for root in parsed_roots],
        "parsed_file_count": path_count,
        "question_attempt_count": question_attempt_count,
        "judged_correct_attempt_count": judged_correct_count,
        "unique_correct_prompt_count": len(rows),
        "multi_correct_prompt_count": sum(1 for values in candidates_by_id.values() if len(values) > 1),
        "dataset_summary": summarize_sft_dataset(rows),
        "level_counts": dict(sorted(level_counts.items())),
        "type_counts": dict(sorted(type_counts.items())),
        "solution_char_length": {
            "avg": mean(solution_lengths) if solution_lengths else 0.0,
            "max": max(solution_lengths) if solution_lengths else 0,
        },
    }
    report_path = ensure_parent(output_path.with_suffix(output_path.suffix + ".report.json"))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
