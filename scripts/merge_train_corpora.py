#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.data import normalize_question_text
from rl.io import ensure_parent, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 manifest 合并训练集并去重。")
    parser.add_argument("--manifest", required=True, help="manifest YAML 路径")
    return parser.parse_args()


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _read_normalized_questions(paths: list[str]) -> set[str]:
    normalized: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            question = str(row.get("question") or row.get("prompt") or "").strip()
            if question:
                normalized.add(normalize_question_text(question))
    return normalized


def _sort_inputs(inputs: list[dict]) -> list[dict]:
    return sorted(inputs, key=lambda item: int(item.get("priority", 0)), reverse=True)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    inputs = _sort_inputs(list(manifest.get("inputs", [])))
    dedup_against = [str(path) for path in manifest.get("dedup_against", [])]
    output_value = manifest.get("merge_output") or manifest.get("output")
    if not output_value:
        raise ValueError("manifest 必须提供 merge_output。")
    output_path = ensure_parent(output_value)
    summary_path = ensure_parent(
        manifest.get("merge_summary_output")
        or manifest.get("summary_output")
        or output_path.with_name(f"{output_path.stem}_summary.json")
    )
    conflicts_path = ensure_parent(manifest.get("conflicts_output") or output_path.with_name(f"{output_path.stem}_conflicts.jsonl"))

    blocked_questions = _read_normalized_questions(dedup_against)
    merged_rows: list[dict] = []
    conflicts: list[dict] = []
    seen_questions: dict[str, dict] = {}
    dedup_against_hits = 0
    per_input_counts: dict[str, dict[str, int]] = {}

    for item in inputs:
        name = str(item["name"])
        path = str(item["path"])
        rows = read_jsonl(path)
        stats = {"input_count": len(rows), "kept_count": 0, "dedup_against_dropped": 0, "duplicate_dropped": 0}
        for row in rows:
            question = str(row.get("question") or row.get("prompt") or "").strip()
            normalized_question = normalize_question_text(question)
            if normalized_question in blocked_questions:
                dedup_against_hits += 1
                stats["dedup_against_dropped"] += 1
                continue
            existing = seen_questions.get(normalized_question)
            if existing is not None:
                stats["duplicate_dropped"] += 1
                if str(existing.get("final_answer", "")).strip() != str(row.get("final_answer", "")).strip():
                    conflicts.append(
                        {
                            "question": question,
                            "kept_id": existing.get("id"),
                            "kept_answer": existing.get("final_answer"),
                            "dropped_id": row.get("id"),
                            "dropped_answer": row.get("final_answer"),
                        }
                    )
                continue
            seen_questions[normalized_question] = row
            merged_rows.append(row)
            stats["kept_count"] += 1
        per_input_counts[name] = stats

    write_jsonl(output_path, merged_rows)
    write_jsonl(conflicts_path, conflicts)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "merged_count": len(merged_rows),
                "dedup_against_hits": dedup_against_hits,
                "conflict_count": len(conflicts),
                "inputs": per_input_counts,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(merged_rows)} rows to {output_path}")
    print(f"Wrote {len(conflicts)} conflicts to {conflicts_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
