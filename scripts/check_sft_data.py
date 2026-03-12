#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.answers import are_equivalent, extract_final_answer, has_boxed_final_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 SFT 训练数据质量。")
    parser.add_argument("--input", default="data/train_sft.jsonl", help="待检查的 JSONL 文件")
    parser.add_argument("--show-failures", type=int, default=10, help="打印多少条失败样本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    boxed = sum(has_boxed_final_answer(row.get("solution", "")) for row in rows)
    empty_final = sum(not (row.get("final_answer") or "").strip() for row in rows)
    parsed = 0
    consistent = 0
    by_type = Counter(row.get("problem_type", "Unknown") for row in rows)
    empty_by_type = Counter(
        row.get("problem_type", "Unknown") for row in rows if not (row.get("final_answer") or "").strip()
    )
    parse_fail_by_type = Counter()
    mismatch_by_type = Counter()

    for row in rows:
        solution = row.get("solution", "")
        final_answer = (row.get("final_answer") or "").strip()
        parsed_answer = extract_final_answer(solution)
        if parsed_answer:
            parsed += 1
        else:
            parse_fail_by_type[row.get("problem_type", "Unknown")] += 1
        if parsed_answer and final_answer and are_equivalent(parsed_answer, final_answer):
            consistent += 1
        elif final_answer:
            mismatch_by_type[row.get("problem_type", "Unknown")] += 1

    print(f"input={path}")
    print(f"total={total}")
    print(f"boxed={boxed}")
    print(f"boxed_rate={boxed / total:.4f}" if total else "boxed_rate=0.0000")
    print(f"parse_success={parsed}")
    print(f"parse_success_rate={parsed / total:.4f}" if total else "parse_success_rate=0.0000")
    print(f"consistent_final_answer={consistent}")
    print(f"consistent_rate={consistent / total:.4f}" if total else "consistent_rate=0.0000")
    print(f"empty_final_answer={empty_final}")
    print(f"empty_rate={empty_final / total:.4f}" if total else "empty_rate=0.0000")
    print("problem_type_counts=", dict(by_type))
    print("empty_by_problem_type=", dict(empty_by_type))
    print("parse_fail_by_problem_type=", dict(parse_fail_by_type))
    print("mismatch_by_problem_type=", dict(mismatch_by_type))

    shown = 0
    print("\nsample_failures")
    for row in rows:
        solution = row.get("solution", "")
        final_answer = (row.get("final_answer") or "").strip()
        parsed_answer = extract_final_answer(solution)
        if final_answer and parsed_answer and are_equivalent(parsed_answer, final_answer):
            continue
        print("=" * 80)
        print(f"id: {row.get('id')}")
        print(f"problem_type: {row.get('problem_type', 'Unknown')}")
        print(row.get("question", "")[:300])
        print("--- solution tail ---")
        print("\n".join(solution.splitlines()[-6:]))
        print("--- final_answer ---")
        print(final_answer)
        print("--- parsed_answer ---")
        print(parsed_answer or "")
        shown += 1
        if shown >= args.show_failures:
            break


if __name__ == "__main__":
    main()
