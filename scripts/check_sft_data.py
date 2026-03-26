#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.answers import are_equivalent, extract_final_answer
from post_train.data import summarize_sft_dataset
from post_train.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 SFT 训练数据质量。")
    parser.add_argument("--input", required=True, help="待检查的 JSONL 文件")
    parser.add_argument("--show-failures", type=int, default=10, help="打印多少条失败样本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    summary = summarize_sft_dataset(rows)

    print(f"input={args.input}")
    for key in [
        "total",
        "boxed",
        "boxed_rate",
        "parse_success",
        "parse_success_rate",
        "consistent_final_answer",
        "consistent_rate",
        "empty_final_answer",
        "empty_rate",
    ]:
        print(f"{key}={summary[key]}")

    shown = 0
    print("\nsample_failures")
    for row in rows:
        solution = str(row.get("solution", ""))
        final_answer = str(row.get("final_answer", "")).strip()
        parsed_answer = extract_final_answer(solution)
        if final_answer and parsed_answer and are_equivalent(parsed_answer, final_answer):
            continue
        print("=" * 80)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        shown += 1
        if shown >= args.show_failures:
            break


if __name__ == "__main__":
    main()
