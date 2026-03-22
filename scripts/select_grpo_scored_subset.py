#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.grpo_data import select_rows_by_budget
from rl.io import ensure_parent, read_jsonl, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 scored GRPO 候选集中筛选训练子集。")
    parser.add_argument(
        "--input-scored",
        default="data/grpo/train_grpo_gsm8k_3857_short_scored.jsonl",
        help="全量 scored 候选集",
    )
    parser.add_argument(
        "--output-filtered",
        default="data/grpo/train_grpo_gsm8k_3857_short_middiff.jsonl",
        help="满足阈值的全量中等难度子集",
    )
    parser.add_argument("--output-selected", default=None, help="最终训练子集输出路径；仅在 target-size 非空时使用")
    parser.add_argument("--target-size", type=int, default=None, help="最终训练子集大小；不传时输出全部符合要求样本")
    parser.add_argument("--min-correct-rate", type=float, default=0.25, help="最小正确率")
    parser.add_argument("--max-correct-rate", type=float, default=0.50, help="最大正确率")
    parser.add_argument("--min-parse-rate", type=float, default=0.50, help="最小解析成功率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--stratify-by", default="problem_type", help="分层字段；传空串表示不分层")
    return parser.parse_args(argv)


def passes_difficulty_filter(
    row: dict[str, Any],
    *,
    min_correct_rate: float,
    max_correct_rate: float,
    min_parse_rate: float,
) -> bool:
    correct_rate = float(row.get("sft_correct_rate", -1.0))
    parse_rate = float(row.get("sft_parse_rate", -1.0))
    return min_correct_rate <= correct_rate <= max_correct_rate and parse_rate >= min_parse_rate

def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_scored)
    filtered = [
        row
        for row in rows
        if passes_difficulty_filter(
            row,
            min_correct_rate=args.min_correct_rate,
            max_correct_rate=args.max_correct_rate,
            min_parse_rate=args.min_parse_rate,
        )
    ]
    filtered_path = ensure_parent(args.output_filtered)
    write_jsonl(filtered_path, filtered)
    print(f"输入 scored 样本数: {len(rows)}")
    print(f"中等难度样本数: {len(filtered)}")
    print(f"输出 filtered 工件: {filtered_path}")

    if args.target_size is None:
        return
    if not args.output_selected:
        raise ValueError("传入 target-size 时必须同时提供 output-selected。")

    stratify_by = args.stratify_by.strip() or None
    selected = select_rows_by_budget(
        filtered,
        target_size=args.target_size,
        stratify_by=stratify_by,
        seed=args.seed,
    )
    selected_path = ensure_parent(args.output_selected)
    write_jsonl(selected_path, selected)
    print(f"抽样输出样本数: {len(selected)}")
    print(f"输出 selected 工件: {selected_path}")


if __name__ == "__main__":
    main()
