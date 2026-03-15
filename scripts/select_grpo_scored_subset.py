#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.io import ensure_parent, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--output-selected", default=None, help="最终训练子集输出路径")
    parser.add_argument("--target-size", type=int, default=None, help="最终训练子集大小")
    parser.add_argument("--min-correct-rate", type=float, default=0.25, help="最小正确率")
    parser.add_argument("--max-correct-rate", type=float, default=0.50, help="最大正确率")
    parser.add_argument("--min-parse-rate", type=float, default=0.50, help="最小解析成功率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--stratify-by", default="problem_type", help="分层字段；传空串表示不分层")
    return parser.parse_args()


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


def _largest_remainder_sample(
    rows: list[dict[str, Any]],
    *,
    target_size: int,
    stratify_by: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    if target_size > len(rows):
        raise ValueError(f"筛后样本不足：{len(rows)} < {target_size}")
    if not stratify_by:
        rng = random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:target_size]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(stratify_by, "Other"))].append(row)

    total = len(rows)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for group_name, group_rows in groups.items():
        raw = target_size * len(group_rows) / total
        base = min(len(group_rows), int(raw))
        allocations[group_name] = base
        remainders.append((raw - base, group_name))

    assigned = sum(allocations.values())
    for _, group_name in sorted(remainders, reverse=True):
        if assigned >= target_size:
            break
        if allocations[group_name] >= len(groups[group_name]):
            continue
        allocations[group_name] += 1
        assigned += 1

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group_name in sorted(groups):
        group_rows = list(groups[group_name])
        rng.shuffle(group_rows)
        selected.extend(group_rows[: allocations[group_name]])

    if len(selected) < target_size:
        remaining = [row for row in rows if row not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_size - len(selected)])
    return selected[:target_size]


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

    if args.output_selected is None and args.target_size is None:
        return
    if not args.output_selected or args.target_size is None:
        raise ValueError("output-selected 和 target-size 必须同时提供")

    stratify_by = args.stratify_by.strip() or None
    selected = _largest_remainder_sample(
        filtered,
        target_size=args.target_size,
        stratify_by=stratify_by,
        seed=args.seed,
    )
    selected_path = ensure_parent(args.output_selected)
    write_jsonl(selected_path, selected)
    print(f"输出 selected 工件: {selected_path}")


if __name__ == "__main__":
    main()
