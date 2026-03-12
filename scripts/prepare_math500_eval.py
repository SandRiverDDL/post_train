#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.data import to_eval_record
from rl.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成固定的 Math500 评测集。")
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500", help="Hugging Face 数据集名")
    parser.add_argument("--split", default="test", help="使用的数据集 split")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dev-size", type=int, default=200, help="开发集大小")
    parser.add_argument("--dev-output", default="data/eval/math500_dev200.jsonl", help="dev 输出路径")
    parser.add_argument("--test-output", default="data/eval/math500_test300.jsonl", help="test 输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    total = len(dataset)
    if total < args.dev_size:
        raise ValueError(f"数据集样本不足：{total} < {args.dev_size}")

    indices = list(range(total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    dev_indices = set(indices[: args.dev_size])
    dev_rows = []
    test_rows = []

    for index in range(total):
        row = to_eval_record(dict(dataset[index]), index, "math500")
        if index in dev_indices:
            dev_rows.append(row)
        else:
            test_rows.append(row)

    write_jsonl(args.dev_output, dev_rows)
    write_jsonl(args.test_output, test_rows)
    print(f"Wrote {len(dev_rows)} rows to {args.dev_output}")
    print(f"Wrote {len(test_rows)} rows to {args.test_output}")


if __name__ == "__main__":
    main()
