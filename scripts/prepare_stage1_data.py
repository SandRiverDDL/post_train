#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.data import prepare_stage1_artifacts, preview_rows
from post_train.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 stage1 SFT 训练与冻结 dev 数据。")
    parser.add_argument("--stage1-dataset", default="UWNSL/MATH_training_split_short_cot", help="stage1 数据集名")
    parser.add_argument("--stage1-split", default="train", help="stage1 数据集 split")
    parser.add_argument("--train-size", type=int, default=2000, help="stage1 训练样本数")
    parser.add_argument("--dev-size", type=int, default=200, help="冻结 dev 样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cache-dir", default=None, help="datasets cache 目录")
    parser.add_argument("--train-output", default="data/stage1_train.jsonl", help="stage1 训练集输出")
    parser.add_argument("--dev-output", default="data/stage1_dev200.jsonl", help="stage1 dev 输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows, dev_rows = prepare_stage1_artifacts(
        dataset_name=args.stage1_dataset,
        split=args.stage1_split,
        train_size=args.train_size,
        dev_size=args.dev_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)
    print("stage1_train_preview")
    print(preview_rows(train_rows))
    print(f"wrote_train={args.train_output}")
    print("stage1_dev_preview")
    print(preview_rows(dev_rows))
    print(f"wrote_dev={args.dev_output}")


if __name__ == "__main__":
    main()
