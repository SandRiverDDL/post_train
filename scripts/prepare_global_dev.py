#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.data import prepare_sampled_eval_artifact, preview_rows
from post_train.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从评测数据集中随机抽样生成全局 dev 集。")
    parser.add_argument("--dataset-name", default="HuggingFaceH4/MATH-500", help="数据集名")
    parser.add_argument("--config-name", default=None, help="数据集 config 名")
    parser.add_argument("--split", default="test", help="数据集 split")
    parser.add_argument("--source", default="math500", help="输出记录 source 字段")
    parser.add_argument("--sample-size", type=int, default=150, help="随机抽样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cache-dir", default=None, help="datasets cache 目录")
    parser.add_argument(
        "--output",
        default="data/eval/global_dev_math500_150.jsonl",
        help="输出 JSONL 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_sampled_eval_artifact(
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        split=args.split,
        source=args.source,
        sample_size=args.sample_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    write_jsonl(args.output, rows)
    print("global_dev_preview")
    print(preview_rows(rows))
    print(f"dataset={args.dataset_name}")
    print(f"split={args.split}")
    print(f"sample_size={args.sample_size}")
    print(f"seed={args.seed}")
    print(f"wrote_global_dev={args.output}")


if __name__ == "__main__":
    main()
