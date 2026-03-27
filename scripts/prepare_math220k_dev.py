#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.math220k_dev import (
    DATASET_NAME,
    DEFAULT_TOKENIZER_NAME,
    default_output_path,
    default_report_path,
    prepare_math220k_dev,
    profile_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Math220K 生成固定 profile 的 dev 集。")
    parser.add_argument("--profile", default="main150", choices=profile_names(), help="内置 profile 名称")
    parser.add_argument("--output", help="输出 JSONL 路径")
    parser.add_argument("--report-output", help="输出 report JSON 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dataset-name", default=DATASET_NAME, help="数据集名称")
    parser.add_argument("--split", default="train", help="数据集 split")
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME, help="用于统计长度的 tokenizer")
    parser.add_argument("--cache-dir", help="datasets cache 目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_math220k_dev(
        profile=args.profile,
        output_path=args.output or default_output_path(args.profile),
        report_path=args.report_output or default_report_path(args.profile),
        seed=args.seed,
        dataset_name=args.dataset_name,
        split=args.split,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_math220k_dev={result['output_path']}")
    print(f"wrote_math220k_dev_report={result['report_path']}")


if __name__ == "__main__":
    main()
