#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.datasets.conpress_sft import build_conpress_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 ConPress 判对 rollout 中构造 SFT/DFT 训练数据。")
    parser.add_argument("--parsed-root", action="append", required=True, help="包含 shard*/parsed_outputs.jsonl 的 rollout 根目录。")
    parser.add_argument("--query-pool", required=True, help="原始 query pool，用于取题目与标准答案。")
    parser.add_argument("--output", required=True, help="输出 train.jsonl 路径。")
    parser.add_argument("--source-name", default="conpress_qwen3_4b_nt_correct", help="写入 meta.source 的来源名。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_conpress_sft_dataset(
        parsed_roots=args.parsed_root,
        query_pool=args.query_pool,
        output_path=args.output,
        source_name=args.source_name,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
