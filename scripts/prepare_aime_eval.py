#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.aime_eval import prepare_aime_eval_artifact, write_aime_eval_artifact
from post_train.data import preview_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 AIME 本地评测数据。")
    parser.add_argument("--year", type=int, required=True, choices=(24, 25), help="AIME 年份")
    parser.add_argument("--dataset-name", default=None, help="覆盖默认数据集名")
    parser.add_argument("--config-name", default=None, help="数据集 config 名")
    parser.add_argument("--split", default="test", help="数据集 split")
    parser.add_argument("--source", default=None, help="覆盖输出记录 source 字段")
    parser.add_argument("--cache-dir", default=None, help="datasets cache 目录")
    parser.add_argument("--output", default=None, help="覆盖输出 JSONL 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_aime_eval_artifact(
        year=args.year,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        split=args.split,
        source=args.source,
        cache_dir=args.cache_dir,
    )
    output_path = write_aime_eval_artifact(
        year=args.year,
        rows=rows,
        output_path=args.output,
    )
    print(f"aime{args.year}_preview")
    print(preview_rows(rows))
    print(f"year={args.year}")
    print(f"split={args.split}")
    print(f"wrote_aime={output_path}")


if __name__ == "__main__":
    main()
