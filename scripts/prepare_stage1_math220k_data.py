#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage1_math220k_data_config
from post_train.stage1_math220k_data import prepare_stage1_math220k_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 stage1 Math220K 对照实验数据。")
    parser.add_argument("--config", required=True, help="stage1 Math220K 数据配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage1_math220k_data_config(args.config)
    result = prepare_stage1_math220k_dataset(cfg)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_stage1_math220k_train={result['output_path']}")
    print(f"wrote_stage1_math220k_report={result['report_path']}")


if __name__ == "__main__":
    main()
