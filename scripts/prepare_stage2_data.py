#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage2_data_config
from post_train.stage2_data import prepare_stage2_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 stage2 SFT 的清洗与筛选数据。")
    parser.add_argument("--config", required=True, help="stage2 数据配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage2_data_config(args.config)
    result = prepare_stage2_dataset(cfg)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_stage2_train={result['output_path']}")
    print(f"wrote_stage2_report={result['report_path']}")


if __name__ == "__main__":
    main()
