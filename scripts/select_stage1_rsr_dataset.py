#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage1_rsr_select_config
from post_train.rsr_data import select_stage1_rsr_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 stage1 RSR 候选池中筛选最终 SFT 数据集。")
    parser.add_argument("--config", required=True, help="stage1 RSR 二段筛选配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage1_rsr_select_config(args.config)
    result = select_stage1_rsr_dataset(cfg)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_stage1_rsr_selected_train={result['output_path']}")
    print(f"wrote_stage1_rsr_selected_report={result['report_path']}")


if __name__ == "__main__":
    main()
