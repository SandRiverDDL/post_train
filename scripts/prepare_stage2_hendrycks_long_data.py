#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage2_hendrycks_long_data_config
from post_train.datasets.stage2_hendrycks_long import prepare_stage2_hendrycks_long_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 hendrycks_math + long CoT 的 stage2 SFT 数据。")
    parser.add_argument("--config", required=True, help="stage2 hendrycks-long 数据配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage2_hendrycks_long_data_config(args.config)
    result = prepare_stage2_hendrycks_long_dataset(cfg)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_stage2_hendrycks_long_train={result['output_path']}")
    print(f"wrote_stage2_hendrycks_long_report={result['report_path']}")
    print(f"wrote_stage2_hendrycks_long_unmatched={result['unmatched_preview_path']}")


if __name__ == "__main__":
    main()
