#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_on_policy_data_config
from post_train.on_policy_data import prepare_on_policy_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备单轮 on-policy SFT 数据。")
    parser.add_argument("--config", required=True, help="on-policy 数据配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_on_policy_data_config(args.config)
    result = prepare_on_policy_sft_dataset(cfg)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_on_policy_queries={result['query_output_path']}")
    print(f"wrote_on_policy_raw_samples={result['raw_samples_output_path']}")
    print(f"wrote_on_policy_train={result['retained_output_path']}")
    print(f"wrote_on_policy_report={result['report_path']}")


if __name__ == "__main__":
    main()
