#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_simpo_data_config
from post_train.simpo_data import prepare_simpo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 SIMPO preference 数据。")
    parser.add_argument("--config", required=True, help="SIMPO 数据配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simpo_data_config(args.config)
    result = prepare_simpo_dataset(cfg)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_simpo_queries={result['query_output_path']}")
    print(f"wrote_simpo_raw_samples={result['raw_samples_output_path']}")
    print(f"wrote_simpo_pairs={result['pair_output_path']}")
    print(f"wrote_simpo_pilot_pairs={result['pilot_output_path']}")
    print(f"wrote_simpo_report={result['report_path']}")


if __name__ == "__main__":
    main()
