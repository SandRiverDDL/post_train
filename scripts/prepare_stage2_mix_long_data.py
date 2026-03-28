#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage2_mix_long_data_config
from post_train.stage2_mix_long_data import prepare_stage2_mix_long_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 Mix-Long SFT 数据。")
    parser.add_argument("--config", required=True, help="Mix-Long 数据配置文件路径")
    parser.add_argument("--sample-size", type=int, default=None, help="过滤后随机抽样的样本数")
    parser.add_argument("--seed", type=int, default=None, help="覆盖抽样随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage2_mix_long_data_config(args.config)
    if args.sample_size is not None:
        cfg.sample_size = args.sample_size
    if args.seed is not None:
        cfg.seed = args.seed
    result = prepare_stage2_mix_long_dataset(cfg)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_mix_long_train={result['output_path']}")
    print(f"wrote_mix_long_report={result['report_path']}")


if __name__ == "__main__":
    main()
