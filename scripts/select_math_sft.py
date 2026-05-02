#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.rollout.math_sft import DEFAULT_MIX_LONG_PATH, select_math_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从已生成的 MATH rollout raw 中选择 SFT 数据。")
    parser.add_argument("--model", required=True, help="生成 raw samples 的模型 ID 或本地路径。")
    parser.add_argument("--raw-samples", required=True, help="merged raw samples JSONL。")
    parser.add_argument("--output-dir", required=True, help="rollout 输出目录。")
    parser.add_argument("--retained-count", type=int, default=1000, help="保留的正确 rollout 轨迹数量。")
    parser.add_argument("--mix-long-sample-size", type=int, default=1000, help="从 mix_long 随机抽取的 SFT 样本数。")
    parser.add_argument("--mix-long-path", default=str(DEFAULT_MIX_LONG_PATH), help="mix_long SFT JSONL 路径。")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = select_math_sft_dataset(
        model=args.model,
        output_dir=args.output_dir,
        raw_samples_path=args.raw_samples,
        retained_count=args.retained_count,
        mix_long_sample_size=args.mix_long_sample_size,
        mix_long_path=args.mix_long_path,
        seed=args.seed,
    )
    print(json.dumps(result["outputs"], ensure_ascii=False, indent=2))
    print(json.dumps(result["sft_selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
