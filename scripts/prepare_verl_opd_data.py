#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.verl_opd import (  # noqa: E402
    DEFAULT_CANDIDATE_SOURCE,
    DEFAULT_DAPO_SOURCE,
    DEFAULT_MATH_LEVEL_OUTPUT_DIR,
    DEFAULT_MIX_OUTPUT_DIR,
    DEFAULT_QWEN3_1P7B_TOKENIZER,
    DEFAULT_SMOKE_OUTPUT_DIR,
    build_math_level_dataset,
    build_mix_dataset,
    build_smoke_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 verl 标准 OPD parquet 数据。")
    parser.add_argument("--mode", choices=("smoke", "mix", "math-levels"), required=True)
    parser.add_argument("--candidate-source", default=str(DEFAULT_CANDIDATE_SOURCE))
    parser.add_argument("--dapo-source", default=str(DEFAULT_DAPO_SOURCE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=128, help="smoke 模式的 DAPO 抽样数量。")
    parser.add_argument("--candidate-size", type=int, default=200, help="mix 模式的 candidate 数量。")
    parser.add_argument("--dapo-size", type=int, default=800, help="mix 模式的 DAPO 数量。")
    parser.add_argument("--dataset", default="EleutherAI/hendrycks_math", help="math-levels 模式的数据集名。")
    parser.add_argument("--split", default="train", help="math-levels 模式的数据 split。")
    parser.add_argument("--levels", type=int, nargs="*", default=[3, 4], help="math-levels 模式的 MATH level。")
    parser.add_argument("--max-prompt-tokens", type=int, default=256, help="math-levels 模式的 chat-template prompt token 上限。")
    parser.add_argument("--tokenizer-name", default=str(DEFAULT_QWEN3_1P7B_TOKENIZER), help="用于计算 prompt token 的 tokenizer。")
    parser.add_argument("--cache-dir", default="/mnt/dataY/fsw/cache/huggingface", help="HF datasets cache dir。")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        result = build_smoke_dataset(
            dapo_source=args.dapo_source,
            output_dir=args.output_dir or DEFAULT_SMOKE_OUTPUT_DIR,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    elif args.mode == "mix":
        result = build_mix_dataset(
            candidate_source=args.candidate_source,
            dapo_source=args.dapo_source,
            output_dir=args.output_dir or DEFAULT_MIX_OUTPUT_DIR,
            candidate_size=args.candidate_size,
            dapo_size=args.dapo_size,
            seed=args.seed,
        )
    else:
        result = build_math_level_dataset(
            output_dir=args.output_dir or DEFAULT_MATH_LEVEL_OUTPUT_DIR,
            dataset_name=args.dataset,
            split=args.split,
            levels=args.levels,
            sample_size=args.sample_size,
            seed=args.seed,
            max_prompt_tokens=args.max_prompt_tokens,
            tokenizer_name=args.tokenizer_name,
            chat_template_enable_thinking=False,
            cache_dir=args.cache_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
