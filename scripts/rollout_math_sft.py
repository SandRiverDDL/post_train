#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.rollout.math_sft import (
    DEFAULT_MATH_DATASET,
    merge_math_rollout_sft_dataset,
    prepare_math_rollout_sft_dataset,
)


def parse_sample_size(value: str) -> int | None:
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--sample-size 必须是正整数或 all。")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 MATH 抽题，用 vLLM rollout，并可合并 shard raw。")
    parser.add_argument("--mode", choices=("full", "shard", "merge"), default="full", help="full=单进程 raw rollout；shard=只生成当前分片 raw；merge=合并分片 raw。")
    parser.add_argument("--model", required=True, help="HF 模型 ID 或本地模型路径，例如 justrl 模型。")
    parser.add_argument("--output-dir", default="data/rollout/math_sft/justrl_math1500", help="输出目录。")
    parser.add_argument("--dataset", default=DEFAULT_MATH_DATASET, help="MATH 数据集名；EleutherAI/hendrycks_math 会自动合并 7 个 config。")
    parser.add_argument("--split", default="train", help="MATH split。")
    parser.add_argument("--sample-size", type=parse_sample_size, default=1500, help="抽取 prompt 数量；传 all 表示使用过滤后的全量。")
    parser.add_argument("--responses-per-prompt", type=int, default=8, help="每题 rollout 轨迹数。")
    parser.add_argument("--levels", type=int, nargs="*", default=None, help="可选难度过滤，例如 --levels 3 4 5。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache dir。")
    parser.add_argument("--num-shards", type=int, default=1, help="分片总数，仅 shard 模式使用。")
    parser.add_argument("--shard-index", type=int, default=None, help="当前分片编号，从 0 开始，仅 shard 模式使用。")
    parser.add_argument("--raw-shards", nargs="*", default=None, help="merge 模式输入的 raw shard jsonl 列表。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "merge":
        if not args.raw_shards:
            raise ValueError("merge 模式必须传 --raw-shards。")
        result = merge_math_rollout_sft_dataset(
            model=args.model,
            output_dir=args.output_dir,
            raw_shards=args.raw_shards,
        )
        print(json.dumps(result["outputs"], ensure_ascii=False, indent=2))
        return

    if args.mode == "shard" and args.shard_index is None:
        raise ValueError("shard 模式必须传 --shard-index。")

    result = prepare_math_rollout_sft_dataset(
        model=args.model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        responses_per_prompt=args.responses_per_prompt,
        dataset_name=args.dataset,
        split=args.split,
        levels=args.levels,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cache_dir=args.cache_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        rollout_only=args.mode == "shard",
    )
    print(json.dumps(result["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
