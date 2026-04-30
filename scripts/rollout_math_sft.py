#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.rollout.math_sft import DEFAULT_MATH_DATASET, DEFAULT_MIX_LONG_PATH, prepare_math_rollout_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 MATH 抽题，用 vLLM rollout，并拒绝采样构造 SFT 数据。")
    parser.add_argument("--model", required=True, help="HF 模型 ID 或本地模型路径，例如 justrl 模型。")
    parser.add_argument("--output-dir", default="data/rollout/math_sft/justrl_math1500", help="输出目录。")
    parser.add_argument("--dataset", default=DEFAULT_MATH_DATASET, help="MATH 数据集名；EleutherAI/hendrycks_math 会自动合并 7 个 config。")
    parser.add_argument("--split", default="train", help="MATH split。")
    parser.add_argument("--sample-size", type=int, default=1500, help="抽取 prompt 数量。")
    parser.add_argument("--responses-per-prompt", type=int, default=8, help="每题 rollout 轨迹数。")
    parser.add_argument("--retained-count", type=int, default=1000, help="拒绝采样保留的正确轨迹数量。")
    parser.add_argument("--mix-long-sample-size", type=int, default=1000, help="从 mix_long 随机抽取的 SFT 样本数。")
    parser.add_argument("--mix-long-path", default=str(DEFAULT_MIX_LONG_PATH), help="mix_long SFT JSONL 路径。")
    parser.add_argument("--levels", type=int, nargs="*", default=None, help="可选难度过滤，例如 --levels 3 4 5。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache dir。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_math_rollout_sft_dataset(
        model=args.model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        responses_per_prompt=args.responses_per_prompt,
        retained_count=args.retained_count,
        mix_long_sample_size=args.mix_long_sample_size,
        mix_long_path=args.mix_long_path,
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
    )
    print(json.dumps(result["outputs"], ensure_ascii=False, indent=2))
    print(json.dumps(result["combined"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
