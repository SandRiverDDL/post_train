#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.grpo_data import build_grpo_candidates, build_train_record, infer_source, load_train_dataset, sample_rows
from rl.io import write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 GRPO 训练数据工件。")
    parser.add_argument("--model-name", default="unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit", help="用于长度过滤的 tokenizer")
    parser.add_argument("--train-dataset", default="nlile/NuminaMath-1.5-RL-Verifiable", help="训练集名")
    parser.add_argument("--train-config", default=None, help="训练集 config，GSM8K 需要传 main 或 socratic")
    parser.add_argument("--train-split", default="train", help="训练集 split")
    parser.add_argument("--source", choices=("auto", "numinamath", "gsm8k"), default="auto", help="训练集来源类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-samples", type=int, default=2000, help="训练集总样本数")
    parser.add_argument("--min-response-tokens", type=int, default=64, help="最小 response token 长度")
    parser.add_argument("--max-response-tokens", type=int, default=256, help="最大 response token 长度")
    parser.add_argument("--prompt-version", choices=("v1", "v2"), default="v1", help="候选 prompt 模板版本")
    parser.add_argument("--cache-dir", default=None, help="datasets cache 目录，默认使用 HF 配置")
    parser.add_argument("--output", default="data/grpo/train_grpo_2k_short.jsonl", help="输出路径")
    return parser.parse_args(argv)

def preview_rows(rows: list[dict[str, object]]) -> None:
    print("\ngrpo train preview")
    for row in rows[:3]:
        print("=" * 80)
        print(f"id: {row['id']}")
        print(f"problem_type: {row['problem_type']}")
        print(f"response_tokens: {row['response_tokens']}")
        print(str(row["question"])[:300])
        print("--- prompt ---")
        print(str(row["prompt"]))
        print("--- reference tail ---")
        reference = str(row.get("reference_solution", ""))
        print("\n".join(reference.splitlines()[-4:]))


def main() -> None:
    args = parse_args()
    import random

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    source = infer_source(args.train_dataset, args.source)

    dataset = load_train_dataset(args.train_dataset, args.train_config, args.train_split, args.cache_dir)
    candidates = build_grpo_candidates(
        dataset,
        source=source,
        tokenizer=tokenizer,
        min_response_tokens=args.min_response_tokens,
        max_response_tokens=args.max_response_tokens,
        prompt_version=args.prompt_version,
    )

    sampled = sample_rows(candidates, rng, args.num_samples, source)
    write_jsonl(args.output, sampled)
    preview_rows(sampled)
    print("train problem_type counts:", Counter(str(row["problem_type"]) for row in sampled))
    print(f"Wrote {len(sampled)} rows to {args.output}")


if __name__ == "__main__":
    main()
