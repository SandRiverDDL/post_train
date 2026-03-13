#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rl.answers import ensure_boxed_final_answer
from rl.data import GSM8K_ANSWER_RE, clean_completion_for_protocol, make_record, to_sft_record
from rl.grpo import build_grpo_record, response_token_length
from rl.io import write_jsonl
from prepare_data import sample_numinamath


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--cache-dir", default=None, help="datasets cache 目录，默认使用 HF 配置")
    parser.add_argument("--output", default="data/grpo/train_grpo_2k_short.jsonl", help="输出路径")
    return parser.parse_args()


def infer_source(dataset_name: str, explicit_source: str) -> str:
    if explicit_source != "auto":
        return explicit_source

    lowered = dataset_name.lower()
    if "gsm8k" in lowered:
        return "gsm8k"
    return "numinamath"


def load_train_dataset(dataset_name: str, dataset_config: str | None, split: str, cache_dir: str | None):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def build_train_record(row: dict[str, object], index: int, source: str) -> dict[str, object]:
    if source == "numinamath":
        return to_sft_record(row, index)

    if source == "gsm8k":
        record = make_record(row, index, source, include_solution=False)
        raw_answer = str(row.get("answer", "")).strip()
        match = GSM8K_ANSWER_RE.search(raw_answer)
        final_answer = match.group(1).strip() if match else str(record["final_answer"])
        record["final_answer"] = final_answer
        record["solution"] = ensure_boxed_final_answer(raw_answer, final_answer)
        record["problem_type"] = "Arithmetic"
        return record

    return make_record(row, index, source, include_solution=True)


def sample_rows(
    rows: list[dict[str, object]],
    rng: random.Random,
    total_samples: int,
    source: str,
) -> list[dict[str, object]]:
    if total_samples <= 0:
        raise ValueError("num_samples 必须大于 0")

    if source == "numinamath":
        return sample_numinamath(rows, rng, total_samples)

    shuffled = list(rows)
    rng.shuffle(shuffled)
    if len(shuffled) < total_samples:
        raise ValueError(f"{source} 过滤后样本不足：{len(shuffled)} < {total_samples}")
    return shuffled[:total_samples]


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
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    source = infer_source(args.train_dataset, args.source)

    dataset = load_train_dataset(args.train_dataset, args.train_config, args.train_split, args.cache_dir)
    candidates: list[dict[str, object]] = []
    for index in range(len(dataset)):
        row = build_train_record(dict(dataset[index]), index, source)
        reference_solution = clean_completion_for_protocol(row["solution"])
        token_length = response_token_length(reference_solution, tokenizer)
        if token_length < args.min_response_tokens or token_length > args.max_response_tokens:
            continue
        candidates.append(
            build_grpo_record(
                row,
                response_tokens=token_length,
                reference_solution=reference_solution,
            )
        )

    sampled = sample_rows(candidates, rng, args.num_samples, source)
    write_jsonl(args.output, sampled)
    preview_rows(sampled)
    print("train problem_type counts:", Counter(str(row["problem_type"]) for row in sampled))
    print(f"Wrote {len(sampled)} rows to {args.output}")


if __name__ == "__main__":
    main()
