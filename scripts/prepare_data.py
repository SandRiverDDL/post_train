#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.data import NUMINA_CATEGORY_TARGETS, format_eval_text, format_sft_text, infer_problem_type, to_eval_record, to_sft_record
from rl.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 NuminaMath / GSM8K / MATH500 数据。")
    parser.add_argument("--model-name", default="unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit", help="用于长度过滤的 tokenizer")
    parser.add_argument("--train-dataset", default="nlile/NuminaMath-1.5-RL-Verifiable", help="训练集名")
    parser.add_argument("--train-split", default="train", help="训练集 split")
    parser.add_argument("--dev-dataset", default="gsm8k", help="开发集名")
    parser.add_argument("--dev-config", default="main", help="GSM8K 配置名")
    parser.add_argument("--dev-split", default="test", help="开发集 split")
    parser.add_argument("--test-dataset", default="HuggingFaceH4/MATH-500", help="测试集名")
    parser.add_argument("--test-split", default="test", help="测试集 split")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train-num-samples", type=int, default=3000, help="训练集总样本数")
    parser.add_argument("--train-min-length", type=int, default=256, help="训练集最小总 token 长度")
    parser.add_argument("--train-max-length", type=int, default=1024, help="训练集最大总 token 长度")
    parser.add_argument("--train-min-response-tokens", type=int, default=None, help="训练集 solution 最小 token 长度")
    parser.add_argument("--train-max-response-tokens", type=int, default=None, help="训练集 solution 最大 token 长度")
    parser.add_argument("--train-output", default="data/train_sft.jsonl", help="训练集输出路径")
    parser.add_argument("--dev-output", default="data/eval/gsm8k_dev200.jsonl", help="开发集输出路径")
    parser.add_argument("--test-output", default="data/eval/math500_test.jsonl", help="测试集输出路径")
    return parser.parse_args()


def record_token_length(record: dict[str, str], tokenizer) -> int:
    if record.get("solution"):
        text = format_sft_text(record["question"], record["solution"])
    else:
        text = format_eval_text(record["question"], record["final_answer"])
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def filter_by_length(records: list[dict[str, str]], tokenizer, min_length: int, max_length: int) -> list[dict[str, str]]:
    filtered = []
    for record in records:
        token_length = record_token_length(record, tokenizer)
        if min_length <= token_length <= max_length:
            filtered.append(record)
    return filtered


def response_token_length(record: dict[str, str], tokenizer) -> int:
    return len(tokenizer(record.get("solution", ""), add_special_tokens=False)["input_ids"])


def filter_train_by_response_length(
    records: list[dict[str, str]],
    tokenizer,
    min_tokens: int | None,
    max_tokens: int | None,
) -> list[dict[str, str]]:
    if min_tokens is None and max_tokens is None:
        return records

    filtered = []
    for record in records:
        token_length = response_token_length(record, tokenizer)
        if min_tokens is not None and token_length < min_tokens:
            continue
        if max_tokens is not None and token_length > max_tokens:
            continue
        filtered.append(record)
    return filtered


def scaled_category_targets(total_samples: int) -> dict[str, int]:
    base_total = sum(NUMINA_CATEGORY_TARGETS.values())
    if total_samples <= 0:
        raise ValueError("train_num_samples 必须大于 0")

    raw_targets = {
        category: total_samples * target / base_total
        for category, target in NUMINA_CATEGORY_TARGETS.items()
    }
    scaled = {category: int(value) for category, value in raw_targets.items()}
    remainder = total_samples - sum(scaled.values())
    ranking = sorted(
        raw_targets.items(),
        key=lambda item: (item[1] - scaled[item[0]], NUMINA_CATEGORY_TARGETS[item[0]]),
        reverse=True,
    )
    for category, _ in ranking[:remainder]:
        scaled[category] += 1
    return scaled


def sample_numinamath(rows: list[dict[str, str]], rng: random.Random, total_samples: int) -> list[dict[str, str]]:
    targets = scaled_category_targets(total_samples)
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row.get("problem_type", infer_problem_type(row))].append(row)

    summary = {category: len(buckets.get(category, [])) for category in targets}
    missing = {
        category: target
        for category, target in targets.items()
        if len(buckets.get(category, [])) < target
    }
    if missing:
        raise ValueError(f"NuminaMath 分层抽样样本不足：{missing}；当前计数：{summary}")

    sampled: list[dict[str, str]] = []
    for category, target in targets.items():
        bucket = list(buckets[category])
        rng.shuffle(bucket)
        sampled.extend(bucket[:target])

    rng.shuffle(sampled)
    return sampled


def preview_rows(name: str, rows: list[dict[str, str]]) -> None:
    print(f"\n{name} preview")
    for row in rows[:3]:
        print("=" * 80)
        print(f"id: {row['id']}")
        if "problem_type" in row:
            print(f"problem_type: {row['problem_type']}")
        print(row["question"][:300])
        print("--- solution tail ---")
        solution = row.get("solution", f"Final answer: \\boxed{{{row['final_answer']}}}")
        print("\n".join(solution.splitlines()[-4:]))


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    rng = random.Random(args.seed)

    train_dataset = load_dataset(args.train_dataset, split=args.train_split)
    train_rows = [to_sft_record(dict(train_dataset[index]), index) for index in range(len(train_dataset))]
    train_rows = filter_train_by_response_length(
        train_rows,
        tokenizer,
        args.train_min_response_tokens,
        args.train_max_response_tokens,
    )
    train_rows = filter_by_length(train_rows, tokenizer, args.train_min_length, args.train_max_length)
    train_rows = sample_numinamath(train_rows, rng, args.train_num_samples)
    write_jsonl(args.train_output, train_rows)
    preview_rows("train", train_rows)
    print("train problem_type counts:", Counter(row["problem_type"] for row in train_rows))
    print(f"Wrote {len(train_rows)} rows to {args.train_output}")

    dev_dataset = load_dataset(args.dev_dataset, args.dev_config, split=args.dev_split)
    dev_rows = [to_eval_record(dict(dev_dataset[index]), index, "gsm8k") for index in range(len(dev_dataset))]
    if len(dev_rows) < 200:
        raise ValueError(f"GSM8K 过滤后样本不足：{len(dev_rows)} < 200")
    rng.shuffle(dev_rows)
    dev_rows = dev_rows[:200]
    write_jsonl(args.dev_output, dev_rows)
    preview_rows("dev", dev_rows)
    print(f"Wrote {len(dev_rows)} rows to {args.dev_output}")

    test_dataset = load_dataset(args.test_dataset, split=args.test_split)
    test_rows = [to_eval_record(dict(test_dataset[index]), index, "math500") for index in range(len(test_dataset))]
    write_jsonl(args.test_output, test_rows)
    preview_rows("test", test_rows)
    print(f"Wrote {len(test_rows)} rows to {args.test_output}")


if __name__ == "__main__":
    main()
