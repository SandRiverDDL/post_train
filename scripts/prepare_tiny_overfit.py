#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.data import clean_completion_for_protocol
from rl.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从现有训练集固定抽样，生成 tiny-overfit 数据。")
    parser.add_argument("--input", default="data/train_sft.jsonl", help="输入训练集路径")
    parser.add_argument("--output", default="data/train_tiny_overfit.jsonl", help="输出 tiny-overfit 路径")
    parser.add_argument("--num-samples", type=int, default=16, help="抽样条数，建议 8~16")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--tokenizer", default=None, help="用于长度过滤的 tokenizer 路径")
    parser.add_argument("--max-completion-tokens", type=int, default=None, help="completion 最大 token 长度")
    parser.add_argument(
        "--mode",
        choices=("raw", "protocol", "shortcot", "longclean"),
        default="raw",
        help="导出模式：raw 保留原始长解答，protocol 只保留统一 boxed 尾部，shortcot 保留短推理和 boxed 尾部，longclean 保留清洗后的长推理。",
    )
    return parser.parse_args()


def _to_protocol_row(row: dict) -> dict:
    final_answer = str(row.get("final_answer", "")).strip()
    if not final_answer:
        raise ValueError(f"id={row.get('id')} 缺少 final_answer，无法导出 protocol-only 数据。")

    cleaned = dict(row)
    cleaned["solution"] = f"Final answer: \\boxed{{{final_answer}}}"
    return cleaned


def _to_shortcot_row(row: dict, tokenizer, max_completion_tokens: int) -> dict:
    final_answer = str(row.get("final_answer", "")).strip()
    if not final_answer:
        raise ValueError(f"id={row.get('id')} 缺少 final_answer，无法导出 short-CoT 数据。")

    cleaned_solution = clean_completion_for_protocol(str(row.get("solution", "")))
    lines = [line.rstrip() for line in cleaned_solution.splitlines()]
    boxed_line = f"Final answer: \\boxed{{{final_answer}}}"

    kept: list[str] = [boxed_line]
    token_count = len(tokenizer(boxed_line, add_special_tokens=False)["input_ids"])

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped or stripped == boxed_line:
            continue
        line_tokens = len(tokenizer(stripped + "\n", add_special_tokens=False)["input_ids"])
        if token_count + line_tokens > max_completion_tokens:
            break
        kept.append(stripped)
        token_count += line_tokens

    kept.reverse()
    cleaned = dict(row)
    cleaned["solution"] = "\n".join(kept)
    return cleaned


def _completion_token_length(row: dict, tokenizer) -> int:
    return len(tokenizer(str(row.get("solution", "")), add_special_tokens=False)["input_ids"])


def _to_longclean_row(row: dict) -> dict:
    cleaned = dict(row)
    cleaned["solution"] = clean_completion_for_protocol(str(row.get("solution", "")))
    return cleaned


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if len(rows) < args.num_samples:
        raise ValueError(f"输入样本不足：{len(rows)} < {args.num_samples}")

    tokenizer = None
    if args.mode == "shortcot" or args.max_completion_tokens is not None:
        tokenizer_name = args.tokenizer or "/home/chy/.cache/huggingface/hub/models--unsloth--Qwen3-1.7B-Base-unsloth-bnb-4bit/snapshots/c7079ee9a92d69fbe9e6616dea1888bf0fb0bbca"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    if args.mode == "protocol":
        rows = [_to_protocol_row(row) for row in rows]
    elif args.mode == "shortcot":
        if args.max_completion_tokens is None:
            raise ValueError("mode=shortcot 时必须显式提供 --max-completion-tokens。")
        assert tokenizer is not None
        rows = [_to_shortcot_row(row, tokenizer, args.max_completion_tokens) for row in rows]
    elif args.mode == "longclean":
        rows = [_to_longclean_row(row) for row in rows]

    if args.max_completion_tokens is not None:
        assert tokenizer is not None
        rows = [row for row in rows if _completion_token_length(row, tokenizer) <= args.max_completion_tokens]

    if len(rows) < args.num_samples:
        raise ValueError(f"过滤后样本不足：{len(rows)} < {args.num_samples}")

    rng = random.Random(args.seed)
    sampled = list(rows)
    rng.shuffle(sampled)
    sampled = sampled[: args.num_samples]
    write_jsonl(args.output, sampled)

    print(
        f"Wrote {len(sampled)} rows to {args.output} "
        f"(mode={args.mode}, max_completion_tokens={args.max_completion_tokens})"
    )
    for row in sampled[:3]:
        print("=" * 80)
        print(f"id: {row['id']}")
        print(row["question"][:200])
        print("--- solution tail ---")
        print("\n".join(clean_completion_for_protocol(row["solution"]).splitlines()[-4:]))


if __name__ == "__main__":
    main()
