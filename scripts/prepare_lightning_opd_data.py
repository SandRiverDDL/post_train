#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import LightningOPDDataConfig, load_lightning_opd_data_config
from post_train.lightning_opd import (
    build_lightning_opd_prompts,
    merge_lightning_opd_shards,
    run_lightning_opd_shard,
    run_lightning_opd_teacher_shard,
)


def _override(base: dict[str, Any], args: argparse.Namespace, key: str) -> None:
    value = getattr(args, key)
    if value is not None:
        base[key] = value


def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_lightning_opd_data_config(args.config) if args.config else LightningOPDDataConfig()
    values = cfg.model_dump()
    for key in [
        "output_dir",
        "raw_rollout_dir",
        "prompt_source",
        "sample_size",
        "seed",
        "num_shards",
        "student_model",
        "student_base_model",
        "teacher_model",
        "tokenizer_name",
        "max_new_tokens",
        "max_model_len",
        "temperature",
        "top_p",
        "top_k",
        "teacher_batch_size",
        "gpu_memory_utilization",
        "teacher_load_in_4bit",
    ]:
        _override(values, args, key)
    values["mode"] = args.mode
    values["raw_rollout_dir"] = args.raw_rollout_dir
    values["shard_index"] = args.shard_index
    for nullable_key in ["student_base_model", "tokenizer_name"]:
        if values.get(nullable_key) == "":
            values[nullable_key] = None
    return argparse.Namespace(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 Lightning-OPD offline 数据。")
    parser.add_argument("--config", default=None, help="YAML 配置文件。CLI 参数会覆盖 YAML。")
    parser.add_argument("--mode", choices=("prompts", "shard", "teacher", "merge"), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-rollout-dir", default=None, help="teacher 模式复用已有 raw_rollouts 的目录；默认等于 output-dir。")
    parser.add_argument("--prompt-source", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--student-base-model", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--teacher-batch-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--teacher-load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = resolve_config(parse_args())
    if args.mode == "prompts":
        result = build_lightning_opd_prompts(
            prompt_source=args.prompt_source,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    elif args.mode == "shard":
        if args.shard_index is None:
            raise ValueError("shard 模式必须传 --shard-index。")
        result = run_lightning_opd_shard(
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            student_model=args.student_model,
            student_base_model=args.student_base_model,
            teacher_model=args.teacher_model,
            tokenizer_name=args.tokenizer_name,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            teacher_batch_size=args.teacher_batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            teacher_load_in_4bit=args.teacher_load_in_4bit,
        )
    elif args.mode == "teacher":
        if args.shard_index is None:
            raise ValueError("teacher 模式必须传 --shard-index。")
        result = run_lightning_opd_teacher_shard(
            output_dir=args.output_dir,
            raw_rollout_dir=args.raw_rollout_dir,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            teacher_model=args.teacher_model,
            tokenizer_name=args.tokenizer_name,
            top_k=args.top_k,
            max_model_len=args.max_model_len,
            teacher_batch_size=args.teacher_batch_size,
            teacher_load_in_4bit=args.teacher_load_in_4bit,
        )
    else:
        result = merge_lightning_opd_shards(
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            raw_rollout_dir=args.raw_rollout_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
