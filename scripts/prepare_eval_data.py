#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.aime_eval import prepare_aime_eval_artifact, write_aime_eval_artifact
from post_train.data import (
    prepare_benchmark_artifact,
    prepare_stratified_math500_dev_artifact,
    preview_rows,
)
from post_train.io import ensure_parent, write_jsonl
from post_train.math220k_dev import (
    DATASET_NAME,
    DEFAULT_TOKENIZER_NAME,
    default_output_path,
    default_report_path,
    prepare_math220k_dev,
    profile_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备本地评测数据。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmarks = subparsers.add_parser("benchmarks", help="准备 GSM8K 与 MATH-500 benchmark 数据")
    benchmarks.add_argument("--cache-dir", default=None, help="datasets cache 目录")
    benchmarks.add_argument("--gsm8k-output", default="data/eval/gsm8k_test.jsonl", help="GSM8K 输出")
    benchmarks.add_argument("--math500-output", default="data/eval/math500_test.jsonl", help="MATH-500 输出")

    global_dev = subparsers.add_parser("global-dev", help="从 HuggingFaceH4/MATH-500 按 level 分层生成 dev 集")
    global_dev.add_argument("--sample-size", type=int, default=200, help="分层抽样条数")
    global_dev.add_argument("--seed", type=int, default=42, help="随机种子")
    global_dev.add_argument("--cache-dir", default=None, help="datasets cache 目录")
    global_dev.add_argument("--output", default="data/eval/math500_dev200.jsonl", help="输出 JSONL 路径")
    global_dev.add_argument("--report-output", default="data/eval/math500_dev200.report.json", help="输出 report 路径")

    aime = subparsers.add_parser("aime", help="准备 AIME 本地评测数据")
    aime.add_argument("--year", type=int, required=True, choices=(24, 25), help="AIME 年份")
    aime.add_argument("--dataset-name", default=None, help="覆盖默认数据集名")
    aime.add_argument("--config-name", default=None, help="数据集 config 名")
    aime.add_argument("--split", default="test", help="数据集 split")
    aime.add_argument("--source", default=None, help="覆盖输出记录 source 字段")
    aime.add_argument("--cache-dir", default=None, help="datasets cache 目录")
    aime.add_argument("--output", default=None, help="覆盖输出 JSONL 路径")

    math220k = subparsers.add_parser("math220k-dev", help="从 Math220K 生成固定 profile 的 dev 集")
    math220k.add_argument("--profile", default="main150", choices=profile_names(), help="内置 profile 名称")
    math220k.add_argument("--output", help="输出 JSONL 路径")
    math220k.add_argument("--report-output", help="输出 report JSON 路径")
    math220k.add_argument("--seed", type=int, default=42, help="随机种子")
    math220k.add_argument("--dataset-name", default=DATASET_NAME, help="数据集名称")
    math220k.add_argument("--split", default="train", help="数据集 split")
    math220k.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME, help="用于统计长度的 tokenizer")
    math220k.add_argument("--cache-dir", help="datasets cache 目录")

    return parser.parse_args()


def _prepare_benchmarks(args: argparse.Namespace) -> None:
    gsm8k_rows = prepare_benchmark_artifact(
        dataset_name="gsm8k",
        config_name="main",
        split="test",
        source="gsm8k",
        cache_dir=args.cache_dir,
    )
    math500_rows = prepare_benchmark_artifact(
        dataset_name="HuggingFaceH4/MATH-500",
        split="test",
        source="math500",
        cache_dir=args.cache_dir,
    )
    write_jsonl(args.gsm8k_output, gsm8k_rows)
    write_jsonl(args.math500_output, math500_rows)
    print(f"wrote_gsm8k={args.gsm8k_output}")
    print(f"wrote_math500={args.math500_output}")


def _prepare_global_dev(args: argparse.Namespace) -> None:
    rows, report = prepare_stratified_math500_dev_artifact(
        sample_size=args.sample_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    write_jsonl(args.output, rows)
    ensure_parent(args.report_output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("global_dev_preview")
    print(preview_rows(rows))
    print("dataset=HuggingFaceH4/MATH-500")
    print("split=test")
    print(f"sample_size={args.sample_size}")
    print(f"seed={args.seed}")
    print(f"wrote_global_dev={args.output}")
    print(f"wrote_global_dev_report={args.report_output}")


def _prepare_aime(args: argparse.Namespace) -> None:
    rows = prepare_aime_eval_artifact(
        year=args.year,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        split=args.split,
        source=args.source,
        cache_dir=args.cache_dir,
    )
    output_path = write_aime_eval_artifact(
        year=args.year,
        rows=rows,
        output_path=args.output,
    )
    print(f"aime{args.year}_preview")
    print(preview_rows(rows))
    print(f"year={args.year}")
    print(f"split={args.split}")
    print(f"wrote_aime={output_path}")


def _prepare_math220k_dev(args: argparse.Namespace) -> None:
    result = prepare_math220k_dev(
        profile=args.profile,
        output_path=args.output or default_output_path(args.profile),
        report_path=args.report_output or default_report_path(args.profile),
        seed=args.seed,
        dataset_name=args.dataset_name,
        split=args.split,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"wrote_math220k_dev={result['output_path']}")
    print(f"wrote_math220k_dev_report={result['report_path']}")


def main() -> None:
    args = parse_args()
    if args.command == "benchmarks":
        _prepare_benchmarks(args)
    elif args.command == "global-dev":
        _prepare_global_dev(args)
    elif args.command == "aime":
        _prepare_aime(args)
    elif args.command == "math220k-dev":
        _prepare_math220k_dev(args)
    else:
        raise ValueError(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
