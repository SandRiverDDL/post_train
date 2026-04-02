#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.experiments import discover_run_summaries, write_experiment_summary_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总当前仓库实验结果。")
    parser.add_argument("--registry", default="experiments/registry.jsonl", help="实验 registry 路径")
    parser.add_argument("--output-md", default="experiments/summary.md", help="Markdown 汇总输出路径")
    parser.add_argument("--output-csv", default="experiments/summary.csv", help="CSV 汇总输出路径")
    parser.add_argument("--output-root", default="outputs", help="实验输出根目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = discover_run_summaries(output_root=args.output_root, registry_path=args.registry)
    md_path, csv_path = write_experiment_summary_table(
        rows=rows,
        markdown_path=args.output_md,
        csv_path=args.output_csv,
    )
    print(f"summary_runs={len(rows)}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_csv={csv_path}")


if __name__ == "__main__":
    main()
