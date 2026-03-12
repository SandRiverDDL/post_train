#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将评测结果追加记录到 results/runs.jsonl。")
    parser.add_argument("--input", required=True, help="eval_dataset.py 的 JSON 输出路径")
    parser.add_argument("--output", default="results/runs.jsonl", help="结果记录文件")
    parser.add_argument("--run-name", required=True, help="实验名")
    parser.add_argument("--dataset", required=True, help="数据集标识")
    parser.add_argument("--mode", required=True, help="实验模式，例如 protocol/shortcot/longclean")
    parser.add_argument("--notes", default="", help="补充说明")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    task_name = next(iter(raw.get("results", {})))
    metrics = raw["results"][task_name]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "dataset": args.dataset,
        "mode": args.mode,
        "task": task_name,
        "metrics": metrics,
        "notes": args.notes,
        "source_file": args.input,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
