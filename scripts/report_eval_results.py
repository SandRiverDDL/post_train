#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.eval.reporting import EvalReportPaths, write_eval_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动汇总 eval result.json 并生成 leaderboard。")
    parser.add_argument("--eval-root", default="outputs/eval", help="eval 结果根目录")
    parser.add_argument("--metadata", default="docs/analysis/eval_metadata.yaml", help="模型元数据 YAML")
    parser.add_argument("--registry", default="docs/analysis/eval_registry.jsonl", help="结构化 eval registry 输出")
    parser.add_argument("--leaderboard", default="docs/analysis/eval_leaderboard.md", help="Markdown leaderboard 输出")
    parser.add_argument(
        "--manual",
        action="append",
        default=None,
        help="额外读取历史手写 markdown 中的 JSON 结果；可重复传入。默认使用 metadata 中配置。",
    )
    parser.add_argument("--no-update-metadata", action="store_true", help="只生成报告，不重写 metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, paths = write_eval_report(
        eval_root=args.eval_root,
        paths=EvalReportPaths(
            metadata=Path(args.metadata),
            registry=Path(args.registry),
            leaderboard=Path(args.leaderboard),
        ),
        manual_result_files=args.manual,
        update_metadata=not args.no_update_metadata,
    )
    active_count = sum(1 for row in rows if not row.get("ignore"))
    ignored_count = len(rows) - active_count
    tracked_count = sum(1 for row in rows if row.get("tracked"))
    print(f"eval_rows={len(rows)}")
    print(f"tracked_rows={tracked_count}")
    print(f"hidden_rows={len(rows) - tracked_count}")
    print(f"active_rows={active_count}")
    print(f"ignored_rows={ignored_count}")
    print(f"wrote_metadata={paths.metadata}")
    print(f"wrote_registry={paths.registry}")
    print(f"wrote_leaderboard={paths.leaderboard}")


if __name__ == "__main__":
    main()
