#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.tracking import log_eval_result, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把已有 outputs/eval/**/result.json 导入 MLflow。")
    parser.add_argument("--eval-root", default="outputs/eval", help="评测结果根目录")
    parser.add_argument("--limit", type=int, default=None, help="最多导入多少个 result.json，默认全量")
    return parser.parse_args()


def infer_model_and_task(result_path: Path, eval_root: Path) -> tuple[str, str]:
    relative = result_path.relative_to(eval_root)
    task = relative.parent.name
    model = str(relative.parent.parent)
    return model, task


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    result_paths = sorted(eval_root.glob("**/result.json"))
    if args.limit is not None:
        result_paths = result_paths[: args.limit]
    imported = 0
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        model, task = infer_model_and_task(result_path, eval_root)
        raw_path = result_path.with_name("raw.json")
        with start_run(
            run_name=f"import-{Path(model).name}",
            route="eval_import",
            params={"model_name": model, "task": task, "result_path": str(result_path)},
        ):
            log_eval_result(
                {
                    "task_name": task,
                    "result": result,
                    "result_path": str(result_path),
                    "raw_result_path": str(raw_path),
                }
            )
        imported += 1
    print(f"imported_results={imported}")


if __name__ == "__main__":
    main()
