#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_eval_config
from post_train.eval import (
    preview_logged_samples,
    resolve_eval_tasks,
    run_eval_task,
    summarize_metrics_for_console,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测模型在本地数学 JSONL 数据集上的表现。")
    parser.add_argument("--config", required=True, help="评测配置文件路径")
    parser.add_argument("--model", default=None, help="覆盖配置中的模型路径")
    parser.add_argument("--dataset", default=None, help="覆盖配置中的评测集路径")
    parser.add_argument("--output", default=None, help="覆盖配置中的结果输出路径")
    parser.add_argument("--tasks", nargs="*", default=None, help="只跑指定任务名，可传多个")
    parser.add_argument("--backend", default=None, choices=("vllm",), help="覆盖评测后端")
    parser.add_argument("--batch-size", default=None, help="覆盖 batch size")
    parser.add_argument("--max-lora-rank", default=None, help="覆盖 vLLM 的 max_lora_rank")
    parser.add_argument("--max-new-tokens", default=None, help="覆盖生成长度上限")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条")
    return parser.parse_args()


def parse_batch_size(value):
    if value is None or value == "auto":
        return value
    return int(value)


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    model_name = args.model or cfg.model_name
    backend = args.backend or cfg.backend
    batch_size = parse_batch_size(args.batch_size) if args.batch_size is not None else cfg.batch_size
    max_lora_rank = int(args.max_lora_rank) if args.max_lora_rank is not None else cfg.max_lora_rank
    max_new_tokens = int(args.max_new_tokens) if args.max_new_tokens is not None else cfg.max_new_tokens
    tasks = resolve_eval_tasks(
        cfg,
        requested_tasks=args.tasks,
        dataset_override=args.dataset,
        output_override=args.output,
    )
    if backend != "vllm":
        raise ValueError("当前评测只支持 backend=vllm。")
    for task in tasks:
        run_result = run_eval_task(
            model_name=model_name,
            task=task,
            eval_cfg=cfg.model_copy(update={"backend": backend}),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            limit=args.limit,
            output_dir=cfg.output_dir,
            max_lora_rank=max_lora_rank,
        )
        raw_result = run_result["raw_result"]
        result = run_result["result"]
        preview = preview_logged_samples(result)
        if preview:
            print(preview)
        print(json.dumps(summarize_metrics_for_console(result["metrics"]), ensure_ascii=False, indent=2))
        print(f"wrote_raw_result={run_result['raw_result_path']}")
        print(f"wrote_result={run_result['result_path']}")


if __name__ == "__main__":
    main()
