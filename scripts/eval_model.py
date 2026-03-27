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
    resolve_model_args,
    resolve_eval_tasks,
    resolve_task_output_paths,
    result_from_harness_logs,
    result_from_vllm_raw_logs,
    run_harness_eval,
    run_vllm_raw_eval,
    summarize_metrics_for_console,
    write_eval_result,
    write_raw_eval_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测模型在本地数学 JSONL 数据集上的表现。")
    parser.add_argument("--config", required=True, help="评测配置文件路径")
    parser.add_argument("--model", default=None, help="覆盖配置中的模型路径")
    parser.add_argument("--dataset", default=None, help="覆盖配置中的评测集路径")
    parser.add_argument("--output", default=None, help="覆盖配置中的结果输出路径")
    parser.add_argument("--tasks", nargs="*", default=None, help="只跑指定任务名，可传多个")
    parser.add_argument("--runner", default=None, choices=("vllm_raw", "lm_eval"), help="覆盖评测执行链路")
    parser.add_argument("--backend", default=None, choices=("hf", "vllm"), help="覆盖评测后端")
    parser.add_argument("--batch-size", default=None, help="覆盖 batch size")
    parser.add_argument("--max-batch-size", default=None, help="覆盖 max batch size")
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
    runner = args.runner or cfg.runner
    backend = args.backend or cfg.backend
    batch_size = parse_batch_size(args.batch_size) if args.batch_size is not None else cfg.batch_size
    max_batch_size = int(args.max_batch_size) if args.max_batch_size is not None else cfg.max_batch_size
    max_lora_rank = int(args.max_lora_rank) if args.max_lora_rank is not None else cfg.max_lora_rank
    max_new_tokens = int(args.max_new_tokens) if args.max_new_tokens is not None else cfg.max_new_tokens
    tasks = resolve_eval_tasks(
        cfg,
        requested_tasks=args.tasks,
        dataset_override=args.dataset,
        output_override=args.output,
    )
    model_args = resolve_model_args(
        model_name,
        cfg.model_name,
        backend=backend,
        max_length=cfg.max_seq_length,
        device=cfg.device,
        attn_implementation=cfg.attn_implementation,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_lora_rank=max_lora_rank,
    )
    if runner == "vllm_raw" and backend != "vllm":
        raise ValueError("vllm_raw runner 只支持 backend=vllm。")
    for task in tasks:
        if runner == "lm_eval":
            raw_result = run_harness_eval(
                backend=backend,
                model_args=model_args,
                dataset_path=task.dataset_path,
                task_name=task.name,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                limit=args.limit,
                max_gen_toks=max_new_tokens,
                samples_per_problem=task.samples_per_problem,
                sampling_temperature=task.sampling_temperature,
                sampling_top_p=task.sampling_top_p,
            )
            result = result_from_harness_logs(
                raw_result,
                task_name=task.name,
                dataset_path=task.dataset_path,
                model_name=model_name,
            )
        else:
            raw_result = run_vllm_raw_eval(
                model_args=model_args,
                dataset_path=task.dataset_path,
                task_name=task.name,
                batch_size=batch_size,
                limit=args.limit,
                max_gen_toks=max_new_tokens,
                samples_per_problem=task.samples_per_problem,
                sampling_temperature=task.sampling_temperature,
                sampling_top_p=task.sampling_top_p,
            )
            result = result_from_vllm_raw_logs(
                raw_result,
                task_name=task.name,
                dataset_path=task.dataset_path,
                model_name=model_name,
            )
        preview = preview_logged_samples(result)
        if preview:
            print(preview)
        print(json.dumps(summarize_metrics_for_console(result["metrics"]), ensure_ascii=False, indent=2))
        final_output_path, raw_output_path = resolve_task_output_paths(task, output_dir=cfg.output_dir, runner=runner)
        write_raw_eval_result(raw_output_path, raw_result)
        write_eval_result(final_output_path, result)
        print(f"wrote_raw_result={raw_output_path}")
        print(f"wrote_result={final_output_path}")


if __name__ == "__main__":
    main()
