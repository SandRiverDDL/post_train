#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import load_eval_config
from rl.harness_tasks import (
    preview_logged_samples,
    resolve_model_args,
    resolve_result_task_name,
    run_harness_eval,
    write_result,
)
from rl.local_eval import run_official_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测本地数学数据集。")
    parser.add_argument("--config", default="configs/eval.yaml", help="评测配置文件路径")
    parser.add_argument(
        "--mode",
        choices=("official", "benchmark"),
        default="official",
        help="official 使用本地 boxed/math-verify 评测；benchmark 使用 harness 原生 benchmark。",
    )
    parser.add_argument("--model", default=None, help="待评测模型路径，默认使用配置中的 base model")
    parser.add_argument("--dataset", default=None, help="评测集路径，默认使用配置中的 eval_dataset")
    parser.add_argument("--limit", default=None, help="仅评测前 N 条")
    parser.add_argument("--output", default=None, help="结果输出路径")
    parser.add_argument("--backend", default=None, help="评测后端，默认使用配置值")
    parser.add_argument("--batch-size", default="4", help="harness 推理 batch size，例如 3 或 auto")
    parser.add_argument("--max-batch-size", default=None, help="batch_size=auto 时的上限")
    parser.add_argument("--device", default="cuda", help="推理设备，默认 cuda")
    parser.add_argument("--max-new-tokens", default=None, help="覆盖配置中的评测生成上限")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="覆盖配置中的 attention 实现",
    )
    return parser.parse_args()


def _normalize_prefixed_value(value: str | None, *prefixes: str) -> str | None:
    if value is None:
        return None
    for prefix in prefixes:
        marker = f"{prefix}="
        if value.startswith(marker):
            return value[len(marker) :]
    return value


def _normalize_int(value: str | None, *prefixes: str) -> int | None:
    normalized = _normalize_prefixed_value(value, *prefixes)
    if normalized in (None, ""):
        return None
    return int(normalized)


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    backend = _normalize_prefixed_value(args.backend, "backend") or cfg.eval_backend
    if backend not in {"vllm", "hf"}:
        raise ValueError(f"不支持的 backend: {backend}")

    dataset_value = _normalize_prefixed_value(args.dataset, "dataset")
    dataset_path = Path(dataset_value or cfg.eval_dataset)
    task_name = dataset_path.stem.replace("-", "_")
    result_task_name = resolve_result_task_name(dataset_path, task_name)
    model_value = _normalize_prefixed_value(args.model, "model")
    attn_implementation = _normalize_prefixed_value(
        args.attn_implementation,
        "attn_implementation",
        "attn-implementation",
    ) or cfg.eval_attn_implementation
    if attn_implementation not in {"sdpa", "flash_attention_2", "eager"}:
        raise ValueError(f"不支持的 attention 实现: {attn_implementation}")

    model_args = resolve_model_args(
        model_value,
        cfg.model_name,
        backend=backend,
        max_length=cfg.max_seq_length,
        device=args.device,
        attn_implementation=attn_implementation,
        gpu_memory_utilization=cfg.eval_gpu_memory_utilization,
    )
    max_gen_toks = _normalize_int(args.max_new_tokens, "max_new_tokens", "max-new-tokens") or cfg.eval_max_new_tokens
    limit = _normalize_int(args.limit, "limit")
    max_batch_size = _normalize_int(args.max_batch_size, "max_batch_size", "max-batch-size")

    if args.mode == "official":
        result = run_official_eval(
            backend=backend,
            requested_model=model_value,
            base_model=cfg.model_name,
            dataset_path=dataset_path,
            limit=limit,
            batch_size=args.batch_size,
            max_new_tokens=max_gen_toks,
            max_length=cfg.max_seq_length,
            device=args.device,
            attn_implementation=attn_implementation,
            gpu_memory_utilization=cfg.eval_gpu_memory_utilization,
            prompt_version=cfg.prompt_version,
            seed=42,
        )
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        output_value = _normalize_prefixed_value(args.output, "output")
        if output_value:
            write_result(output_value, result)
        return

    try:
        result = run_harness_eval(
            backend=backend,
            model_args=model_args,
            dataset_path=dataset_path,
            task_name=task_name,
            batch_size=args.batch_size,
            max_batch_size=max_batch_size,
            limit=limit,
            strict=True,
            max_gen_toks=max_gen_toks,
        )
    except ValueError as exc:
        if backend != "vllm" or "decoder prompt cannot be empty" not in str(exc):
            raise
        print("检测到 vllm 在当前 benchmark 请求上返回空 prompt，自动回退到 hf 后端重试。")
        backend = "hf"
        model_args = resolve_model_args(
            model_value,
            cfg.model_name,
            backend=backend,
            max_length=cfg.max_seq_length,
            device=args.device,
            attn_implementation=attn_implementation,
            gpu_memory_utilization=cfg.eval_gpu_memory_utilization,
        )
        result = run_harness_eval(
            backend=backend,
            model_args=model_args,
            dataset_path=dataset_path,
            task_name=task_name,
            batch_size=args.batch_size,
            max_batch_size=max_batch_size,
            limit=limit,
            strict=True,
            max_gen_toks=max_gen_toks,
        )
    preview_logged_samples(result, result_task_name, count=3)
    print(json.dumps(result["results"].get(result_task_name, {}), ensure_ascii=False, indent=2))
    output_value = _normalize_prefixed_value(args.output, "output")
    if output_value:
        write_result(output_value, result)


if __name__ == "__main__":
    main()
