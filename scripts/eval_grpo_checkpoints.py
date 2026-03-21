#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import load_eval_config
from rl.harness_tasks import resolve_model_args, resolve_result_task_name, run_harness_eval
from rl.io import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="顺序评测 GRPO 输出目录下的 checkpoints。")
    parser.add_argument("--config", default="configs/eval.yaml", help="评测配置文件路径")
    parser.add_argument("--run-dir", required=True, help="包含 checkpoint-* 的训练输出目录")
    parser.add_argument("--dataset", default=None, help="评测集路径，默认使用配置中的 eval_dataset")
    parser.add_argument("--backend", default=None, help="评测后端，默认使用配置值")
    parser.add_argument("--batch-size", default="4", help="评测 batch size")
    parser.add_argument("--max-batch-size", default=None, help="batch_size=auto 时上限")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--max-new-tokens", default=None, help="覆盖配置中的生成上限")
    parser.add_argument("--attn-implementation", default=None, help="覆盖 attention 实现")
    parser.add_argument("--limit", default=None, help="限制评测样本数")
    parser.add_argument("--include-final", action="store_true", help="把 run-dir 本身也作为最终模型一起评测")
    parser.add_argument("--output", default=None, help="保存汇总结果的 JSONL 路径")
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


def _checkpoint_step(path: Path) -> int:
    if path.name.startswith("checkpoint-"):
        return int(path.name.split("-", 1)[1])
    return 10**12


def collect_model_dirs(run_dir: Path, include_final: bool) -> list[Path]:
    checkpoints = sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=_checkpoint_step,
    )
    if include_final and (run_dir / "adapter_config.json").exists():
        checkpoints.append(run_dir)
    return checkpoints


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    backend = _normalize_prefixed_value(args.backend, "backend") or cfg.eval_backend
    dataset_value = _normalize_prefixed_value(args.dataset, "dataset")
    dataset_path = Path(dataset_value or cfg.eval_dataset)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"未找到输出目录: {run_dir}")

    checkpoints = collect_model_dirs(run_dir, include_final=args.include_final)
    if not checkpoints:
        raise FileNotFoundError(f"{run_dir} 下未找到 checkpoint-* 目录。")

    attn_implementation = _normalize_prefixed_value(
        args.attn_implementation,
        "attn_implementation",
        "attn-implementation",
    ) or cfg.eval_attn_implementation
    max_gen_toks = _normalize_int(args.max_new_tokens, "max_new_tokens", "max-new-tokens") or cfg.eval_max_new_tokens
    limit = _normalize_int(args.limit, "limit")
    max_batch_size = _normalize_int(args.max_batch_size, "max_batch_size", "max-batch-size")

    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        ensure_parent(output_path).write_text("", encoding="utf-8")

    result_task_name = resolve_result_task_name(dataset_path, dataset_path.stem.replace("-", "_"))
    summary_rows: list[dict[str, object]] = []
    for model_dir in checkpoints:
        model_args = resolve_model_args(
            str(model_dir),
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
            task_name=dataset_path.stem.replace("-", "_"),
            batch_size=args.batch_size,
            max_batch_size=max_batch_size,
            limit=limit,
            strict=True,
            max_gen_toks=max_gen_toks,
        )
        metrics = result["results"].get(result_task_name, {})
        row = {
            "model_dir": str(model_dir),
            "checkpoint": model_dir.name,
            "step": None if model_dir == run_dir else _checkpoint_step(model_dir),
            "strict_match": metrics.get("exact_match,strict-match"),
            "strict_stderr": metrics.get("exact_match_stderr,strict-match"),
            "flexible_extract": metrics.get("exact_match,flexible-extract"),
            "flexible_stderr": metrics.get("exact_match_stderr,flexible-extract"),
        }
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
