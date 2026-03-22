#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import load_eval_config
from rl.io import ensure_parent
from rl.local_eval import run_official_eval


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


def default_output_path(run_dir: Path, dataset_path: Path) -> Path:
    return run_dir / f"checkpoint_eval_{dataset_path.stem}.jsonl"


def default_json_output_path(run_dir: Path, dataset_path: Path) -> Path:
    return run_dir / f"checkpoint_eval_{dataset_path.stem}.json"


def collect_model_dirs(run_dir: Path, include_final: bool) -> list[Path]:
    checkpoints = sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=_checkpoint_step,
    )
    if include_final and (run_dir / "adapter_config.json").exists():
        checkpoints.append(run_dir)
    return checkpoints


def _best_checkpoint_by_accuracy(rows: list[dict[str, object]]) -> dict[str, object] | None:
    scored_rows = [row for row in rows if row.get("normalized_accuracy") is not None]
    if not scored_rows:
        return None
    return max(scored_rows, key=lambda row: float(row["normalized_accuracy"]))


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
    output_path = ensure_parent(Path(args.output) if args.output else default_output_path(run_dir, dataset_path))
    json_output_path = ensure_parent(default_json_output_path(run_dir, dataset_path))
    output_path.write_text("", encoding="utf-8")

    summary_rows: list[dict[str, object]] = []
    for index, model_dir in enumerate(checkpoints, start=1):
        print(f"[{index}/{len(checkpoints)}] evaluating {model_dir.name}")
        result = run_official_eval(
            backend=backend,
            requested_model=str(model_dir),
            base_model=cfg.model_name,
            dataset_path=dataset_path,
            batch_size=args.batch_size,
            limit=limit,
            max_new_tokens=max_gen_toks,
            max_length=cfg.max_seq_length,
            device=args.device,
            attn_implementation=attn_implementation,
            gpu_memory_utilization=cfg.eval_gpu_memory_utilization,
            prompt_version=cfg.prompt_version,
            seed=42,
            preview_count=0,
        )
        metrics = result["metrics"]
        row = {
            "model_dir": str(model_dir),
            "checkpoint": model_dir.name,
            "step": None if model_dir == run_dir else _checkpoint_step(model_dir),
            "dataset": metrics.get("dataset", str(dataset_path)),
            "samples": metrics.get("samples"),
            "format_success_rate": metrics.get("format_success_rate"),
            "format_success_stderr": metrics.get("format_success_stderr"),
            "parse_success_rate": metrics.get("parse_success_rate"),
            "parse_success_stderr": metrics.get("parse_success_stderr"),
            "normalized_accuracy": metrics.get("normalized_accuracy"),
            "normalized_accuracy_stderr": metrics.get("normalized_accuracy_stderr"),
        }
        summary_rows.append(row)
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "rows": summary_rows,
        "best_checkpoint_by_normalized_accuracy": _best_checkpoint_by_accuracy(summary_rows),
    }
    with json_output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"wrote {len(summary_rows)} rows to {output_path}")
    best_row = summary["best_checkpoint_by_normalized_accuracy"]
    if best_row is not None:
        print(
            "best normalized_accuracy: "
            f"{best_row['checkpoint']} = {best_row['normalized_accuracy']}"
        )
    print(f"wrote summary json to {json_output_path}")


if __name__ == "__main__":
    main()
