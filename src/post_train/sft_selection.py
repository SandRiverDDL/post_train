from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from post_train.config import EvalConfig, EvalTaskConfig
from post_train.eval import (
    run_eval_task,
)
from post_train.io import ensure_parent

CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(path: str | Path) -> int:
    match = CHECKPOINT_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"无效 checkpoint 目录名：{path}")
    return int(match.group(1))


def scan_checkpoint_dirs(output_dir: str | Path) -> list[Path]:
    base = Path(output_dir)
    checkpoints = [path for path in base.iterdir() if path.is_dir() and CHECKPOINT_RE.search(path.name)]
    return sorted(checkpoints, key=checkpoint_step)


def _parse_batch_size(value: int | str) -> int | str:
    if value == "auto":
        return "auto"
    return int(value)


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    eval_cfg: EvalConfig,
    batch_size: int | str,
    max_new_tokens: int,
    limit: int | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_path)
    run_result = run_eval_task(
        model_name=str(checkpoint_dir),
        task=EvalTaskConfig(
            name=Path(dataset_path).stem.replace("-", "_"),
            dataset_path=Path(dataset_path),
        ),
        eval_cfg=eval_cfg,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        limit=limit,
        output_dir=Path(output_dir) / checkpoint_dir.name,
    )
    result = run_result["result"]
    return {
        "checkpoint_path": str(checkpoint_dir),
        "global_step": checkpoint_step(checkpoint_dir),
        "metrics": result["metrics"],
        "result_path": str(run_result["result_path"]),
        "raw_result_path": str(run_result["raw_result_path"]),
    }


def _selection_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
    metrics = record["metrics"]
    return (
        float(metrics.get("normalized_accuracy", 0.0)),
        float(metrics.get("boxed_rate", 0.0)),
        float(metrics.get("parse_success_rate", 0.0)),
        -int(record.get("global_step", 0)),
    )


def select_best_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("没有可选的 checkpoint 评测结果。")
    return max(records, key=_selection_key)


def write_best_checkpoint_summary(
    *,
    output_dir: str | Path,
    dataset_path: str | Path,
    records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    base = Path(output_dir)
    summary_path = ensure_parent(base / "dev_ranking.json")
    best_path = ensure_parent(base / "best_checkpoint.json")
    best_record = select_best_checkpoint(records)
    summary = {
        "dataset_path": str(dataset_path),
        "runner": "vllm_raw",
        "selection_metric": "normalized_accuracy",
        "records": records,
        "best": best_record,
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with best_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "checkpoint_path": best_record["checkpoint_path"],
                "global_step": best_record["global_step"],
                "metrics": best_record["metrics"],
                "runner": "vllm_raw",
                "dev_dataset": str(dataset_path),
                "selection_metric": "normalized_accuracy",
                "ranking_path": str(summary_path),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    return summary_path, best_path


def run_checkpoint_selection(
    *,
    train_output_dir: str | Path,
    dataset_path: str | Path,
    eval_cfg: EvalConfig,
    backend: str | None = None,
    batch_size: int | str | None = None,
    max_new_tokens: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    resolved_backend = backend or eval_cfg.backend
    if resolved_backend != "vllm":
        raise ValueError("当前评测只支持 backend=vllm。")
    resolved_batch_size = _parse_batch_size(batch_size if batch_size is not None else eval_cfg.batch_size)
    resolved_max_new_tokens = int(max_new_tokens if max_new_tokens is not None else eval_cfg.max_new_tokens)

    checkpoint_dirs = scan_checkpoint_dirs(train_output_dir)
    ranking_output_dir = Path(train_output_dir) / "dev_eval"
    records = [
        evaluate_checkpoint(
            checkpoint_path=checkpoint_dir,
            dataset_path=dataset_path,
            eval_cfg=eval_cfg,
            batch_size=resolved_batch_size,
            max_new_tokens=resolved_max_new_tokens,
            limit=limit,
            output_dir=ranking_output_dir,
        )
        for checkpoint_dir in checkpoint_dirs
    ]
    ranking_path, best_path = write_best_checkpoint_summary(
        output_dir=ranking_output_dir,
        dataset_path=dataset_path,
        records=records,
    )
    return {
        "records": records,
        "best": select_best_checkpoint(records),
        "ranking_path": str(ranking_path),
        "best_path": str(best_path),
    }
