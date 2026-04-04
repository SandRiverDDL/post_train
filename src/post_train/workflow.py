from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_train.config import (
    EvalConfig,
    EvalTaskConfig,
    GRPOTrainConfig,
    WorkflowConfig,
    load_eval_config,
    load_grpo_reward_config,
    load_grpo_train_config,
)
from post_train.eval import build_model_output_path, run_eval_task
from post_train.experiments import register_training_run
from post_train.grpo import train_grpo
from post_train.io import ensure_parent
from post_train.sft_selection import run_checkpoint_selection


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _collect_existing_metrics(*, model_name: str, task_names: list[str], eval_output_dir: str | Path) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    base = Path(eval_output_dir) / build_model_output_path(model_name)
    for task_name in task_names:
        payload = _read_json_if_exists(base / task_name / "result.json")
        if payload is None:
            continue
        metrics[task_name] = {
            "metrics": dict(payload.get("metrics", {})),
            "result_path": str(base / task_name / "result.json"),
        }
    return metrics


def _resolve_benchmark_tasks(eval_cfg: EvalConfig, benchmark_tasks: list[str]) -> list[EvalTaskConfig]:
    selected = {name for name in benchmark_tasks}
    tasks = [task for task in eval_cfg.tasks if task.name in selected]
    if len(tasks) != len(selected):
        missing = sorted(selected.difference({task.name for task in tasks}))
        raise ValueError(f"未匹配到 benchmark 任务：{missing}")
    return tasks


def _run_benchmark_tasks(
    *,
    model_name: str,
    tasks: list[EvalTaskConfig],
    eval_cfg: EvalConfig,
) -> dict[str, dict[str, Any]]:
    benchmarks: dict[str, dict[str, Any]] = {}
    for task in tasks:
        run_result = run_eval_task(
            model_name=model_name,
            task=task,
            eval_cfg=eval_cfg,
            batch_size=eval_cfg.batch_size,
            max_new_tokens=eval_cfg.max_new_tokens,
            limit=None,
            output_dir=eval_cfg.output_dir,
            max_lora_rank=eval_cfg.max_lora_rank,
        )
        benchmarks[task.name] = {
            "metrics": dict(run_result["result"]["metrics"]),
            "result_path": str(run_result["result_path"]),
            "raw_result_path": str(run_result["raw_result_path"]),
        }
    return benchmarks


def _resolve_baseline_metrics(
    *,
    train_cfg: GRPOTrainConfig,
    eval_cfg: EvalConfig,
    benchmark_tasks: list[EvalTaskConfig],
) -> dict[str, dict[str, Any]]:
    task_names = [task.name for task in benchmark_tasks]
    existing = _collect_existing_metrics(
        model_name=train_cfg.model_name_or_path,
        task_names=task_names,
        eval_output_dir=eval_cfg.output_dir,
    )
    missing_tasks = [task for task in benchmark_tasks if task.name not in existing]
    if not missing_tasks:
        return existing

    evaluated = _run_benchmark_tasks(
        model_name=train_cfg.model_name_or_path,
        tasks=missing_tasks,
        eval_cfg=eval_cfg,
    )
    return {**existing, **evaluated}


def _compare_benchmarks(
    *,
    benchmark_metrics: dict[str, dict[str, Any]],
    baseline_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    deltas: list[float] = []
    for task_name, current in sorted(benchmark_metrics.items()):
        current_score = float(current.get("metrics", {}).get("pass_at_1", 0.0))
        baseline_score = float(baseline_metrics.get(task_name, {}).get("metrics", {}).get("pass_at_1", 0.0))
        delta = current_score - baseline_score
        tasks[task_name] = {
            "current": current_score,
            "baseline": baseline_score,
            "delta": delta,
        }
        deltas.append(delta)

    if deltas and all(delta == 0.0 for delta in deltas):
        overall_status = "no_change"
    elif deltas and any(delta < 0.0 for delta in deltas):
        overall_status = "regressed"
    elif deltas:
        overall_status = "improved"
    else:
        overall_status = "no_change"
    return {
        "overall_status": overall_status,
        "tasks": tasks,
    }


def _write_failure_summary(
    *,
    output_dir: str | Path,
    workflow_cfg: WorkflowConfig,
    failure_stage: str,
    error: Exception,
) -> Path:
    return _write_json(
        Path(output_dir) / workflow_cfg.workflow_failure_name,
        {
            "status": "failed",
            "failure_stage": failure_stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def run_workflow(cfg: WorkflowConfig) -> dict[str, Any]:
    if cfg.route != "grpo":
        raise ValueError(f"当前 workflow 只支持 route=grpo，收到：{cfg.route}")

    train_cfg = load_grpo_train_config(cfg.train_config)
    eval_cfg = load_eval_config(cfg.eval_config)
    reward_cfg = load_grpo_reward_config(train_cfg.reward_config)

    try:
        output_dir = Path(train_grpo(train_cfg, reward_cfg))
    except Exception as exc:
        _write_failure_summary(output_dir=train_cfg.output_dir, workflow_cfg=cfg, failure_stage="training", error=exc)
        raise

    try:
        selection = run_checkpoint_selection(
            train_output_dir=output_dir,
            dataset_path=cfg.dev_dataset,
            eval_cfg=eval_cfg,
        )
        best = selection["best"]
        benchmark_tasks = _resolve_benchmark_tasks(eval_cfg, cfg.benchmark_tasks)
        benchmark_metrics = _run_benchmark_tasks(
            model_name=str(best["checkpoint_path"]),
            tasks=benchmark_tasks,
            eval_cfg=eval_cfg,
        )
        baseline_metrics = _resolve_baseline_metrics(
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            benchmark_tasks=benchmark_tasks,
        )
        comparison = _compare_benchmarks(
            benchmark_metrics=benchmark_metrics,
            baseline_metrics=baseline_metrics,
        )
        register_summary = register_training_run(
            route=cfg.route,
            config_path=cfg.train_config,
            output_dir=output_dir,
            train_dataset=train_cfg.train_dataset,
            base_model=train_cfg.model_name_or_path,
        )
        summary = {
            "status": "finished",
            "route": cfg.route,
            "train_output_dir": str(output_dir),
            "best_checkpoint_path": str(best["checkpoint_path"]),
            "best_global_step": int(best["global_step"]),
            "dev_dataset": str(cfg.dev_dataset),
            "dev_metrics": dict(best["metrics"]),
            "benchmark_metrics": benchmark_metrics,
            "baseline_metrics": baseline_metrics,
            "comparison": comparison,
            "run_summary_path": str(register_summary["summary_path"]),
            "ranking_path": str(selection["ranking_path"]),
            "best_path": str(selection["best_path"]),
        }
        summary_path = _write_json(output_dir / cfg.workflow_summary_name, summary)
        summary["workflow_summary_path"] = str(summary_path)
        return summary
    except Exception as exc:
        _write_failure_summary(output_dir=output_dir, workflow_cfg=cfg, failure_stage="post_training", error=exc)
        raise

