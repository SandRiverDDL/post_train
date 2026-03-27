from __future__ import annotations

from pathlib import Path

from post_train.config import EvalConfig, EvalTaskConfig

from .io import resolve_task_output_paths, write_eval_result, write_raw_eval_result
from .metrics import result_from_vllm_raw_logs
from .vllm import build_task_name, resolve_model_args, run_vllm_raw_eval


def resolve_eval_tasks(
    cfg: EvalConfig,
    *,
    requested_tasks: list[str] | None = None,
    dataset_override: str | None = None,
    output_override: str | None = None,
) -> list[EvalTaskConfig]:
    if dataset_override is not None:
        dataset_path = Path(dataset_override)
        task_name = build_task_name(dataset_path)
        return [
            EvalTaskConfig(
                name=task_name,
                dataset_path=dataset_path,
                output_path=Path(output_override) if output_override else None,
            )
        ]

    if not requested_tasks:
        return list(cfg.tasks)

    selected = {name for name in requested_tasks}
    tasks = [task for task in cfg.tasks if task.name in selected]
    if not tasks:
        raise ValueError(f"未匹配到任何任务：{requested_tasks}")
    return tasks


def run_eval_task(
    *,
    model_name: str,
    task: EvalTaskConfig,
    eval_cfg: EvalConfig,
    batch_size: int | str,
    max_new_tokens: int,
    limit: int | None,
    output_dir: str | Path | None = None,
    max_lora_rank: int | None = None,
) -> dict[str, object]:
    model_args = resolve_model_args(
        model_name,
        eval_cfg.model_name,
        backend=eval_cfg.backend,
        max_length=eval_cfg.max_seq_length,
        device=eval_cfg.device,
        attn_implementation=eval_cfg.attn_implementation,
        gpu_memory_utilization=eval_cfg.gpu_memory_utilization,
        max_lora_rank=max_lora_rank if max_lora_rank is not None else eval_cfg.max_lora_rank,
    )
    raw_result = run_vllm_raw_eval(
        model_args=model_args,
        dataset_path=task.dataset_path,
        task_name=task.name,
        batch_size=batch_size,
        limit=limit,
        max_gen_toks=max_new_tokens,
        samples_per_problem=task.samples_per_problem,
        sampling_temperature=task.sampling_temperature,
        sampling_top_p=task.sampling_top_p,
    )
    result = result_from_vllm_raw_logs(
        raw_result,
        task_name=task.name,
        dataset_path=str(task.dataset_path),
        model_name=model_name,
    )
    final_output_path, raw_output_path = resolve_task_output_paths(
        task,
        output_dir=Path(output_dir) if output_dir is not None else eval_cfg.output_dir,
        model_name=model_name,
    )
    write_raw_eval_result(raw_output_path, raw_result)
    write_eval_result(final_output_path, result)
    return {
        "task_name": task.name,
        "raw_result": raw_result,
        "result": result,
        "result_path": str(final_output_path),
        "raw_result_path": str(raw_output_path),
    }
