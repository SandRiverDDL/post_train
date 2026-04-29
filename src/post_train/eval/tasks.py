from __future__ import annotations

from pathlib import Path

from post_train.config import EvalConfig, EvalTaskConfig


def build_task_name(dataset_path: str | Path) -> str:
    return Path(dataset_path).stem.replace("-", "_")


def resolve_eval_tasks(
    cfg: EvalConfig,
    *,
    requested_tasks: list[str] | None = None,
    dataset_override: str | None = None,
    output_override: str | None = None,
) -> list[EvalTaskConfig]:
    if dataset_override is not None:
        dataset_path = Path(dataset_override)
        return [
            EvalTaskConfig(
                name=build_task_name(dataset_path),
                dataset_path=dataset_path,
                output_path=Path(output_override) if output_override else None,
            )
        ]

    if not requested_tasks:
        return list(cfg.tasks)

    selected = set(requested_tasks)
    tasks = [task for task in cfg.tasks if task.name in selected]
    if not tasks:
        raise ValueError(f"未匹配到任何任务：{requested_tasks}")
    return tasks
