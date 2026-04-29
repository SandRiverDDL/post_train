from __future__ import annotations

from pathlib import Path

from post_train.config import EvalConfig, EvalTaskConfig

from .runner import EvalRunner, EvalRunOverrides
from .tasks import resolve_eval_tasks
from .vllm import VLLMRunner


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
    runner: VLLMRunner | None = None,
) -> dict[str, object]:
    eval_runner = EvalRunner(eval_cfg, model_name=model_name, backend_runner=runner)
    return eval_runner.run_task(
        task,
        overrides=EvalRunOverrides(
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            limit=limit,
            output_dir=output_dir,
            max_lora_rank=max_lora_rank,
        ),
    )
