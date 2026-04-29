from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_train.config import EvalConfig, EvalTaskConfig

from .io import resolve_task_output_paths, write_eval_result, write_raw_eval_result
from .metrics import result_from_vllm_raw_logs
from .model import describe_model_resolution, resolve_model_args
from .vllm import VLLMRunner, run_vllm_raw_eval


@dataclass(frozen=True)
class EvalRunOverrides:
    batch_size: int | str | None = None
    max_new_tokens: int | None = None
    limit: int | None = None
    output_dir: str | Path | None = None
    max_lora_rank: int | None = None


class EvalRunner:
    def __init__(
        self,
        eval_cfg: EvalConfig,
        *,
        model_name: str | None = None,
        backend_runner: VLLMRunner | None = None,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.model_name = model_name or eval_cfg.model_name
        self.backend_runner = backend_runner

    def _resolve_model_args(self, overrides: EvalRunOverrides) -> dict[str, Any]:
        return resolve_model_args(
            self.model_name,
            self.eval_cfg.model_name,
            backend=self.eval_cfg.backend,
            max_length=self.eval_cfg.max_seq_length,
            device=self.eval_cfg.device,
            attn_implementation=self.eval_cfg.attn_implementation,
            gpu_memory_utilization=self.eval_cfg.gpu_memory_utilization,
            max_lora_rank=overrides.max_lora_rank if overrides.max_lora_rank is not None else self.eval_cfg.max_lora_rank,
        )

    def run_task(
        self,
        task: EvalTaskConfig,
        overrides: EvalRunOverrides | None = None,
    ) -> dict[str, object]:
        resolved_overrides = overrides or EvalRunOverrides()
        batch_size = resolved_overrides.batch_size if resolved_overrides.batch_size is not None else self.eval_cfg.batch_size
        max_new_tokens = (
            resolved_overrides.max_new_tokens
            if resolved_overrides.max_new_tokens is not None
            else self.eval_cfg.max_new_tokens
        )
        output_dir = (
            Path(resolved_overrides.output_dir)
            if resolved_overrides.output_dir is not None
            else self.eval_cfg.output_dir
        )

        model_args = self._resolve_model_args(resolved_overrides)
        model_resolution = describe_model_resolution(self.model_name, self.eval_cfg.model_name, model_args)
        for key, value in sorted(model_resolution.items()):
            print(f"eval.model_resolution.{key}={value}")

        raw_eval_kwargs: dict[str, Any] = {
            "model_args": model_args,
            "dataset_path": task.dataset_path,
            "task_name": task.name,
            "batch_size": batch_size,
            "limit": resolved_overrides.limit,
            "max_gen_toks": int(max_new_tokens),
            "samples_per_problem": task.samples_per_problem,
            "sampling_temperature": task.sampling_temperature,
            "sampling_top_p": task.sampling_top_p,
        }
        if self.backend_runner is None:
            raw_result = run_vllm_raw_eval(**raw_eval_kwargs)
        else:
            raw_result = self.backend_runner.generate_raw_eval(**raw_eval_kwargs)

        result = result_from_vllm_raw_logs(
            raw_result,
            task_name=task.name,
            dataset_path=str(task.dataset_path),
            model_name=self.model_name,
        )
        final_output_path, raw_output_path = resolve_task_output_paths(
            task,
            output_dir=output_dir,
            model_name=self.model_name,
        )
        write_raw_eval_result(raw_output_path, raw_result)
        write_eval_result(final_output_path, result)
        return {
            "task_name": task.name,
            "raw_result": raw_result,
            "result": result,
            "result_path": str(final_output_path),
            "raw_result_path": str(raw_output_path),
            "model_resolution": model_resolution,
        }

    def run_tasks(
        self,
        tasks: list[EvalTaskConfig],
        overrides: EvalRunOverrides | None = None,
    ) -> list[dict[str, object]]:
        return [self.run_task(task, overrides=overrides) for task in tasks]
