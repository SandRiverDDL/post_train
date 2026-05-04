from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_path: Path
    output_path: Path | None = None
    raw_output_path: Path | None = None
    samples_per_problem: int = Field(default=1, ge=1)
    sampling_temperature: float | None = Field(default=None, ge=0.0)
    sampling_top_p: float | None = Field(default=None, gt=0.0, le=1.0)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    output_dir: Path = Path("outputs/eval")
    tasks: list[EvalTaskConfig] = Field(default_factory=list)
    dataset_path: Path | None = None
    output_path: Path | None = None
    backend: Literal["vllm"] = "vllm"
    batch_size: int | str = 1
    max_new_tokens: int = Field(default=512, ge=1)
    max_seq_length: int = Field(default=2048, ge=1)
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    gpu_memory_utilization: float = Field(default=0.7, gt=0.0, le=1.0)
    max_lora_rank: int | None = Field(default=None, ge=1)
    seed: int = 42
    prompt_style: Literal["default", "justrl_math"] = "justrl_math"
    use_chat_template: bool = False
    system_prompt: str | None = None
    assistant_prefill: str | None = None

    @model_validator(mode="after")
    def _normalize_tasks(self) -> "EvalConfig":
        if self.tasks:
            return self
        if self.dataset_path is None:
            raise ValueError("eval 配置至少需要提供 tasks 或 dataset_path。")
        task_name = self.dataset_path.stem.replace("-", "_")
        self.tasks = [EvalTaskConfig(name=task_name, dataset_path=self.dataset_path, output_path=self.output_path)]
        return self
