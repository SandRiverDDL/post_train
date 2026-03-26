from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SFTTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    train_dataset: Path
    output_dir: Path
    seed: int = 42
    max_seq_length: int = 1024
    learning_rate: float = 1.0e-4
    epochs: float = 1.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    output_dir: Path = Path("outputs/eval")
    tasks: list["EvalTaskConfig"] = Field(default_factory=list)
    dataset_path: Path | None = None
    output_path: Path | None = None
    backend: Literal["hf", "vllm"] = "hf"
    batch_size: int | str = 1
    max_batch_size: int | None = None
    max_new_tokens: int = Field(default=512, ge=1)
    max_seq_length: int = Field(default=2048, ge=1)
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    gpu_memory_utilization: float = Field(default=0.7, gt=0.0, le=1.0)
    max_lora_rank: int | None = Field(default=None, ge=1)
    seed: int = 42

    @model_validator(mode="after")
    def _normalize_tasks(self) -> "EvalConfig":
        if self.tasks:
            return self
        if self.dataset_path is None:
            raise ValueError("eval 配置至少需要提供 tasks 或 dataset_path。")
        task_name = self.dataset_path.stem.replace("-", "_")
        self.tasks = [EvalTaskConfig(name=task_name, dataset_path=self.dataset_path, output_path=self.output_path)]
        return self


class EvalTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_path: Path
    output_path: Path | None = None
    raw_output_path: Path | None = None


class SimPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_name: str
    model_name_or_path: str
    train_dataset: Path
    output_dir: Path
    seed: int = 42
    max_length: int = 1024
    max_prompt_length: int = 768
    max_completion_length: int = 256
    learning_rate: float = 5.0e-7
    epochs: float = 1.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    beta: float = 2.0
    simpo_gamma: float = 1.0
    cpo_alpha: float = 0.0
    report_to: str = "none"


def load_yaml_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_sft_config(path: str | Path) -> SFTTrainConfig:
    return SFTTrainConfig.model_validate(load_yaml_config(path))


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig.model_validate(load_yaml_config(path))


def load_simpo_config(path: str | Path) -> SimPOConfig:
    return SimPOConfig.model_validate(load_yaml_config(path))
