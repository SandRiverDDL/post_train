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
    logging_steps: int = Field(default=5, ge=1)
    group_by_length: bool = True
    save_strategy: Literal["epoch", "steps"] = "epoch"
    save_steps: int | None = Field(default=None, ge=1)
    save_total_limit: int | None = Field(default=None, ge=1)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    output_dir: Path = Path("outputs/eval")
    tasks: list["EvalTaskConfig"] = Field(default_factory=list)
    dataset_path: Path | None = None
    output_path: Path | None = None
    runner: Literal["vllm_raw", "lm_eval"] = "vllm_raw"
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
    logging_steps: int = Field(default=5, ge=1)
    save_strategy: Literal["epoch", "steps"] = "steps"
    save_steps: int | None = Field(default=25, ge=1)
    save_total_limit: int | None = Field(default=3, ge=1)
    resume_from_checkpoint: str | None = None
    beta: float = 2.0
    simpo_gamma: float = 1.0
    cpo_alpha: float = 0.0
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["auto", "bfloat16", "float16"] = "auto"
    report_to: str = "none"


class SimPODataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_dataset: str = "UWNSL/MATH_training_split_short_cot"
    query_split: str = "train"
    query_config_name: str | None = None
    query_source: str = "UWNSL/MATH_training_split_short_cot"
    exclude_paths: list[Path] = Field(
        default_factory=lambda: [
            Path("data/stage1_train.jsonl"),
            Path("data/stage1_dev200.jsonl"),
            Path("data/eval/global_dev_math500_150.jsonl"),
            Path("data/eval/gsm8k_test.jsonl"),
            Path("data/eval/math500_test.jsonl"),
        ]
    )
    query_count: int = Field(default=800, ge=1)
    responses_per_query: int = Field(default=4, ge=2)
    target_pair_count: int = Field(default=500, ge=1)
    pilot_pair_count: int = Field(default=150, ge=1)
    generation_model: str = "outputs/stage1_sft/checkpoint-50"
    base_model_name: str
    max_model_length: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=768, ge=1)
    gpu_memory_utilization: float = Field(default=0.7, gt=0.0, le=1.0)
    max_lora_rank: int | None = Field(default=None, ge=1)
    seed: int = 42
    min_length_gap_tokens: int = Field(default=30, ge=0)
    min_length_gap_ratio: float = Field(default=0.2, ge=0.0)
    query_output_path: Path = Path("data/simpo/query_pool.jsonl")
    raw_samples_output_path: Path = Path("data/simpo/raw_samples.jsonl")
    pair_output_path: Path = Path("data/simpo/train.jsonl")
    pilot_output_path: Path = Path("data/simpo/train.pilot.jsonl")
    report_path: Path = Path("data/simpo/train.report.json")
    cache_dir: str | None = None


class Stage2DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage1_input: Path
    math220k_dataset: str = "qingy2024/OpenR1-Math-220k-Cleaned"
    math220k_split: str = "train"
    output_path: Path = Path("data/stage2_train.jsonl")
    report_path: Path = Path("data/stage2_train.report.json")
    intermediate_dir: Path = Path("outputs/stage2_data")
    tokenizer_name: str
    seed: int = 42
    total_size: int = 2000
    stage1_random_quota: int = 400
    math220k_short_quota: int = 1200
    math220k_long_quota: int = 400
    short_max_tokens: int = 768
    long_max_tokens: int = 1380
    cache_dir: str | None = None
    save_intermediate: bool = False
    exact_dedup: bool = False
    near_dedup: bool = False
    decontam: bool = False

    @model_validator(mode="after")
    def _validate_quotas(self) -> "Stage2DataConfig":
        expected_total = (
            self.stage1_random_quota
            + self.math220k_short_quota
            + self.math220k_long_quota
        )
        if expected_total != self.total_size:
            raise ValueError(
                "stage2 配额之和必须等于 total_size。"
            )
        if self.short_max_tokens >= self.long_max_tokens:
            raise ValueError("short_max_tokens 必须小于 long_max_tokens。")
        return self


def load_yaml_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_sft_config(path: str | Path) -> SFTTrainConfig:
    return SFTTrainConfig.model_validate(load_yaml_config(path))


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig.model_validate(load_yaml_config(path))


def load_simpo_config(path: str | Path) -> SimPOConfig:
    return SimPOConfig.model_validate(load_yaml_config(path))


def load_simpo_data_config(path: str | Path) -> SimPODataConfig:
    return SimPODataConfig.model_validate(load_yaml_config(path))


def load_stage2_data_config(path: str | Path) -> Stage2DataConfig:
    return Stage2DataConfig.model_validate(load_yaml_config(path))
