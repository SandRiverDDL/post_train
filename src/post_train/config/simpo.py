from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
            Path("data/stage1/train.jsonl"),
            Path("data/stage1/dev200.jsonl"),
            Path("data/eval/math500_dev200.jsonl"),
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
