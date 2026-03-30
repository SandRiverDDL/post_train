from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OnPolicyDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_name: str = "round1"
    query_file_path: Path | None = None
    query_dataset: str = "UWNSL/MATH_training_split_short_cot"
    query_split: str = "train"
    query_config_name: str | None = None
    query_source: str = "UWNSL/MATH_training_split_short_cot"
    exclude_paths: list[Path] = Field(
        default_factory=lambda: [
            Path("data/stage1_dev200_5000.jsonl"),
            Path("data/eval/global_dev_math500_150.jsonl"),
            Path("data/eval/gsm8k_test.jsonl"),
            Path("data/eval/math500_test.jsonl"),
        ]
    )
    query_limit: int | None = Field(default=None, ge=1)
    responses_per_query: int = Field(default=4, ge=2)
    generation_model: str = "outputs/stage1_sft_5000/checkpoint-200"
    base_model_name: str
    max_model_length: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=768, ge=1)
    max_completion_tokens: int = Field(default=512, ge=1)
    min_retained_count: int = Field(default=32, ge=1)
    primary_selector: Literal["any_correct_shortest", "mixed_only_shortest"] = "any_correct_shortest"
    gpu_memory_utilization: float = Field(default=0.7, gt=0.0, le=1.0)
    max_lora_rank: int | None = Field(default=None, ge=1)
    seed: int = 42
    query_output_path: Path = Path("data/on_policy/query_pool.jsonl")
    raw_samples_output_path: Path = Path("data/on_policy/raw_samples.jsonl")
    retained_output_path: Path = Path("data/on_policy/train.jsonl")
    report_path: Path = Path("data/on_policy/train.report.json")
    cache_dir: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_query_count(cls, raw_value):
        if not isinstance(raw_value, dict):
            return raw_value
        value = dict(raw_value)
        legacy_query_count = value.pop("query_count", None)
        if "query_limit" not in value and legacy_query_count is not None:
            value["query_limit"] = legacy_query_count
        return value


class OnPolicyLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_model: str = "outputs/stage1_sft_5000/checkpoint-200"
    base_on_policy_data_config: Path = Path("configs/on_policy/data.yaml")
    base_sft_config: Path = Path("configs/on_policy/sft.yaml")
    eval_config: Path = Path("configs/eval/default.yaml")
    stop_dataset: Path = Path("data/eval/global_dev_math500_150.jsonl")
    max_rounds: int = Field(default=10, ge=1)
    round_query_count: int = Field(default=512, ge=1)
    query_strategy: Literal["uniform_epoch", "mixed_bootstrap_candidate", "candidate_random_mix"] = "uniform_epoch"
    bootstrap_all_correct_raw_samples: Path | None = None
    candidate_query_file: Path | None = None
    bootstrap_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    candidate_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    random_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    candidate_freeze_all_correct_hits: int = Field(default=2, ge=1)
    train_selector: Literal["primary_retained", "mixed_plus_anchor"] = "mixed_plus_anchor"
    anchor_dataset_path: Path | None = Path("data/stage1_train_5000.jsonl")
    anchor_share: float = Field(default=0.25, gt=0.0, lt=1.0)
    train_epochs: float = Field(default=1.0, gt=0.0)
    train_learning_rate: float = Field(default=5.0e-6, gt=0.0)
    advance_teacher_on_improvement_only: bool = False
    checkpoint_selection_enabled: bool = True
    checkpoint_target_count: int = Field(default=4, ge=1)
    checkpoint_save_total_limit: int = Field(default=4, ge=1)
    patience: int = Field(default=3, ge=1)
    min_delta: float = Field(default=0.0, ge=0.0)
    round_base_dir: Path = Path("outputs/on_policy_loop")
    data_base_dir: Path = Path("data/on_policy_loop")
    backend: Literal["vllm"] | None = None
    batch_size: int | str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_strategy(self) -> "OnPolicyLoopConfig":
        ratio_sum = self.bootstrap_ratio + self.candidate_ratio
        if self.query_strategy == "mixed_bootstrap_candidate" and abs(ratio_sum - 1.0) > 1.0e-6:
            raise ValueError("bootstrap_ratio 与 candidate_ratio 之和必须等于 1.0。")
        if self.query_strategy == "candidate_random_mix":
            ratio_sum = self.candidate_ratio + self.random_ratio
            if abs(ratio_sum - 1.0) > 1.0e-6:
                raise ValueError("candidate_ratio 与 random_ratio 之和必须等于 1.0。")
        if self.query_strategy == "mixed_bootstrap_candidate":
            if self.bootstrap_all_correct_raw_samples is None:
                raise ValueError("mixed_bootstrap_candidate 需要提供 bootstrap_all_correct_raw_samples。")
            if self.candidate_query_file is None:
                raise ValueError("mixed_bootstrap_candidate 需要提供 candidate_query_file。")
        if self.query_strategy == "candidate_random_mix" and self.candidate_query_file is None:
            raise ValueError("candidate_random_mix 需要提供 candidate_query_file。")
        if self.train_selector == "mixed_plus_anchor" and self.anchor_dataset_path is None:
            raise ValueError("mixed_plus_anchor 需要提供 anchor_dataset_path。")
        return self
