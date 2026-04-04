from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GRPODataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rd211_dataset: str = "rd211/Big-Math-RL-Verified-Filtered"
    dataset_split: str = "train"
    solve_rate_lower: float = Field(default=0.25, ge=0.0, le=1.0)
    solve_rate_upper: float = Field(default=0.5, ge=0.0, le=1.0)
    anchor_dataset_path: Path = Path("data/on_policy_loop/query_strategy/candidate_pool.jsonl")
    anchor_share: float = Field(default=0.3, ge=0.0, lt=1.0)
    seed: int = 42
    output_path: Path = Path("data/grpo/train.jsonl")
    report_path: Path = Path("data/grpo/train.report.json")
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _validate_ranges(self) -> "GRPODataConfig":
        if self.solve_rate_lower >= self.solve_rate_upper:
            raise ValueError("solve_rate_lower 必须严格小于 solve_rate_upper。")
        if not self.dataset_split.strip():
            raise ValueError("dataset_split 不能为空。")
        return self


class RewardComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    weight: float = 1.0


class GRPORewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: RewardComponentConfig = RewardComponentConfig(weight=1.0)
    parse_penalty: RewardComponentConfig = RewardComponentConfig(weight=-0.5)


class GRPOTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_name: str
    model_name_or_path: str
    train_dataset: Path
    output_dir: Path
    reward_config: Path = Path("configs/grpo/reward.yaml")
    seed: int = 42
    max_train_samples: int | None = Field(default=None, ge=1)
    train_subset_seed: int = 42
    train_subset_mode: Literal["fixed_random", "head"] = "fixed_random"
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    max_prompt_length: int = Field(default=768, ge=1)
    max_completion_length: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=1.0e-6, gt=0.0)
    epochs: float = Field(default=1.0, gt=0.0)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    logging_steps: int = Field(default=5, ge=1)
    save_strategy: Literal["epoch", "steps"] = "steps"
    save_steps: int | None = Field(default=25, ge=1)
    save_total_limit: int | None = Field(default=3, ge=1)
    resume_from_checkpoint: str | None = None
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["auto", "bfloat16", "float16"] = "auto"
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0)
    num_generations: int = Field(default=4, ge=2)
    temperature: float = Field(default=0.8, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    loss_type: Literal["grpo", "bnpo", "dr_grpo", "dapo"] = "dr_grpo"
    epsilon: float = Field(default=0.2, gt=0.0)
    epsilon_high: float | None = Field(default=None, gt=0.0)
    beta: float = Field(default=0.0, ge=0.0)
    scale_rewards: bool = False
    importance_sampling_level: Literal["token", "sequence"] = "sequence"
    mask_truncated_completions: bool = True
    use_vllm: bool = False
    vllm_mode: Literal["server", "colocate"] = "colocate"
    vllm_gpu_memory_utilization: float = Field(default=0.25, gt=0.0, le=1.0)
    vllm_tensor_parallel_size: int = Field(default=1, ge=1)
    report_to: Literal["none", "wandb"] = "wandb"
    wandb_project: str = "post-train-grpo"
    wandb_run_name: str | None = None
    wandb_debug_metrics: bool = False
    wandb_smoothing_window: int = Field(default=5, ge=1)
