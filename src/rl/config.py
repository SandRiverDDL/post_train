from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SFTConfig(BaseModel):
    """SFT 和评测共用的最小配置。"""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    seed: int = 42
    train_dataset: Path
    eval_dataset: Path
    test_dataset: Path | None = None
    output_dir: Path = Path("outputs/sft")
    max_seq_length: int = 768
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    eval_max_new_tokens: int = Field(default=256, ge=1)
    eval_backend: str = "vllm"
    eval_attn_implementation: str = "sdpa"
    eval_gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=1.0)


class GRPOTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_name: str
    cold_start_model: Path
    train_dataset: Path
    output_dir: Path = Path("outputs/grpo")
    seed: int = 42
    max_seq_length: int = 1024
    max_prompt_length: int = 512
    max_completion_length: int = Field(default=384, ge=1)
    learning_rate: float = 1e-6
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    num_generations: int = 4
    num_iterations: int = 1
    beta: float = 0.0
    loss_type: str = "dapo"
    mask_truncated_completions: bool = True
    top_entropy_quantile: float = Field(default=1.0, gt=0.0, le=1.0)
    logging_steps: int = 10
    log_completions: bool = True
    num_completions_to_print: int = 2
    use_unsloth: bool = True
    use_vllm: bool = False


def load_config(path: str | Path) -> SFTConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return SFTConfig.model_validate(raw)


def load_grpo_config(path: str | Path) -> GRPOTrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return GRPOTrainConfig.model_validate(raw)
