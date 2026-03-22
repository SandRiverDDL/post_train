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
    prompt_version: str = "v1"


class EvalConfig(BaseModel):
    """评测脚本使用的最小配置。"""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    eval_dataset: Path
    test_dataset: Path | None = None
    max_seq_length: int = 768
    eval_max_new_tokens: int = Field(default=256, ge=1)
    eval_backend: str = "vllm"
    eval_attn_implementation: str = "sdpa"
    eval_gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=1.0)
    prompt_version: str = "v1"


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
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    load_in_4bit: bool = True
    fast_inference: bool = False
    gpu_memory_utilization: float = Field(default=0.6, gt=0.0, le=1.0)
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    num_generations: int = 4
    num_iterations: int = 1
    beta: float = 0.0
    loss_type: str = "dapo"
    reward_correct: float = 1.0
    reward_wrong: float = -0.2
    reward_parse_fail: float = -0.2
    reward_strict_boxed_bonus: float = 0.02
    reward_length_coef: float = 1e-4
    reward_use_relaxed_correctness: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    generation_kwargs: dict = Field(default_factory=dict)
    mask_truncated_completions: bool = True
    top_entropy_quantile: float = Field(default=1.0, gt=0.0, le=1.0)
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 50
    save_total_limit: int | None = None
    log_completions: bool = True
    num_completions_to_print: int = 2
    report_to: str = "none"
    wandb_mode: str = "offline"
    wandb_project: str = "qwen3-math-posttrain"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = Field(default_factory=list)
    prompt_version: str = "v1"


def load_config(path: str | Path) -> SFTConfig:
    return SFTConfig.model_validate(_load_yaml_with_base(path))


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig.model_validate(_load_yaml_with_base(path))


def load_grpo_config(path: str | Path) -> GRPOTrainConfig:
    return GRPOTrainConfig.model_validate(_load_yaml_with_base(path))


def _load_yaml_with_base(path: str | Path, *, seen: set[Path] | None = None) -> dict:
    config_path = Path(path).resolve()
    active_seen = seen or set()
    if config_path in active_seen:
        raise ValueError(f"检测到循环配置继承: {config_path}")
    active_seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base_value = raw.pop("_base_", None)
    if base_value is None:
        return raw
    base_path = (config_path.parent / base_value).resolve()
    base_raw = _load_yaml_with_base(base_path, seen=active_seen)
    return _deep_merge_dict(base_raw, raw)


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged
