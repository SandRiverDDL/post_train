from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    backend: Literal["unsloth", "trl_peft"] = "unsloth"
    loss_mode: Literal["standard", "opsft"] = "standard"
    profit_enabled: bool = False
    profit_threshold: float = Field(default=0.1, gt=0.0, lt=1.0)
    save_strategy: Literal["epoch", "steps", "no"] = "epoch"
    save_steps: int | None = Field(default=None, ge=1)
    save_total_limit: int | None = Field(default=None, ge=1)
    export_final_model: bool = True

    @model_validator(mode="after")
    def _validate_loss_mode(self) -> "SFTTrainConfig":
        if self.loss_mode == "opsft" and self.profit_enabled:
            raise ValueError("opsft 与 profit_enabled 不能同时开启。")
        return self
