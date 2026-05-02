from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SFTTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    adapter_base_model: str | None = None
    train_dataset: Path
    output_dir: Path
    seed: int = 42
    max_seq_length: int = 1024
    learning_rate: float = 1.0e-4
    lr_scheduler_type: Literal["linear", "cosine", "constant", "constant_with_warmup"] = "linear"
    max_steps: int = Field(default=-1, ge=-1)
    epochs: float = 1.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    logging_steps: int = Field(default=5, ge=1)
    group_by_length: bool = True
    backend: Literal["unsloth", "trl_peft"] = "unsloth"
    loss_mode: Literal["standard", "opsft", "lightning_opd"] = "standard"
    profit_enabled: bool = False
    profit_threshold: float = Field(default=0.1, gt=0.0, lt=1.0)
    distill_top_k: int = Field(default=1, ge=1)
    topk_kd_weight: float = Field(default=0.0, ge=0.0)
    opd_weight: float = Field(default=1.0, ge=0.0)
    save_strategy: Literal["epoch", "steps", "no"] = "epoch"
    save_steps: int | None = Field(default=None, ge=1)
    save_total_limit: int | None = Field(default=None, ge=1)
    export_final_model: bool = True

    @model_validator(mode="after")
    def _validate_loss_mode(self) -> "SFTTrainConfig":
        if self.loss_mode == "opsft" and self.profit_enabled:
            raise ValueError("opsft 与 profit_enabled 不能同时开启。")
        if self.loss_mode == "lightning_opd" and self.profit_enabled:
            raise ValueError("lightning_opd 与 profit_enabled 不能同时开启。")
        if self.loss_mode != "lightning_opd" and (
            self.distill_top_k != 1 or self.topk_kd_weight != 0.0 or self.opd_weight != 1.0
        ):
            raise ValueError("distill_top_k/topk_kd_weight/opd_weight 只允许用于 lightning_opd。")
        return self
