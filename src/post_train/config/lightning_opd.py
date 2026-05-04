from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from post_train.lightning_opd.data import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_SOURCE,
    DEFAULT_STUDENT_BASE_MODEL,
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
)


class LightningOPDDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = DEFAULT_OUTPUT_DIR
    prompt_source: Path = DEFAULT_PROMPT_SOURCE
    sample_size: int = Field(default=1000, ge=1)
    seed: int = 42
    num_shards: int = Field(default=2, ge=1)
    student_model: str = str(DEFAULT_STUDENT_MODEL)
    student_base_model: str | None = str(DEFAULT_STUDENT_BASE_MODEL)
    teacher_model: str = str(DEFAULT_TEACHER_MODEL)
    tokenizer_name: str | None = None
    max_new_tokens: int = Field(default=2048, ge=1)
    max_model_len: int = Field(default=2560, ge=1)
    temperature: float = Field(default=0.6, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=32, ge=0)
    teacher_batch_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=1.0)
    teacher_load_in_4bit: bool = True
    gpu0: int = Field(default=3, ge=0)
    gpu1: int = Field(default=4, ge=0)
