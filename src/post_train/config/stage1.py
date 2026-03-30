from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Stage1Math220kDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "qingy2024/OpenR1-Math-220k-Cleaned"
    dataset_split: str = "train"
    output_path: Path = Path("data/stage1_math220k_train.jsonl")
    report_path: Path = Path("data/stage1_math220k_train.report.json")
    tokenizer_name: str
    seed: int = 42
    sample_size: int = Field(default=2000, ge=1)
    min_solution_tokens: int = Field(default=0, ge=0)
    max_solution_tokens: int = Field(default=768, ge=1)
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _validate_token_bounds(self) -> "Stage1Math220kDataConfig":
        if self.min_solution_tokens >= self.max_solution_tokens:
            raise ValueError("min_solution_tokens 必须小于 max_solution_tokens。")
        return self


class Stage1RSRCandidatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_dataset: str = "UWNSL/MATH_training_split_short_cot"
    short_split: str = "train"
    long_dataset: str = "UWNSL/MATH_training_split_long_cot"
    long_split: str = "train"
    small_teacher_dataset: str = "UWNSL/MATH_training_split_distill_small_teacher"
    small_teacher_split: str = "train"
    large_teacher_dataset: str = "UWNSL/MATH_training_split_distill_large_teacher"
    large_teacher_split: str = "train"
    student_model_name: str
    tokenizer_name: str | None = None
    output_path: Path = Path("data/stage1_rsr_candidates.jsonl")
    report_path: Path = Path("data/stage1_rsr_candidates.report.json")
    unmatched_preview_path: Path = Path("data/stage1_rsr_candidates.unmatched.preview.jsonl")
    seed: int = 42
    batch_size: int = Field(default=2, ge=1)
    max_seq_length: int = Field(default=1536, ge=8)
    rank_clip_r: int = Field(default=100, ge=1)
    unmatched_preview_count: int = Field(default=50, ge=0)
    drop_truncated: bool = True
    load_in_4bit: bool = True
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _fill_tokenizer_name(self) -> "Stage1RSRCandidatesConfig":
        if not self.tokenizer_name:
            self.tokenizer_name = self.student_model_name
        return self


class Stage1RSRSelectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: Path = Path("data/stage1_rsr_candidates.jsonl")
    output_path: Path = Path("data/stage1_rsr_selected_train.jsonl")
    report_path: Path = Path("data/stage1_rsr_selected_train.report.json")
    seed: int = 42
    sample_size: int = Field(default=2000, ge=1)
    min_solution_tokens: int = Field(default=0, ge=0)
    max_solution_tokens: int | None = Field(default=None, ge=1)
    allowed_sources: list[str] | None = None
    drop_truncated_trajectories: bool = True
    prefer_lower_rsr: bool = True

    @model_validator(mode="after")
    def _validate_token_bounds(self) -> "Stage1RSRSelectConfig":
        if self.max_solution_tokens is not None and self.min_solution_tokens >= self.max_solution_tokens:
            raise ValueError("min_solution_tokens 必须小于 max_solution_tokens。")
        if self.allowed_sources is not None:
            allowed = {"short", "long", "small", "large"}
            invalid = [source for source in self.allowed_sources if source not in allowed]
            if invalid:
                raise ValueError(f"allowed_sources 包含未知来源：{invalid}")
            self.allowed_sources = list(dict.fromkeys(self.allowed_sources))
        return self
