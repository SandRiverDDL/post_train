from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
            raise ValueError("stage2 配额之和必须等于 total_size。")
        if self.short_max_tokens >= self.long_max_tokens:
            raise ValueError("short_max_tokens 必须小于 long_max_tokens。")
        return self


class Stage2MixLongDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "UWNSL/Mix-Long_long_0.2_short_0.8"
    dataset_split: str = "train"
    output_path: Path = Path("data/stage2_mix_long_train.jsonl")
    report_path: Path = Path("data/stage2_mix_long_train.report.json")
    tokenizer_name: str
    seed: int = 42
    max_solution_tokens: int = Field(default=2048, ge=1)
    sample_size: int | None = Field(default=None, ge=1)
    cache_dir: str | None = None


class Stage2HendrycksLongDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_dataset_path: Path = Path("data/stage1_train_5000.jsonl")
    hendrycks_dataset: str = "EleutherAI/hendrycks_math"
    hendrycks_split: str = "train"
    hendrycks_config_name: str | None = None
    hendrycks_config_names: list[str] = Field(
        default_factory=lambda: [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
    )
    long_cot_dataset: str = "UWNSL/MATH_training_split_long_cot"
    long_cot_split: str = "train"
    long_cot_config_name: str | None = None
    output_path: Path = Path("data/stage2_hendrycks_long_train.jsonl")
    report_path: Path = Path("data/stage2_hendrycks_long_train.report.json")
    unmatched_preview_path: Path = Path("data/stage2_hendrycks_long_unmatched.preview.jsonl")
    tokenizer_name: str
    seed: int = 42
    total_size: int = Field(default=2000, ge=1)
    long_ratio: int = Field(default=1, ge=1)
    short_ratio: int = Field(default=2, ge=1)
    levels: list[int] = Field(default_factory=lambda: [4, 5])
    max_solution_tokens: int = Field(default=4096, ge=1)
    unmatched_preview_count: int = Field(default=50, ge=0)
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _validate_ratios(self) -> "Stage2HendrycksLongDataConfig":
        if not self.levels:
            raise ValueError("levels 不能为空。")
        if self.hendrycks_config_name and not self.hendrycks_config_names:
            self.hendrycks_config_names = [self.hendrycks_config_name]
        if not self.hendrycks_config_names:
            raise ValueError("hendrycks_config_names 不能为空。")
        return self
