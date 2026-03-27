from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


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
