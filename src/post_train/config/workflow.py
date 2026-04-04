from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["grpo"] = "grpo"
    train_config: Path
    eval_config: Path
    dev_dataset: Path = Path("data/eval/math500_dev200.jsonl")
    benchmark_tasks: list[str] = Field(default_factory=lambda: ["math500", "gsm8k"])
    baseline_policy: Literal["reuse_or_eval"] = "reuse_or_eval"
    failure_policy: Literal["stop"] = "stop"
    workflow_summary_name: str = "workflow_summary.json"
    workflow_failure_name: str = "workflow_failure.json"

