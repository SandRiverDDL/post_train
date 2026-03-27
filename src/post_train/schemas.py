from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SFTRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    solution: str
    final_answer: str
    meta: dict[str, Any] = Field(default_factory=dict)


class PreferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    chosen: str
    rejected: str
    meta: dict[str, Any] = Field(default_factory=dict)


class EvalPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    raw_generation: str
    predicted_answer: str
    expected_answer: str
    boxed: bool
    parse_ok: bool
    correct: bool
    sample_index: int | None = None
