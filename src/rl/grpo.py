from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import TrainerCallback

from rl.answers import evaluate_prediction
from rl.data import clean_completion_for_protocol, format_protocol_prompt
from rl.io import ensure_parent, read_jsonl


def build_grpo_prompt(question: str) -> str:
    return format_protocol_prompt(question)


def build_grpo_dataset(path: str | Path) -> Dataset:
    rows = read_jsonl(path)
    return Dataset.from_list(rows)


def build_grpo_record(
    row: dict[str, Any],
    *,
    prompt: str | None = None,
    response_tokens: int | None = None,
    reference_solution: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": str(row["id"]),
        "question": row["question"],
        "prompt": prompt or build_grpo_prompt(str(row["question"])),
        "final_answer": str(row["final_answer"]),
        "source": str(row.get("source", "numinamath")),
        "problem_type": str(row.get("problem_type", "Other")),
    }
    if response_tokens is not None:
        record["response_tokens"] = int(response_tokens)
    if reference_solution is not None:
        record["reference_solution"] = reference_solution
    return record


def correctness_reward(
    prompts: list[str],
    completions: list[str],
    final_answer: list[str],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, answer in zip(completions, final_answer, strict=True):
        result = evaluate_prediction(completion, answer, require_boxed=True)
        rewards.append(1.0 if bool(result["correct"]) else 0.0)
    return rewards


def parse_reward(
    prompts: list[str],
    completions: list[str],
    final_answer: list[str],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, answer in zip(completions, final_answer, strict=True):
        result = evaluate_prediction(completion, answer, require_boxed=True)
        rewards.append(1.0 if bool(result["extract_ok"]) else -1.0)
    return rewards


def format_reward(
    prompts: list[str],
    completions: list[str],
    final_answer: list[str],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, answer in zip(completions, final_answer, strict=True):
        result = evaluate_prediction(completion, answer, require_boxed=True)
        rewards.append(1.0 if bool(result["format_ok"]) else -1.0)
    return rewards


def reward_functions() -> list:
    return [correctness_reward, parse_reward, format_reward]


def default_reward_weights() -> list[float]:
    return [1.0, 0.02, 0.02]


def response_token_length(solution: str, tokenizer) -> int:
    cleaned = clean_completion_for_protocol(solution)
    return len(tokenizer(cleaned, add_special_tokens=False)["input_ids"])


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = ensure_parent(output_path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update(logs)
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return control
