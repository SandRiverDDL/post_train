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
    for row in rows:
        row["prompt"] = build_grpo_prompt(str(row["question"]))
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


_REWARD_TOKENIZER = None
_LAST_REWARD_STATS: dict[str, float] = {}
_REWARD_CONFIG = {
    "correct": 1.0,
    "wrong": -0.2,
    "parse_fail": -0.2,
    "strict_boxed_bonus": 0.02,
    "length_coef": 1e-4,
    "use_relaxed_correctness": True,
}


def set_reward_tokenizer(tokenizer: Any) -> None:
    global _REWARD_TOKENIZER
    _REWARD_TOKENIZER = tokenizer


def set_reward_config(
    *,
    correct: float,
    wrong: float,
    parse_fail: float,
    strict_boxed_bonus: float,
    length_coef: float,
    use_relaxed_correctness: bool,
) -> None:
    global _REWARD_CONFIG
    _REWARD_CONFIG = {
        "correct": float(correct),
        "wrong": float(wrong),
        "parse_fail": float(parse_fail),
        "strict_boxed_bonus": float(strict_boxed_bonus),
        "length_coef": float(length_coef),
        "use_relaxed_correctness": bool(use_relaxed_correctness),
    }


def reward_token_length(completion: str) -> int:
    cleaned = clean_completion_for_protocol(completion)
    if _REWARD_TOKENIZER is None:
        return len(cleaned.split())
    return len(_REWARD_TOKENIZER(cleaned, add_special_tokens=False)["input_ids"])


def _compute_reward_components(completion: str, answer: str) -> tuple[float, dict[str, float]]:
    relaxed_result = evaluate_prediction(completion, answer, require_boxed=False)
    strict_result = evaluate_prediction(completion, answer, require_boxed=True)
    token_count = reward_token_length(completion)
    length_penalty = -_REWARD_CONFIG["length_coef"] * token_count
    extract_ok = bool(relaxed_result["extract_ok"])
    strict_boxed = bool(relaxed_result["format_ok"])
    relaxed_correct = bool(relaxed_result["correct"])
    strict_correct = bool(strict_result["correct"])
    relaxed_correct_flag = 1.0 if relaxed_correct else 0.0
    strict_correct_flag = 1.0 if strict_correct else 0.0
    strict_bonus = _REWARD_CONFIG["strict_boxed_bonus"] if extract_ok and strict_boxed else 0.0

    if _REWARD_CONFIG["use_relaxed_correctness"]:
        active_extract_ok = extract_ok
        active_correct = relaxed_correct
    else:
        active_extract_ok = bool(strict_result["extract_ok"])
        active_correct = strict_correct

    if not active_extract_ok:
        base_reward = _REWARD_CONFIG["parse_fail"]
        wrong = 0.0
        parse_fail = 1.0
    else:
        parse_fail = 0.0
        wrong = 0.0 if active_correct else 1.0
        base_reward = _REWARD_CONFIG["correct"] if active_correct else _REWARD_CONFIG["wrong"]

    reward = base_reward + strict_bonus + length_penalty
    stats = {
        "rewards/correct_rate": relaxed_correct_flag,
        "rewards/wrong_rate": wrong,
        "rewards/parse_fail_rate": parse_fail,
        "rewards/format_rate": 1.0 if strict_boxed else 0.0,
        "rewards/strict_boxed_rate": 1.0 if strict_boxed else 0.0,
        "rewards/relaxed_correct_rate": relaxed_correct_flag,
        "rewards/strict_correct_rate": strict_correct_flag,
        "rewards/mean_length_penalty": length_penalty,
    }
    return reward, stats


def combined_reward(
    prompts: list[str],
    completions: list[str],
    final_answer: list[str],
    **_: Any,
) -> list[float]:
    global _LAST_REWARD_STATS
    rewards: list[float] = []
    totals = {
        "rewards/correct_rate": 0.0,
        "rewards/wrong_rate": 0.0,
        "rewards/parse_fail_rate": 0.0,
        "rewards/format_rate": 0.0,
        "rewards/strict_boxed_rate": 0.0,
        "rewards/relaxed_correct_rate": 0.0,
        "rewards/strict_correct_rate": 0.0,
        "rewards/mean_length_penalty": 0.0,
    }
    count = 0
    for completion, answer in zip(completions, final_answer, strict=True):
        reward, stats = _compute_reward_components(completion, answer)
        rewards.append(reward)
        for key, value in stats.items():
            totals[key] += value
        count += 1
    if count:
        _LAST_REWARD_STATS = {key: value / count for key, value in totals.items()}
    else:
        _LAST_REWARD_STATS = {}
    return rewards


def last_reward_stats() -> dict[str, float]:
    return dict(_LAST_REWARD_STATS)


def reward_functions() -> list:
    return [combined_reward]


def default_reward_weights() -> list[float]:
    return [1.0]


def response_token_length(solution: str, tokenizer) -> int:
    cleaned = clean_completion_for_protocol(solution)
    return len(tokenizer(cleaned, add_special_tokens=False)["input_ids"])


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, output_path: str | Path, *, reset: bool = True, wandb_run: Any | None = None) -> None:
        self.output_path = ensure_parent(output_path)
        self.wandb_run = wandb_run
        if reset:
            self.output_path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update(logs)
        record.update(last_reward_stats())
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(record, step=state.global_step)
        return control
