from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from post_train.answers import evaluate_prediction
from post_train.io import ensure_parent


def _as_list(values: Any, *, fallback: list[Any]) -> list[Any]:
    if values is None:
        return fallback
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values for _ in fallback]


class QuestionStatsRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._pending: list[dict[str, Any]] = []
        self._call_index = 0

    def record_correctness_batch(
        self,
        *,
        completions: list[Any],
        final_answer: list[Any],
        correctness: list[bool],
        parse_ok: list[bool],
        kwargs: dict[str, Any],
    ) -> None:
        fallback = list(range(len(completions)))
        ids = _as_list(kwargs.get("id"), fallback=fallback)
        questions = _as_list(kwargs.get("question"), fallback=["" for _ in completions])
        completion_ids = _as_list(kwargs.get("completion_ids"), fallback=[None for _ in completions])
        grouped: dict[str, dict[str, Any]] = {}

        for index, (sample_id, question, expected, is_correct, is_parse_ok, token_ids, completion) in enumerate(
            zip(ids, questions, final_answer, correctness, parse_ok, completion_ids, completions)
        ):
            key = str(sample_id)
            if key not in grouped:
                grouped[key] = {
                    "question_id": key,
                    "question": str(question),
                    "final_answer": str(expected),
                    "num_generations": 0,
                    "num_correct": 0,
                    "num_parse_ok": 0,
                    "completion_token_lengths": [],
                    "completion_char_lengths": [],
                    "first_batch_index": index,
                }
            row = grouped[key]
            row["num_generations"] += 1
            row["num_correct"] += int(is_correct)
            row["num_parse_ok"] += int(is_parse_ok)
            if token_ids is not None:
                row["completion_token_lengths"].append(len(token_ids))
            row["completion_char_lengths"].append(len(str(completion)))

        for row in grouped.values():
            token_lengths = row.pop("completion_token_lengths")
            char_lengths = row.pop("completion_char_lengths")
            row["correct_rate"] = row["num_correct"] / max(row["num_generations"], 1)
            row["parse_success_rate"] = row["num_parse_ok"] / max(row["num_generations"], 1)
            row["mean_completion_token_length"] = (
                sum(token_lengths) / len(token_lengths) if token_lengths else None
            )
            row["mean_completion_char_length"] = sum(char_lengths) / len(char_lengths) if char_lengths else 0.0
            row["reward_call_index"] = self._call_index
            self._pending.append(row)
        self._call_index += 1

    def flush(self, *, global_step: int | None = None, epoch: float | None = None) -> None:
        if not self._pending:
            return
        output_path = ensure_parent(self.path)
        with output_path.open("a", encoding="utf-8") as handle:
            for row in self._pending:
                payload = {
                    "global_step": global_step,
                    "epoch": epoch,
                    **row,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._pending.clear()


class QuestionStatsCallback:
    def __init__(self, recorder: QuestionStatsRecorder) -> None:
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            def __init__(self, outer) -> None:
                self.outer = outer

            def on_step_end(self, args, state, control, **kwargs):
                self.outer.recorder.flush(global_step=state.global_step, epoch=state.epoch)
                return control

            def on_train_end(self, args, state, control, **kwargs):
                self.outer.recorder.flush(global_step=state.global_step, epoch=state.epoch)
                return control

        self.recorder = recorder
        self.callback = _Impl(self)


def build_correctness_reward(cfg, *, stats_recorder: QuestionStatsRecorder | None = None) -> Callable[..., list[float]]:
    weight = cfg.correctness.weight

    def reward(completions, final_answer, **kwargs):
        rewards: list[float] = []
        correctness: list[bool] = []
        parse_ok: list[bool] = []
        for completion, expected_answer in zip(completions, final_answer):
            result = evaluate_prediction(str(completion), str(expected_answer), require_boxed=True)
            is_correct = bool(result["correct"])
            correctness.append(is_correct)
            parse_ok.append(bool(result["parse_ok"]))
            rewards.append(weight if is_correct else 0.0)
        if stats_recorder is not None:
            stats_recorder.record_correctness_batch(
                completions=list(completions),
                final_answer=list(final_answer),
                correctness=correctness,
                parse_ok=parse_ok,
                kwargs=kwargs,
            )
        return rewards

    reward.__name__ = "correctness_reward"
    return reward


def build_parse_penalty_reward(cfg) -> Callable[..., list[float]]:
    weight = cfg.parse_penalty.weight

    def reward(completions, final_answer, **kwargs):
        rewards: list[float] = []
        for completion, expected_answer in zip(completions, final_answer):
            result = evaluate_prediction(str(completion), str(expected_answer), require_boxed=True)
            rewards.append(weight if not bool(result["parse_ok"]) else 0.0)
        return rewards

    reward.__name__ = "parse_penalty_reward"
    return reward


def build_soft_overlong_reward(cfg, *, max_completion_length: int) -> Callable[..., list[float]]:
    weight = cfg.soft_overlong.weight
    cache_tokens = cfg.soft_overlong.cache_tokens
    soft_start = max(max_completion_length - cache_tokens, 0)

    def reward(completion_ids, **kwargs):
        rewards: list[float] = []
        for token_ids in completion_ids:
            length = len(token_ids)
            if length <= soft_start:
                rewards.append(0.0)
                continue
            overlong_ratio = min((length - soft_start) / max(cache_tokens, 1), 1.0)
            rewards.append(-weight * overlong_ratio)
        return rewards

    reward.__name__ = "soft_overlong_reward"
    return reward


def build_reward_functions(
    cfg,
    *,
    max_completion_length: int,
    stats_recorder: QuestionStatsRecorder | None = None,
) -> list[Callable[..., list[float]]]:
    reward_funcs: list[Callable[..., list[float]]] = []
    if cfg.correctness.enabled:
        reward_funcs.append(build_correctness_reward(cfg, stats_recorder=stats_recorder))
    if cfg.parse_penalty.enabled:
        reward_funcs.append(build_parse_penalty_reward(cfg))
    if cfg.soft_overlong.enabled:
        reward_funcs.append(build_soft_overlong_reward(cfg, max_completion_length=max_completion_length))
    if not reward_funcs:
        raise ValueError("至少需要启用一个 reward 组件。")
    return reward_funcs
