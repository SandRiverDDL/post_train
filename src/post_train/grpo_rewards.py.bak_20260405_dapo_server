from __future__ import annotations

from collections.abc import Callable

from post_train.answers import evaluate_prediction


def build_correctness_reward(cfg) -> Callable[..., list[float]]:
    weight = cfg.correctness.weight

    def reward(completions, final_answer, **kwargs):
        rewards: list[float] = []
        for completion, expected_answer in zip(completions, final_answer):
            result = evaluate_prediction(str(completion), str(expected_answer), require_boxed=True)
            rewards.append(weight if bool(result["correct"]) else 0.0)
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


def build_reward_functions(cfg) -> list[Callable[..., list[float]]]:
    reward_funcs: list[Callable[..., list[float]]] = []
    if cfg.correctness.enabled:
        reward_funcs.append(build_correctness_reward(cfg))
    if cfg.parse_penalty.enabled:
        reward_funcs.append(build_parse_penalty_reward(cfg))
    if not reward_funcs:
        raise ValueError("至少需要启用一个 reward 组件。")
    return reward_funcs
