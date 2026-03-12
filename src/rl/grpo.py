from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any

from datasets import Dataset
import torch
from torch import nn
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


def ensure_trl_model_compat(model: Any) -> Any:
    # `trl==0.24.0` 会直接访问 `warnings_issued` 和 `add_model_tags`。
    # 在当前 `peft/transformers/unsloth` 组合下，这些属性不一定存在，需要显式补齐。
    if not hasattr(model, "warnings_issued") or getattr(model, "warnings_issued") is None:
        model.warnings_issued = {}

    if not hasattr(model, "add_model_tags"):
        def _add_model_tags(_: Any) -> None:
            return None

        model.add_model_tags = _add_model_tags

    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        if not hasattr(base_model, "warnings_issued") or getattr(base_model, "warnings_issued") is None:
            base_model.warnings_issued = model.warnings_issued
        if not hasattr(base_model, "add_model_tags"):
            base_model.add_model_tags = model.add_model_tags

    return model


def model_uses_kbit_quantization(model: Any) -> bool:
    return bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False))


def find_unquantized_linear_modules(model: Any) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if type(module) is nn.Linear:
            names.append(name)
    return names


def stabilize_unquantized_kbit_linears(model: Any) -> tuple[Any, list[str]]:
    if not model_uses_kbit_quantization(model):
        return model, []

    unquantized_linear_names = find_unquantized_linear_modules(model)
    names_to_upcast = [name for name in unquantized_linear_names if not name.endswith("lm_head")]

    for name, module in model.named_modules():
        if name in names_to_upcast:
            module.to(dtype=torch.float32)

    return model, names_to_upcast


def align_lm_head_dtype(model: Any, target_dtype: torch.dtype) -> bool:
    lm_head = getattr(model, "lm_head", None)
    if type(lm_head) is not nn.Linear:
        return False
    if lm_head.weight.dtype == target_dtype:
        return False
    lm_head.to(dtype=target_dtype)
    return True


def ensure_trl_vllm_import_compat(*, use_vllm: bool) -> None:
    try:
        sampling_params = importlib.import_module("vllm.sampling_params")
    except ImportError:
        return

    if hasattr(sampling_params, "GuidedDecodingParams"):
        return

    structured_outputs_cls = getattr(sampling_params, "StructuredOutputsParams", None)
    if structured_outputs_cls is None:
        return

    if use_vllm:
        raise RuntimeError(
            "当前环境的 vLLM 不再提供 GuidedDecodingParams，而 trl==0.24.0 仍依赖该接口。"
            "如果要启用 use_vllm=true，请将 vllm 降到 trl 支持的版本，或同步升级 trl。"
        )

    # 这里只是为了让 `from trl import GRPOTrainer` 在 `use_vllm=false` 时可以正常导入。
    # 真正走 vLLM 路径时，trl 仍需要兼容的新接口。
    sampling_params.GuidedDecodingParams = structured_outputs_cls


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
