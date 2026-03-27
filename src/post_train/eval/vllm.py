from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from post_train.io import read_jsonl
from post_train.prompts import build_eval_prompt


def _looks_like_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def resolve_model_args(
    requested_model: str | None,
    base_model: str,
    *,
    backend: str,
    max_length: int,
    device: str,
    attn_implementation: str,
    gpu_memory_utilization: float,
    max_lora_rank: int | None = None,
) -> dict[str, Any]:
    if backend != "vllm":
        raise ValueError("当前评测只支持 backend=vllm。")
    target = requested_model or base_model
    model_args: dict[str, Any] = {
        "pretrained": base_model if _looks_like_adapter_dir(target) else target,
        "dtype": "auto",
        "trust_remote_code": True,
        "max_length": max_length,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if _looks_like_adapter_dir(target):
        model_args["lora_local_path"] = target
        if max_lora_rank is not None:
            model_args["max_lora_rank"] = max_lora_rank
    return model_args


def build_task_name(dataset_path: str | Path) -> str:
    return Path(dataset_path).stem.replace("-", "_")


def run_vllm_raw_eval(
    *,
    model_args: dict[str, Any],
    dataset_path: str | Path,
    task_name: str,
    batch_size: int | str,
    limit: int | None,
    max_gen_toks: int,
    samples_per_problem: int = 1,
    sampling_temperature: float | None = None,
    sampling_top_p: float | None = None,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if batch_size == "auto":
        raise ValueError("当前评测不支持 batch_size=auto，请显式传整数 batch_size。")

    rows = read_jsonl(dataset_path)
    if limit is not None:
        rows = rows[:limit]

    prompts = [build_eval_prompt(str(row["question"])) for row in rows]
    llm_kwargs = dict(model_args)
    model_name = str(llm_kwargs.pop("pretrained"))
    max_length = llm_kwargs.pop("max_length", None)
    lora_path = llm_kwargs.pop("lora_local_path", None)
    max_lora_rank = llm_kwargs.pop("max_lora_rank", None)
    if max_length is not None:
        llm_kwargs["max_model_len"] = max_length
    if lora_path is not None:
        llm_kwargs["enable_lora"] = True
    if max_lora_rank is not None:
        llm_kwargs["max_lora_rank"] = max_lora_rank

    llm = LLM(model=model_name, **llm_kwargs)
    if samples_per_problem > 1:
        temperature = 0.6 if sampling_temperature is None else sampling_temperature
        top_p = 0.95 if sampling_top_p is None else sampling_top_p
        if temperature <= 0.0:
            raise ValueError("samples_per_problem > 1 时 temperature 必须大于 0。")
    else:
        temperature = 0.0
        top_p = 1.0
    sampling_params = SamplingParams(
        n=samples_per_problem,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_gen_toks,
        stop=["</s>", "<|im_end|>"],
    )
    lora_request = None
    if lora_path is not None:
        lora_request = LoRARequest("eval_adapter", 1, lora_path)

    started_at = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
        lora_request=lora_request,
    )
    total_seconds = time.perf_counter() - started_at

    samples: list[dict[str, Any]] = []
    for row, prompt, output in zip(rows, prompts, outputs, strict=True):
        generations = [candidate.text for candidate in output.outputs]
        samples.append(
            {
                "id": str(row.get("id", "")),
                "doc": row,
                "prompt": prompt,
                "resps": generations,
                "metrics": ["raw_count"],
            }
        )

    return {
        "runner": "vllm_raw",
        "task_name": task_name,
        "samples_per_problem": samples_per_problem,
        "samples": {task_name: samples},
        "timing": {"total_seconds": total_seconds},
    }
