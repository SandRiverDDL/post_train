from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from lm_eval import simple_evaluate
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager

from rl.answers import evaluate_prediction
from rl.data import format_protocol_prompt


def strict_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction_text = results[0] if results else ""
    result = evaluate_prediction(
        prediction_text,
        str(doc.get("final_answer", "")),
        require_boxed=True,
    )
    return {
        "format_success": float(bool(result["format_ok"])),
        "parse_success": float(bool(result["extract_ok"])),
        "normalized_accuracy": float(bool(result["correct"])),
    }


def relaxed_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction_text = results[0] if results else ""
    result = evaluate_prediction(
        prediction_text,
        str(doc.get("final_answer", "")),
        require_boxed=False,
    )
    return {
        "parse_success": float(bool(result["extract_ok"])),
        "normalized_accuracy": float(bool(result["correct"])),
    }


def build_task_config(dataset_path: str | Path, task_name: str, *, strict: bool) -> dict[str, Any]:
    return {
        "task": task_name,
        "task_alias": task_name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"train": str(dataset_path)}},
        "test_split": "train",
        "output_type": "generate_until",
        "doc_to_text": lambda doc: format_protocol_prompt(str(doc["question"])),
        "doc_to_target": "final_answer",
        "generation_kwargs": {
            "until": ["</s>", "<|im_end|>"],
            "do_sample": False,
        },
        "process_results": strict_process_results if strict else relaxed_process_results,
        "metric_list": [
            {"metric": "format_success", "aggregation": "mean", "higher_is_better": True},
            {"metric": "parse_success", "aggregation": "mean", "higher_is_better": True},
            {"metric": "normalized_accuracy", "aggregation": "mean", "higher_is_better": True},
        ]
        if strict
        else [
            {"metric": "parse_success", "aggregation": "mean", "higher_is_better": True},
            {"metric": "normalized_accuracy", "aggregation": "mean", "higher_is_better": True},
        ],
    }


def build_local_task(dataset_path: str | Path, task_name: str, *, strict: bool) -> ConfigurableTask:
    task_config = build_task_config(dataset_path, task_name, strict=strict)
    return ConfigurableTask(config=task_config)


def resolve_benchmark_task(dataset_path: str | Path) -> tuple[str | None, list[int] | None]:
    path = Path(dataset_path)
    if path.name == "gsm8k_dev200.jsonl":
        return "gsm8k_cot_zeroshot", _read_sample_ids(path)
    if path.name == "math500_test.jsonl":
        return "hendrycks_math500", _read_sample_ids(path)
    return None, None


def resolve_result_task_name(dataset_path: str | Path, fallback_task_name: str) -> str:
    native_task_name, _ = resolve_benchmark_task(dataset_path)
    return native_task_name or fallback_task_name


def _read_sample_ids(path: Path) -> list[int]:
    sample_ids: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_ids.append(int(row["id"]))
    return sample_ids


def _limit_sample_ids(sample_ids: list[int] | None, limit: int | None) -> list[int] | None:
    if sample_ids is None or limit is None:
        return sample_ids
    return sample_ids[:limit]


def resolve_model_args(
    requested_model: str | None,
    base_model: str,
    *,
    backend: str,
    max_length: int,
    device: str,
    attn_implementation: str,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    target = requested_model or base_model
    if backend == "vllm":
        model_args: dict[str, Any] = {
            "pretrained": base_model if _looks_like_adapter_dir(target) else target,
            "dtype": "auto",
            "trust_remote_code": True,
            "max_length": max_length,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if _looks_like_adapter_dir(target):
            model_args["lora_local_path"] = target
        return model_args

    model_args = {
        "pretrained": base_model if _looks_like_adapter_dir(target) else target,
        "dtype": "auto",
        "device": device,
        "trust_remote_code": True,
        "max_length": max_length,
        "attn_implementation": attn_implementation,
    }
    if _looks_like_adapter_dir(target):
        model_args["peft"] = target
    return model_args


def _looks_like_adapter_dir(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def run_harness_eval(
    *,
    backend: str,
    model_args: dict[str, Any],
    dataset_path: str | Path,
    task_name: str,
    batch_size: int | str,
    max_batch_size: int | None,
    limit: int | None,
    strict: bool,
    max_gen_toks: int,
) -> dict[str, Any]:
    if backend == "vllm":
        _ensure_vllm_available()
    native_task_name, sample_ids = resolve_benchmark_task(dataset_path)
    if native_task_name:
        sample_ids = _limit_sample_ids(sample_ids, limit)
        return simple_evaluate(
            model=backend,
            model_args=model_args,
            tasks=[native_task_name],
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            limit=None,
            samples={native_task_name: sample_ids} if sample_ids else None,
            log_samples=True,
            num_fewshot=None,
            gen_kwargs={"max_gen_toks": max_gen_toks, "do_sample": False},
        )
    task_manager = TaskManager(include_defaults=False)
    local_task = build_local_task(dataset_path, task_name, strict=strict)
    _register_local_task(task_manager, task_name, dataset_path)
    return simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=[local_task],
        task_manager=task_manager,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        limit=limit,
        log_samples=True,
        num_fewshot=0,
        gen_kwargs={"max_gen_toks": max_gen_toks, "do_sample": False},
    )


def _ensure_vllm_available() -> None:
    if importlib.util.find_spec("vllm") is not None:
        return

    candidate_paths = [
        Path("/home/chy/code/active/template/.venv/lib/python3.11/site-packages"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            if importlib.util.find_spec("vllm") is not None:
                return

    raise ModuleNotFoundError(
        "当前环境未找到 vllm；请先安装 `vllm>=0.15.0`，或提供一个包含 vllm 的 site-packages 路径。"
    )


def _register_local_task(task_manager: TaskManager, task_name: str, dataset_path: str | Path) -> None:
    dataset_ref = str(dataset_path)
    task_manager.task_index[task_name] = {
        "type": "task",
        "yaml_path": dataset_ref,
        "task": None,
    }


def _extract_preview_raw_generation(sample: dict[str, Any]) -> str:
    resps = sample.get("resps", [])
    if resps:
        first = resps[0]
        if isinstance(first, list) and first:
            return str(first[0])
        if isinstance(first, str):
            return first

    filtered = sample.get("filtered_resps", [])
    if filtered:
        first = filtered[0]
        if isinstance(first, list) and first:
            return str(first[0])
        if isinstance(first, str):
            return first

    return "[invalid]"


def _extract_preview_gold_answer(sample: dict[str, Any], doc: dict[str, Any]) -> str:
    if doc.get("final_answer"):
        return str(doc["final_answer"])
    if sample.get("target"):
        return str(sample["target"])
    if doc.get("answer"):
        return str(doc["answer"])
    return ""


def preview_logged_samples(result: dict[str, Any], task_name: str, *, count: int = 3) -> None:
    samples = result.get("samples", {}).get(task_name, [])
    for sample in samples[:count]:
        doc = sample.get("doc", {})
        raw_generation = _extract_preview_raw_generation(sample)
        print("=" * 80)
        print(f"id: {doc.get('id', sample.get('doc_id', 'unknown'))}")
        print(doc.get("question", ""))
        print("--- raw generation ---")
        print(raw_generation)
        print("--- gold answer ---")
        print(_extract_preview_gold_answer(sample, doc))
        metric_keys = [key for key in ("format_success", "parse_success", "normalized_accuracy") if key in sample]
        if metric_keys:
            print("--- metrics ---")
            print(json.dumps({key: sample[key] for key in metric_keys}, ensure_ascii=False))


def write_result(path: str | Path, result: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
