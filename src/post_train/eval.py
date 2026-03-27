from __future__ import annotations

import json
import math
import time
from inspect import getsource
from pathlib import Path
from typing import Any, Iterable

from lm_eval import simple_evaluate
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager

from post_train.answers import evaluate_prediction
from post_train.config import EvalConfig, EvalTaskConfig
from post_train.io import ensure_parent, read_jsonl
from post_train.prompts import build_eval_prompt
from post_train.schemas import EvalPrediction


def rate_stderr(rate: float, sample_count: int) -> float:
    if sample_count <= 0:
        return 0.0
    return math.sqrt(rate * (1.0 - rate) / sample_count)


def mean_stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _output_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def build_eval_result(
    rows: Iterable[dict[str, Any]],
    generations: Iterable[str | list[str]],
    *,
    model_name: str,
    dataset_name: str,
    runner: str | None = None,
    total_seconds: float | None = None,
) -> dict[str, Any]:
    row_list = list(rows)
    generation_groups: list[list[str]] = []
    for generation in generations:
        if isinstance(generation, str):
            generation_groups.append([generation])
            continue
        generation_groups.append([str(item) for item in generation])
    if len(row_list) != len(generation_groups):
        raise ValueError("rows 与 generations 数量不一致。")

    predictions: list[dict[str, Any]] = []
    boxed_total = 0.0
    parse_total = 0.0
    correct_total = 0.0
    output_token_total = 0
    pass_at_1_values: list[float] = []

    for row, problem_generations in zip(row_list, generation_groups, strict=True):
        if not problem_generations:
            problem_generations = [""]
        problem_correct_total = 0.0
        for sample_index, generation in enumerate(problem_generations, start=1):
            verdict = evaluate_prediction(
                generation,
                str(row.get("final_answer", "")),
                require_boxed=True,
            )
            boxed = bool(verdict["boxed"])
            parse_ok = bool(verdict["parse_ok"])
            correct = bool(verdict["correct"])
            boxed_total += float(boxed)
            parse_total += float(parse_ok)
            correct_total += float(correct)
            problem_correct_total += float(correct)
            output_token_total += _output_tokens(generation)
            prediction = EvalPrediction(
                id=str(row.get("id", "")),
                question=str(row.get("question", "")),
                raw_generation=generation,
                predicted_answer=str(verdict.get("parsed_answer", "")),
                expected_answer=str(row.get("final_answer", "")),
                boxed=boxed,
                parse_ok=parse_ok,
                correct=correct,
                sample_index=sample_index,
            )
            predictions.append(prediction.model_dump())
        pass_at_1_values.append(problem_correct_total / len(problem_generations))

    sample_count = len(row_list)
    total_generations = len(predictions)
    boxed_rate = boxed_total / total_generations if total_generations else 0.0
    parse_success_rate = parse_total / total_generations if total_generations else 0.0
    pass_at_1 = sum(pass_at_1_values) / sample_count if sample_count else 0.0
    avg_output_tokens = output_token_total / total_generations if total_generations else 0.0
    samples_per_problem = (total_generations / sample_count) if sample_count else 0.0
    samples_per_second = (sample_count / total_seconds) if total_seconds and total_seconds > 0 else 0.0
    generations_per_second = (total_generations / total_seconds) if total_seconds and total_seconds > 0 else 0.0
    return {
        "metrics": {
            "model": model_name,
            "dataset": dataset_name,
            "runner": runner or "",
            "samples": sample_count,
            "total_generations": total_generations,
            "samples_per_problem": samples_per_problem,
            "boxed_rate": boxed_rate,
            "boxed_rate_stderr": rate_stderr(boxed_rate, total_generations),
            "parse_success_rate": parse_success_rate,
            "parse_success_rate_stderr": rate_stderr(parse_success_rate, total_generations),
            "pass_at_1": pass_at_1,
            "pass_at_1_stderr": mean_stderr(pass_at_1_values),
            "normalized_accuracy": pass_at_1,
            "normalized_accuracy_stderr": mean_stderr(pass_at_1_values),
            "avg_output_tokens": avg_output_tokens,
            "total_seconds": total_seconds or 0.0,
            "samples_per_second": samples_per_second,
            "generations_per_second": generations_per_second,
        },
        "predictions": predictions,
    }


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
            if max_lora_rank is not None:
                model_args["max_lora_rank"] = max_lora_rank
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


def normalize_batch_settings(
    batch_size: int | str,
    max_batch_size: int | None,
) -> tuple[int | str, int | None]:
    if batch_size == "auto":
        return batch_size, max_batch_size
    return batch_size, None


def build_task_name(dataset_path: str | Path) -> str:
    return Path(dataset_path).stem.replace("-", "_")


def process_results_stub(_doc: dict[str, Any], _results: list[str]) -> dict[str, float]:
    return {"raw_count": 1.0}


def build_local_task(dataset_path: str | Path, *, task_name: str | None = None) -> ConfigurableTask:
    resolved_task_name = task_name or build_task_name(dataset_path)
    task_config = {
        "task": resolved_task_name,
        "task_alias": resolved_task_name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"train": str(dataset_path)}},
        "test_split": "train",
        "output_type": "generate_until",
        "doc_to_text": lambda doc: build_eval_prompt(str(doc["question"])),
        "doc_to_target": "final_answer",
        "generation_kwargs": {
            "until": ["</s>", "<|im_end|>"],
            "do_sample": False,
        },
        "process_results": process_results_stub,
        "metric_list": [
            {"metric": "raw_count", "aggregation": "mean", "higher_is_better": True},
        ],
    }
    return ConfigurableTask(config=task_config)


def run_harness_eval(
    *,
    backend: str,
    model_args: dict[str, Any],
    dataset_path: str | Path,
    task_name: str,
    batch_size: int | str,
    max_batch_size: int | None,
    limit: int | None,
    max_gen_toks: int,
    samples_per_problem: int = 1,
    sampling_temperature: float | None = None,
    sampling_top_p: float | None = None,
) -> dict[str, Any]:
    if samples_per_problem > 1:
        raise ValueError("lm_eval runner 暂不支持 samples_per_problem > 1；请改用 vllm_raw。")
    batch_size, max_batch_size = normalize_batch_settings(batch_size, max_batch_size)
    task_manager = TaskManager(include_defaults=False)
    task_manager.task_index[task_name] = {
        "type": "task",
        "yaml_path": str(dataset_path),
        "task": None,
    }
    started_at = time.perf_counter()
    result = simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=[build_local_task(dataset_path, task_name=task_name)],
        task_manager=task_manager,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        limit=limit,
        log_samples=True,
        num_fewshot=0,
        gen_kwargs={"max_gen_toks": max_gen_toks, "do_sample": False},
    )
    result["timing"] = {"total_seconds": time.perf_counter() - started_at}
    result["runner"] = "lm_eval"
    return result


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
        raise ValueError("vllm_raw runner 不支持 batch_size=auto，请显式传整数 batch_size。")

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


def _extract_generation(sample: dict[str, Any]) -> str:
    responses = sample.get("resps", [])
    if responses:
        first = responses[0]
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
    return ""


def result_from_harness_logs(
    harness_result: dict[str, Any],
    *,
    task_name: str,
    dataset_path: str | Path,
    model_name: str,
) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    samples = harness_result.get("samples", {}).get(task_name, [])
    generations = [_extract_generation(sample) for sample in samples]
    total_seconds = float(harness_result.get("timing", {}).get("total_seconds", 0.0))
    return build_eval_result(
        rows[: len(generations)],
        generations,
        model_name=model_name,
        dataset_name=str(dataset_path),
        runner=str(harness_result.get("runner", "lm_eval")),
        total_seconds=total_seconds,
    )


def result_from_vllm_raw_logs(
    raw_result: dict[str, Any],
    *,
    task_name: str,
    dataset_path: str | Path,
    model_name: str,
) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    samples = raw_result.get("samples", {}).get(task_name, [])
    generations = [
        [str(generation) for generation in sample.get("resps", [])]
        if sample.get("resps")
        else [""]
        for sample in samples
    ]
    total_seconds = float(raw_result.get("timing", {}).get("total_seconds", 0.0))
    return build_eval_result(
        rows[: len(generations)],
        generations,
        model_name=model_name,
        dataset_name=str(dataset_path),
        runner=str(raw_result.get("runner", "vllm_raw")),
        total_seconds=total_seconds,
    )


def preview_logged_samples(result: dict[str, Any], *, count: int = 3) -> str:
    previews: list[str] = []
    for row in result.get("predictions", [])[:count]:
        previews.append(
            "\n".join(
                [
                    "=" * 80,
                    f"id: {row['id']}",
                    row["question"],
                    "--- raw generation ---",
                    row["raw_generation"],
                    "--- gold answer ---",
                    row["expected_answer"],
                ]
            )
        )
    return "\n".join(previews)


def summarize_metrics_for_console(metrics: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": metrics.get("model"),
        "dataset": metrics.get("dataset"),
        "samples": metrics.get("samples"),
    }
    if "pass_at_1" in metrics:
        summary["pass_at_1"] = metrics.get("pass_at_1")
        summary["pass_at_1_stderr"] = metrics.get("pass_at_1_stderr")
    else:
        summary["normalized_accuracy"] = metrics.get("normalized_accuracy")
        summary["normalized_accuracy_stderr"] = metrics.get("normalized_accuracy_stderr")
    summary["boxed_rate"] = metrics.get("boxed_rate")
    summary["parse_success_rate"] = metrics.get("parse_success_rate")
    summary["avg_output_tokens"] = metrics.get("avg_output_tokens")
    return summary


def write_eval_result(path: str | Path, result: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return output_path


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if callable(value):
        try:
            return getsource(value)
        except (OSError, TypeError):
            return str(value)
    return str(value)


def write_raw_eval_result(path: str | Path, result: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, ensure_ascii=False, indent=2)
    return output_path


def resolve_eval_tasks(
    cfg: EvalConfig,
    *,
    requested_tasks: list[str] | None = None,
    dataset_override: str | None = None,
    output_override: str | None = None,
) -> list[EvalTaskConfig]:
    if dataset_override is not None:
        dataset_path = Path(dataset_override)
        task_name = dataset_path.stem.replace("-", "_")
        return [
            EvalTaskConfig(
                name=task_name,
                dataset_path=dataset_path,
                output_path=Path(output_override) if output_override else None,
            )
        ]

    if not requested_tasks:
        return list(cfg.tasks)

    selected = {name for name in requested_tasks}
    tasks = [task for task in cfg.tasks if task.name in selected]
    if not tasks:
        raise ValueError(f"未匹配到任何任务：{requested_tasks}")
    return tasks


def resolve_task_output_paths(task: EvalTaskConfig, *, output_dir: Path, runner: str) -> tuple[Path, Path]:
    default_output = output_dir / f"{task.name}.{runner}.json"
    final_output = task.output_path or default_output
    if task.output_path is not None and final_output.suffix == ".json":
        final_output = final_output.with_name(f"{final_output.stem}.{runner}.json")
    raw_output = task.raw_output_path or final_output.with_suffix(".raw.json")
    return final_output, raw_output
