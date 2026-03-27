from __future__ import annotations

import math
from typing import Any, Iterable

from post_train.answers import evaluate_prediction
from post_train.io import read_jsonl
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
            "runner": "vllm_raw",
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


def result_from_vllm_raw_logs(
    raw_result: dict[str, Any],
    *,
    task_name: str,
    dataset_path: str,
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
