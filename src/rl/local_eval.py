from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from rl.answers import evaluate_prediction
from rl.data import format_protocol_prompt
from rl.harness_tasks import resolve_model_args
from rl.io import read_jsonl


def rate_stderr(rate: float, sample_count: int) -> float:
    if sample_count <= 0:
        return 0.0
    return math.sqrt(rate * (1.0 - rate) / sample_count)


def build_official_result(
    rows: list[dict[str, Any]],
    generations: list[str],
    *,
    model_name: str,
    dataset_path: Path,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    format_total = 0.0
    parse_total = 0.0
    correct_total = 0.0

    for row, generation in zip(rows, generations, strict=True):
        verdict = evaluate_prediction(
            generation,
            str(row.get("final_answer", "")),
            require_boxed=True,
        )
        format_ok = bool(verdict["format_ok"])
        extract_ok = bool(verdict["extract_ok"])
        correct = bool(verdict["correct"])
        format_total += float(format_ok)
        parse_total += float(extract_ok)
        correct_total += float(correct)
        predictions.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "raw_generation": generation,
                "predicted_answer": str(verdict.get("parsed_answer", "")),
                "expected_answer": str(row.get("final_answer", "")),
                "format_ok": format_ok,
                "extract_ok": extract_ok,
                "correct": correct,
            }
        )

    sample_count = len(rows)
    format_rate = format_total / sample_count if sample_count else 0.0
    parse_rate = parse_total / sample_count if sample_count else 0.0
    accuracy = correct_total / sample_count if sample_count else 0.0
    return {
        "metrics": {
            "model": model_name,
            "dataset": str(dataset_path),
            "samples": sample_count,
            "format_success_rate": format_rate,
            "format_success_stderr": rate_stderr(format_rate, sample_count),
            "parse_success_rate": parse_rate,
            "parse_success_stderr": rate_stderr(parse_rate, sample_count),
            "normalized_accuracy": accuracy,
            "normalized_accuracy_stderr": rate_stderr(accuracy, sample_count),
        },
        "predictions": predictions,
    }


def _normalize_batch_size(batch_size: int | str) -> int:
    if isinstance(batch_size, int):
        return batch_size
    if batch_size == "auto":
        raise ValueError("official 模式暂不支持 batch_size=auto，请传显式整数。")
    return int(batch_size)


def _preview_predictions(result: dict[str, Any], *, count: int = 3) -> None:
    for row in result.get("predictions", [])[:count]:
        print("=" * 80)
        print(f"id: {row['id']}")
        print(row["question"])
        print("--- raw generation ---")
        print(row["raw_generation"])
        print("--- gold answer ---")
        print(row["expected_answer"])
        print("--- metrics ---")
        print(
            {
                "format_ok": row["format_ok"],
                "extract_ok": row["extract_ok"],
                "correct": row["correct"],
            }
        )


def _generate_with_vllm(
    prompts: list[str],
    *,
    model_args: dict[str, Any],
    base_model_name: str,
    max_length: int,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora_path = model_args.get("lora_local_path")
    llm = LLM(
        model=model_args["pretrained"],
        trust_remote_code=bool(model_args.get("trust_remote_code", True)),
        dtype=str(model_args.get("dtype", "auto")),
        gpu_memory_utilization=float(model_args.get("gpu_memory_utilization", 0.7)),
        max_model_len=max_length,
        seed=seed,
        enable_lora=bool(lora_path),
    )
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("eval", 1, lora_path, base_model_name=base_model_name)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=["</s>", "<|im_end|>"],
    )
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )
    return [output.outputs[0].text if output.outputs else "" for output in outputs]


def _generate_with_hf(
    prompts: list[str],
    *,
    model_args: dict[str, Any],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pretrained = model_args["pretrained"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=bool(model_args.get("trust_remote_code", True)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        pretrained,
        torch_dtype="auto",
        trust_remote_code=bool(model_args.get("trust_remote_code", True)),
        attn_implementation=model_args.get("attn_implementation"),
    )
    if model_args.get("peft"):
        model = PeftModel.from_pretrained(model, model_args["peft"])
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    generations: list[str] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for index, output in enumerate(outputs):
                completion_ids = output[int(prompt_lengths[index]) :]
                generations.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return generations


def run_official_eval(
    *,
    backend: str,
    requested_model: str | None,
    base_model: str,
    dataset_path: Path,
    limit: int | None,
    batch_size: int | str,
    max_new_tokens: int,
    max_length: int,
    device: str,
    attn_implementation: str,
    gpu_memory_utilization: float,
    prompt_version: str = "v1",
    seed: int = 42,
    preview_count: int = 3,
) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    if limit is not None:
        rows = rows[:limit]
    prompts = [format_protocol_prompt(str(row["question"]), prompt_version=prompt_version) for row in rows]
    model_name = requested_model or base_model
    model_args = resolve_model_args(
        requested_model,
        base_model,
        backend=backend,
        max_length=max_length,
        device=device,
        attn_implementation=attn_implementation,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if backend == "vllm":
        generations = _generate_with_vllm(
            prompts,
            model_args=model_args,
            base_model_name=base_model,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
    elif backend == "hf":
        generations = _generate_with_hf(
            prompts,
            model_args=model_args,
            max_new_tokens=max_new_tokens,
            batch_size=_normalize_batch_size(batch_size),
        )
    else:
        raise ValueError(f"不支持的 backend: {backend}")

    result = build_official_result(
        rows,
        generations,
        model_name=model_name,
        dataset_path=dataset_path,
    )
    if preview_count > 0:
        _preview_predictions(result, count=preview_count)
    return result
