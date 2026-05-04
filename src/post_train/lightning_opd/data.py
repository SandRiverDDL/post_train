from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from post_train.answers import evaluate_prediction
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.prompts import build_eval_prompt
from post_train.rollout.common import select_shard_rows, shard_suffix


DEFAULT_PROMPT_SOURCE = Path("data/on_policy_loop/query_strategy/candidate_pool.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/lightning_opd/candidate")
DEFAULT_STUDENT_BASE_MODEL = Path(
    "/mnt/dataY/fsw/cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
)
DEFAULT_STUDENT_MODEL = Path("outputs/stage1_mix_long_sft/checkpoint-300")
DEFAULT_TEACHER_MODEL = Path(
    "/mnt/dataY/fsw/cache/huggingface/hub/models--hbx--JustRL-DeepSeek-1.5B/snapshots/0637e4096c789c67f9eecbe8355e0bdeddede1c2"
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]


def _length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def build_lightning_opd_prompts(
    *,
    prompt_source: str | Path = DEFAULT_PROMPT_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_size: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    rows = read_jsonl(prompt_source)
    if sample_size > len(rows):
        raise ValueError(f"prompt 数量不足：需要 {sample_size}，实际只有 {len(rows)}")
    sampled = random.Random(seed).sample(rows, sample_size)
    prompt_rows: list[dict[str, Any]] = []
    for index, row in enumerate(sampled):
        meta = dict(row.get("meta", {}))
        meta.update({"prompt_source_path": str(prompt_source), "global_query_index": index})
        prompt_rows.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "final_answer": str(row.get("final_answer", "")),
                "prompt": str(row.get("prompt") or build_eval_prompt(str(row.get("question", "")))),
                "meta": meta,
            }
        )
    output_path = Path(output_dir) / "prompts.jsonl"
    write_jsonl(output_path, prompt_rows)
    report = {
        "prompt_source": str(prompt_source),
        "raw_prompt_count": len(rows),
        "sample_size": sample_size,
        "seed": seed,
        "outputs": {"prompts": str(output_path)},
    }
    _write_json(Path(output_dir) / "report.json", report)
    return report


def _is_lora_adapter(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def _read_lora_rank(path: str | Path) -> int | None:
    adapter_config_path = Path(path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    rank = payload.get("r")
    return int(rank) if rank is not None else None


def _sampled_logprobs_and_topk(
    logits: "torch.Tensor",
    target_ids: "torch.Tensor",
    *,
    top_k: int,
) -> tuple[list[float], list[list[int]], list[list[float]]]:
    import torch

    if logits.ndim != 2:
        raise ValueError(f"logits 必须是 [tokens, vocab]，实际 shape={tuple(logits.shape)}")
    if target_ids.ndim != 1 or target_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"target_ids 必须是 [tokens] 且与 logits 对齐，实际 target={tuple(target_ids.shape)} logits={tuple(logits.shape)}"
        )

    # sampled-token OPD 只需要目标 token 的 logprob；不要 materialize [tokens, vocab] 的 log_softmax。
    work_logits = logits.float()
    log_denominators = torch.logsumexp(work_logits, dim=-1)
    target_logits = work_logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    token_logprobs = (target_logits - log_denominators).tolist()

    if top_k <= 0:
        return [float(value) for value in token_logprobs], [[] for _ in token_logprobs], [[] for _ in token_logprobs]

    k = min(int(top_k), int(work_logits.shape[-1]))
    top_values, top_indices = torch.topk(work_logits, k=k, dim=-1)
    top_logprobs = top_values - log_denominators.unsqueeze(-1)
    return (
        [float(value) for value in token_logprobs],
        [[int(token_id) for token_id in row] for row in top_indices.tolist()],
        [[float(value) for value in row] for row in top_logprobs.tolist()],
    )


def _rollout_with_vllm(
    prompt_rows: list[dict[str, Any]],
    *,
    student_model: str | Path,
    student_base_model: str | Path | None,
    max_new_tokens: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
    trust_remote_code: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_path = str(student_model)
    llm_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    lora_request = None
    if _is_lora_adapter(student_model):
        if student_base_model is None:
            raise ValueError("student_model 是 LoRA adapter 时必须提供 student_base_model。")
        model_path = str(student_base_model)
        llm_kwargs["enable_lora"] = True
        lora_rank = _read_lora_rank(student_model)
        if lora_rank is not None:
            llm_kwargs["max_lora_rank"] = lora_rank
        lora_request = LoRARequest("lightning_opd_student_adapter", 1, str(student_model))

    llm = LLM(model=model_path, **llm_kwargs)
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=["</s>", "<|im_end|>"],
    )
    prompts = [str(row["prompt"]) for row in prompt_rows]
    started_at = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True, lora_request=lora_request)
    total_seconds = time.perf_counter() - started_at
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del llm
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    raw_rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    boxed_count = 0
    parse_ok_count = 0
    correct_count = 0
    for row, output in zip(prompt_rows, outputs, strict=True):
        text = output.outputs[0].text if output.outputs else ""
        verdict = evaluate_prediction(text, str(row.get("final_answer", "")), require_boxed=True)
        output_tokens = len(text.strip().split()) if text.strip() else 0
        lengths.append(output_tokens)
        boxed_count += int(bool(verdict["boxed"]))
        parse_ok_count += int(bool(verdict["parse_ok"]))
        correct_count += int(bool(verdict["correct"]))
        raw_rows.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "final_answer": str(row.get("final_answer", "")),
                "prompt": str(row.get("prompt", "")),
                "response": text,
                "metrics": {
                    "boxed": bool(verdict["boxed"]),
                    "parse_ok": bool(verdict["parse_ok"]),
                    "correct": bool(verdict["correct"]),
                    "predicted_answer": str(verdict.get("parsed_answer", "")),
                    "output_tokens_whitespace": output_tokens,
                },
                "meta": dict(row.get("meta", {})),
            }
        )
    return raw_rows, {
        "student_model": str(student_model),
        "student_base_model": str(student_base_model) if student_base_model is not None else None,
        "prompt_count": len(prompt_rows),
        "total_seconds": total_seconds,
        "responses_per_second": len(prompt_rows) / total_seconds if total_seconds > 0 else 0.0,
        "boxed_count": boxed_count,
        "parse_ok_count": parse_ok_count,
        "correct_count": correct_count,
        "output_tokens_whitespace": _length_stats(lengths),
    }


def _token_logprobs_and_topk(
    raw_rows: list[dict[str, Any]],
    *,
    teacher_model: str | Path,
    tokenizer_name: str | Path,
    top_k: int,
    max_model_len: int,
    batch_size: int,
    load_in_4bit: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else None
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(str(teacher_model), **model_kwargs)
    model.eval()
    device = next(model.parameters()).device

    encoded_rows: list[dict[str, Any]] = []
    dropped_truncated = 0
    response_lengths: list[int] = []
    for row in raw_rows:
        prompt_ids = tokenizer(str(row.get("prompt", "")), add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(str(row.get("response", "")), add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids
        response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        if len(input_ids) > max_model_len:
            dropped_truncated += 1
            continue
        encoded_rows.append(
            {
                "row": row,
                "input_ids": input_ids,
                "response_mask": response_mask,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids),
            }
        )
        response_lengths.append(len(response_ids))

    output_rows: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started_at = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(encoded_rows), batch_size):
            batch = encoded_rows[start : start + batch_size]
            max_len = max(len(item["input_ids"]) for item in batch)
            input_tensor = torch.tensor(
                [item["input_ids"] + [pad_id] * (max_len - len(item["input_ids"])) for item in batch],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.tensor(
                [[1] * len(item["input_ids"]) + [0] * (max_len - len(item["input_ids"])) for item in batch],
                dtype=torch.long,
                device=device,
            )
            logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits
            for batch_index, item in enumerate(batch):
                input_ids = item["input_ids"]
                response_positions = [
                    pos
                    for pos, mask_value in enumerate(item["response_mask"])
                    if mask_value and pos > 0
                ]
                if response_positions:
                    position_tensor = torch.tensor([pos - 1 for pos in response_positions], dtype=torch.long, device=device)
                    target_tensor = torch.tensor(
                        [int(input_ids[pos]) for pos in response_positions],
                        dtype=torch.long,
                        device=device,
                    )
                    token_logits = logits[batch_index].index_select(dim=0, index=position_tensor)
                    token_logprobs, topk_token_ids, topk_logprobs = _sampled_logprobs_and_topk(
                        token_logits,
                        target_tensor,
                        top_k=top_k,
                    )
                else:
                    token_logprobs = []
                    topk_token_ids = []
                    topk_logprobs = []
                row = item["row"]
                meta = dict(row.get("meta", {}))
                meta.update(
                    {
                        "teacher_model": str(teacher_model),
                        "tokenizer_name": str(tokenizer_name),
                        "top_k": top_k,
                        "prompt_tokens": item["prompt_tokens"],
                        "response_tokens": item["response_tokens"],
                    }
                )
                output_rows.append(
                    {
                        "id": str(row.get("id", "")),
                        "question": str(row.get("question", "")),
                        "final_answer": str(row.get("final_answer", "")),
                        "prompt": str(row.get("prompt", "")),
                        "response": str(row.get("response", "")),
                        "input_ids": input_ids,
                        "response_mask": item["response_mask"],
                        "teacher_token_logprobs": token_logprobs,
                        "teacher_topk_token_ids": topk_token_ids,
                        "teacher_topk_logprobs": topk_logprobs,
                        "metrics": dict(row.get("metrics", {})),
                        "meta": meta,
                    }
                )
    total_seconds = time.perf_counter() - started_at
    return output_rows, {
        "teacher_model": str(teacher_model),
        "tokenizer_name": str(tokenizer_name),
        "top_k": top_k,
        "batch_size": batch_size,
        "raw_rows": len(raw_rows),
        "kept_rows": len(output_rows),
        "dropped_truncated": dropped_truncated,
        "total_seconds": total_seconds,
        "response_tokens": _length_stats(response_lengths),
    }


def run_lightning_opd_shard(
    *,
    output_dir: str | Path,
    num_shards: int,
    shard_index: int,
    student_model: str | Path = DEFAULT_STUDENT_MODEL,
    student_base_model: str | Path | None = DEFAULT_STUDENT_BASE_MODEL,
    teacher_model: str | Path = DEFAULT_TEACHER_MODEL,
    tokenizer_name: str | Path | None = None,
    max_new_tokens: int = 2048,
    max_model_len: int = 2560,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 32,
    teacher_batch_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    teacher_load_in_4bit: bool = True,
) -> dict[str, Any]:
    base = Path(output_dir)
    prompts_path = base / "prompts.jsonl"
    prompt_rows = read_jsonl(prompts_path)
    suffix = shard_suffix(shard_index, num_shards)
    shard_dir = base / "shards" / suffix
    shard_rows = select_shard_rows(prompt_rows, num_shards=num_shards, shard_index=shard_index)
    raw_rows, rollout_report = _rollout_with_vllm(
        shard_rows,
        student_model=student_model,
        student_base_model=student_base_model,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        temperature=temperature,
        top_p=top_p,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    raw_path = shard_dir / "raw_rollouts.jsonl"
    write_jsonl(raw_path, raw_rows)
    topk_rows, teacher_report = _token_logprobs_and_topk(
        raw_rows,
        teacher_model=teacher_model,
        tokenizer_name=tokenizer_name or teacher_model,
        top_k=top_k,
        max_model_len=max_model_len,
        batch_size=teacher_batch_size,
        load_in_4bit=teacher_load_in_4bit,
    )
    topk_path = shard_dir / "teacher_topk.jsonl"
    write_jsonl(topk_path, topk_rows)
    return {
        "shard": suffix,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "prompt_count": len(shard_rows),
        "outputs": {"raw_rollouts": str(raw_path), "teacher_topk": str(topk_path)},
        "rollout": rollout_report,
        "teacher": teacher_report,
    }


def run_lightning_opd_teacher_shard(
    *,
    output_dir: str | Path,
    raw_rollout_dir: str | Path | None = None,
    num_shards: int,
    shard_index: int,
    teacher_model: str | Path = DEFAULT_TEACHER_MODEL,
    tokenizer_name: str | Path | None = None,
    top_k: int = 32,
    max_model_len: int = 2560,
    teacher_batch_size: int = 1,
    teacher_load_in_4bit: bool = True,
) -> dict[str, Any]:
    suffix = shard_suffix(shard_index, num_shards)
    raw_base = Path(raw_rollout_dir) if raw_rollout_dir is not None else Path(output_dir)
    shard_dir = Path(output_dir) / "shards" / suffix
    raw_path = raw_base / "shards" / suffix / "raw_rollouts.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"缺少可复用的 student rollout：{raw_path}")

    raw_rows = read_jsonl(raw_path)
    topk_rows, teacher_report = _token_logprobs_and_topk(
        raw_rows,
        teacher_model=teacher_model,
        tokenizer_name=tokenizer_name or teacher_model,
        top_k=top_k,
        max_model_len=max_model_len,
        batch_size=teacher_batch_size,
        load_in_4bit=teacher_load_in_4bit,
    )
    topk_path = shard_dir / "teacher_topk.jsonl"
    write_jsonl(topk_path, topk_rows)
    return {
        "shard": suffix,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "raw_rollout_rows": len(raw_rows),
        "teacher_topk_rows": len(topk_rows),
        "outputs": {"raw_rollouts": str(raw_path), "teacher_topk": str(topk_path)},
        "teacher": teacher_report,
    }


def merge_lightning_opd_shards(
    *,
    output_dir: str | Path,
    num_shards: int,
    raw_rollout_dir: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(output_dir)
    raw_base = Path(raw_rollout_dir) if raw_rollout_dir is not None else base
    merged_rows: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        suffix = shard_suffix(shard_index, num_shards)
        topk_path = base / "shards" / suffix / "teacher_topk.jsonl"
        raw_path = raw_base / "shards" / suffix / "raw_rollouts.jsonl"
        rows = read_jsonl(topk_path)
        raw_rows = read_jsonl(raw_path) if raw_path.exists() else []
        merged_rows.extend(rows)
        shard_reports.append(
            {
                "shard": suffix,
                "raw_rollout_rows": len(raw_rows),
                "teacher_topk_rows": len(rows),
                "raw_rollouts": str(raw_path),
                "teacher_topk": str(topk_path),
            }
        )
    merged_rows.sort(key=lambda row: int(row.get("meta", {}).get("global_query_index", 0)))
    train_path = base / "train.jsonl"
    write_jsonl(train_path, merged_rows)

    response_lengths = [int(row.get("meta", {}).get("response_tokens", 0)) for row in merged_rows]
    shape_errors = 0
    top_k_values: set[int] = set()
    for row in merged_rows:
        response_count = int(sum(int(value) for value in row.get("response_mask", [])))
        if response_count != len(row.get("teacher_token_logprobs", [])):
            shape_errors += 1
        topk_ids = row.get("teacher_topk_token_ids", [])
        topk_logprobs = row.get("teacher_topk_logprobs", [])
        if len(topk_ids) != len(topk_logprobs) or len(topk_ids) != response_count:
            shape_errors += 1
        for token_topk in topk_ids:
            top_k_values.add(len(token_topk))

    report_path = base / "report.json"
    existing = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report = {
        **existing,
        "num_shards": num_shards,
        "shards": shard_reports,
        "train_rows": len(merged_rows),
        "shape_errors": shape_errors,
        "top_k_values": sorted(top_k_values),
        "response_tokens": _length_stats(response_lengths),
        "outputs": {**dict(existing.get("outputs", {})), "train": str(train_path), "report": str(report_path)},
    }
    _write_json(report_path, report)
    return report
