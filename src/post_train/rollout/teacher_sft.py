from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from post_train.answers import evaluate_prediction
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.prompts import build_eval_prompt
from post_train.rollout.common import length_stats, render_chat_prompt, select_shard_rows, shard_suffix


DEFAULT_PROMPT_SOURCE = Path("data/on_policy_loop/query_strategy/candidate_pool.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/rollout/teacher_sft/qwen3_4b_non_thinking_step500")


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _output_words(text: str) -> int:
    stripped = text.strip()
    return len(stripped.split()) if stripped else 0


def build_teacher_sft_prompts(
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
        question = str(row.get("question", ""))
        prompt = str(row.get("prompt") or build_eval_prompt(question))
        meta = dict(row.get("meta", {}))
        meta.update({"prompt_source_path": str(prompt_source), "global_query_index": index})
        prompt_rows.append(
            {
                "id": str(row.get("id", "")),
                "question": question,
                "final_answer": str(row.get("final_answer", "")),
                "prompt": prompt,
                "meta": meta,
            }
        )

    base = Path(output_dir)
    prompts_path = base / "prompts.jsonl"
    write_jsonl(prompts_path, prompt_rows)
    report = {
        "prompt_source": str(prompt_source),
        "raw_rows": len(rows),
        "sample_size": sample_size,
        "seed": seed,
        "outputs": {"prompts": str(prompts_path)},
    }
    _write_json(base / "report.prompts.json", report)
    return report


def _render_prompts(
    prompt_rows: list[dict[str, Any]],
    *,
    model: str | Path,
    tokenizer_name: str | Path | None,
    use_chat_template: bool,
    chat_template_enable_thinking: bool | None,
    system_prompt: str | None,
    assistant_prefill: str | None,
    trust_remote_code: bool,
) -> tuple[list[str], object | None, dict[str, Any]]:
    template_tokenizer_name = str(tokenizer_name or model)
    tokenizer = None
    if use_chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(template_tokenizer_name, trust_remote_code=trust_remote_code)
        rendered = [
            render_chat_prompt(
                tokenizer,
                str(row["prompt"]),
                system_prompt=system_prompt,
                assistant_prefill=assistant_prefill,
                enable_thinking=chat_template_enable_thinking,
            )
            for row in prompt_rows
        ]
    else:
        rendered = [str(row["prompt"]) + (assistant_prefill or "") for row in prompt_rows]

    return rendered, tokenizer, {
        "tokenizer_name": template_tokenizer_name,
        "use_chat_template": use_chat_template,
        "chat_template_enable_thinking": chat_template_enable_thinking,
        "system_prompt": system_prompt,
        "assistant_prefill": assistant_prefill,
    }


def _token_count(text: str, tokenizer: object | None) -> int:
    if tokenizer is None:
        return _output_words(text)
    return len(tokenizer(text, add_special_tokens=False).input_ids)  # type: ignore[operator]


def run_teacher_sft_shard(
    *,
    output_dir: str | Path,
    num_shards: int,
    shard_index: int,
    model: str | Path,
    tokenizer_name: str | Path | None = None,
    use_chat_template: bool = True,
    chat_template_enable_thinking: bool | None = False,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
    responses_per_prompt: int = 1,
    max_new_tokens: int = 3072,
    max_model_len: int = 4096,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 32,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = False,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    base = Path(output_dir)
    prompt_rows = read_jsonl(base / "prompts.jsonl")
    shard_rows = select_shard_rows(prompt_rows, num_shards=num_shards, shard_index=shard_index)
    for local_index, row in enumerate(shard_rows):
        meta = dict(row.get("meta", {}))
        meta.update({"num_shards": num_shards, "shard_index": shard_index, "shard_local_index": local_index})
        row["meta"] = meta

    rendered_prompts, tokenizer, render_report = _render_prompts(
        shard_rows,
        model=model,
        tokenizer_name=tokenizer_name,
        use_chat_template=use_chat_template,
        chat_template_enable_thinking=chat_template_enable_thinking,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
        trust_remote_code=trust_remote_code,
    )

    llm = LLM(
        model=str(model),
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    sampling_params = SamplingParams(
        n=responses_per_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        stop=["</s>", "<|im_end|>"],
    )

    started_at = time.perf_counter()
    outputs = llm.generate(rendered_prompts, sampling_params=sampling_params, use_tqdm=True)
    total_seconds = time.perf_counter() - started_at

    response_count = 0
    boxed_count = 0
    parse_ok_count = 0
    correct_count = 0
    truncated_count = 0
    token_lengths: list[int] = []
    word_lengths: list[int] = []
    finish_reasons: dict[str, int] = {}
    think_count = 0
    think_end_count = 0
    raw_rows: list[dict[str, Any]] = []

    for row, rendered_prompt, output in zip(shard_rows, rendered_prompts, outputs, strict=True):
        responses: list[dict[str, Any]] = []
        for rollout_index, candidate in enumerate(output.outputs):
            text = str(candidate.text)
            finish_reason = str(getattr(candidate, "finish_reason", ""))
            finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
            output_tokens = _token_count(text, tokenizer)
            output_words = _output_words(text)
            truncated = finish_reason == "length" or output_tokens >= max_new_tokens
            verdict = evaluate_prediction(text, str(row.get("final_answer", "")), require_boxed=True)

            response_count += 1
            boxed_count += int(bool(verdict["boxed"]))
            parse_ok_count += int(bool(verdict["parse_ok"]))
            correct_count += int(bool(verdict["correct"]))
            truncated_count += int(truncated)
            think_count += int("<think>" in text)
            think_end_count += int("</think>" in text)
            token_lengths.append(output_tokens)
            word_lengths.append(output_words)
            responses.append(
                {
                    "rollout_index": rollout_index,
                    "text": text,
                    "predicted_answer": str(verdict.get("parsed_answer", "")),
                    "boxed": bool(verdict["boxed"]),
                    "parse_ok": bool(verdict["parse_ok"]),
                    "correct": bool(verdict["correct"]),
                    "output_tokens": output_tokens,
                    "output_tokens_whitespace": output_words,
                    "finish_reason": finish_reason,
                    "truncated": truncated,
                }
            )
        raw_rows.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "final_answer": str(row.get("final_answer", "")),
                "prompt": str(row.get("prompt", "")),
                "rendered_prompt": rendered_prompt,
                "responses": responses,
                "meta": dict(row.get("meta", {})),
            }
        )

    suffix = shard_suffix(shard_index, num_shards)
    shard_dir = base / "shards" / suffix
    raw_path = shard_dir / "raw_rollouts.jsonl"
    write_jsonl(raw_path, raw_rows)

    report = {
        "shard": suffix,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "model": str(model),
        **render_report,
        "prompt_count": len(shard_rows),
        "responses_per_prompt": responses_per_prompt,
        "response_count": response_count,
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "total_seconds": total_seconds,
        "responses_per_second": response_count / total_seconds if total_seconds > 0 else 0.0,
        "finish_reasons": finish_reasons,
        "token_len": length_stats(token_lengths),
        "word_len": length_stats(word_lengths),
        "boxed_count": boxed_count,
        "parse_ok_count": parse_ok_count,
        "correct_count": correct_count,
        "truncated_count": truncated_count,
        "boxed_rate": boxed_count / response_count if response_count else 0.0,
        "parse_ok_rate": parse_ok_count / response_count if response_count else 0.0,
        "correct_rate": correct_count / response_count if response_count else 0.0,
        "trunc_rate": truncated_count / response_count if response_count else 0.0,
        "response_think_count": think_count,
        "response_think_end_count": think_end_count,
        "rendered_prompt_empty_think_prefix_count": sum("<think>\n\n</think>\n\n" in row["rendered_prompt"] for row in raw_rows),
        "outputs": {"raw_rollouts": str(raw_path)},
    }
    report_path = shard_dir / "report.json"
    _write_json(report_path, report)
    report["outputs"]["report"] = str(report_path)
    return report


def merge_teacher_sft_shards(*, output_dir: str | Path, num_shards: int, model: str | Path) -> dict[str, Any]:
    base = Path(output_dir)
    merged: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    duplicate_count = 0
    for shard_index in range(num_shards):
        suffix = shard_suffix(shard_index, num_shards)
        raw_path = base / "shards" / suffix / "raw_rollouts.jsonl"
        report_path = base / "shards" / suffix / "report.json"
        rows = read_jsonl(raw_path)
        for row in rows:
            key = (str(row.get("id", "")), int(row.get("meta", {}).get("global_query_index", -1)))
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            merged.append(row)
        shard_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        shard_reports.append({"shard": suffix, "raw_rollouts": str(raw_path), "rows": len(rows), "report": shard_report})

    merged.sort(key=lambda row: int(row.get("meta", {}).get("global_query_index", 0)))
    raw_path = base / "raw" / "raw_rollouts.merged.jsonl"
    write_jsonl(raw_path, merged)
    analysis = analyze_teacher_sft_raw(merged)
    report = {
        "model": str(model),
        "num_shards": num_shards,
        "merge": {
            "shards": shard_reports,
            "raw_rows": sum(report["rows"] for report in shard_reports),
            "merged_rows": len(merged),
            "duplicates_skipped": duplicate_count,
        },
        "analysis": analysis,
        "outputs": {"raw_rollouts": str(raw_path)},
    }
    report_path = base / "report.json"
    _write_json(report_path, report)
    report["outputs"]["report"] = str(report_path)
    return report


def analyze_teacher_sft_raw(rows: list[dict[str, Any]]) -> dict[str, Any]:
    response_count = 0
    boxed_count = 0
    parse_ok_count = 0
    correct_count = 0
    truncated_count = 0
    token_lengths: list[int] = []
    word_lengths: list[int] = []
    finish_reasons: dict[str, int] = {}
    think_count = 0
    think_end_count = 0
    for row in rows:
        for response in row.get("responses", []):
            response_count += 1
            boxed_count += int(bool(response.get("boxed")))
            parse_ok_count += int(bool(response.get("parse_ok")))
            correct_count += int(bool(response.get("correct")))
            truncated_count += int(bool(response.get("truncated")))
            token_lengths.append(int(response.get("output_tokens", 0)))
            word_lengths.append(int(response.get("output_tokens_whitespace", 0)))
            finish_reason = str(response.get("finish_reason", ""))
            finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
            text = str(response.get("text", ""))
            think_count += int("<think>" in text)
            think_end_count += int("</think>" in text)

    return {
        "prompt_count": len(rows),
        "response_count": response_count,
        "finish_reasons": finish_reasons,
        "token_len": length_stats(token_lengths),
        "word_len": length_stats(word_lengths),
        "boxed_count": boxed_count,
        "parse_ok_count": parse_ok_count,
        "correct_count": correct_count,
        "truncated_count": truncated_count,
        "boxed_rate": boxed_count / response_count if response_count else 0.0,
        "parse_ok_rate": parse_ok_count / response_count if response_count else 0.0,
        "correct_rate": correct_count / response_count if response_count else 0.0,
        "trunc_rate": truncated_count / response_count if response_count else 0.0,
        "response_think_count": think_count,
        "response_think_end_count": think_end_count,
        "rendered_prompt_empty_think_prefix_count": sum("<think>\n\n</think>\n\n" in str(row.get("rendered_prompt", "")) for row in rows),
    }
