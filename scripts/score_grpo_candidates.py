#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.answers import evaluate_prediction
from rl.config import load_grpo_config
from rl.data import format_protocol_prompt
from rl.harness_tasks import resolve_model_args
from rl.io import ensure_parent, read_jsonl, write_jsonl


PROMPT_VERSION = "protocol_prompt_v1"
PROTOCOL_VERSION = "strict_boxed_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 vLLM 给 GRPO 候选题离线打分。")
    parser.add_argument("--config", default="configs/grpo.yaml", help="GRPO 配置文件")
    parser.add_argument(
        "--candidate-dataset",
        default="data/grpo/train_grpo_gsm8k_3857_short.jsonl",
        help="候选数据集 JSONL",
    )
    parser.add_argument(
        "--output-scored",
        default="data/grpo/train_grpo_gsm8k_3857_short_scored.jsonl",
        help="全量打分结果输出路径",
    )
    parser.add_argument("--num-samples-per-problem", type=int, default=4, help="每题采样次数")
    parser.add_argument("--batch-size", type=int, default=32, help="vLLM 推理批大小")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="最大生成长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否复用已有 scored 文件中的命中结果",
    )
    return parser.parse_args()


def resolve_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).strip()
    if prompt:
        return prompt
    return format_protocol_prompt(str(row["question"]))


def build_sampling_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_samples_per_problem": args.num_samples_per_problem,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }


def build_score_model_id(cfg) -> str:
    return f"base={cfg.base_model_name}|adapter={cfg.cold_start_model}"


def build_cache_key(
    row: dict[str, Any],
    *,
    prompt: str,
    model_id: str,
    sampling: dict[str, Any],
    prompt_version: str = PROMPT_VERSION,
    protocol_version: str = PROTOCOL_VERSION,
) -> str:
    payload = {
        "id": str(row["id"]),
        "question": str(row["question"]),
        "final_answer": str(row["final_answer"]),
        "prompt": prompt,
        "score_model_id": model_id,
        "score_prompt_version": prompt_version,
        "score_protocol_version": protocol_version,
        "score_sampling": sampling,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def summarize_trial_results(
    completions: list[str],
    token_counts: list[int],
    final_answer: str,
) -> dict[str, Any]:
    if len(completions) != len(token_counts):
        raise ValueError("completions 和 token_counts 长度不一致")
    if not completions:
        raise ValueError("至少需要一条 completion")

    correct_count = 0
    parse_count = 0
    format_count = 0
    preview_samples: list[dict[str, Any]] = []
    for index, (completion, token_count) in enumerate(zip(completions, token_counts, strict=True)):
        result = evaluate_prediction(completion, final_answer, require_boxed=True)
        correct_count += int(bool(result["correct"]))
        parse_count += int(bool(result["extract_ok"]))
        format_count += int(bool(result["format_ok"]))
        if index < 2:
            preview_samples.append(
                {
                    "text": completion,
                    "parsed_answer": str(result["parsed_answer"]),
                    "correct": bool(result["correct"]),
                    "parse_ok": bool(result["extract_ok"]),
                    "format_ok": bool(result["format_ok"]),
                    "generated_tokens": token_count,
                }
            )

    total = len(completions)
    return {
        "sft_num_trials": total,
        "sft_correct_count": correct_count,
        "sft_parse_count": parse_count,
        "sft_format_count": format_count,
        "sft_correct_rate": correct_count / total,
        "sft_parse_rate": parse_count / total,
        "sft_format_rate": format_count / total,
        "sft_mean_generated_tokens": sum(token_counts) / total,
        "preview_samples": preview_samples,
    }


def build_scored_record(
    row: dict[str, Any],
    *,
    prompt: str,
    model_id: str,
    sampling: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    scored = dict(row)
    scored["prompt"] = prompt
    scored["score_model_id"] = model_id
    scored["score_prompt_version"] = PROMPT_VERSION
    scored["score_protocol_version"] = PROTOCOL_VERSION
    scored["score_sampling"] = sampling
    scored["score_cache_key"] = build_cache_key(
        row,
        prompt=prompt,
        model_id=model_id,
        sampling=sampling,
    )
    scored.update(metrics)
    return scored


def build_resume_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("score_cache_key")): row
        for row in rows
        if row.get("score_cache_key")
    }


def load_existing_index(path: Path, *, enabled: bool) -> dict[str, dict[str, Any]]:
    if not enabled or not path.exists():
        return {}
    return build_resume_index(read_jsonl(path))


def init_vllm(cfg):
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    model_args = resolve_model_args(
        str(cfg.cold_start_model),
        cfg.base_model_name,
        backend="vllm",
        max_length=cfg.max_seq_length,
        device="cuda",
        attn_implementation="sdpa",
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    lora_path = model_args.get("lora_local_path")
    llm = LLM(
        model=model_args["pretrained"],
        trust_remote_code=bool(model_args.get("trust_remote_code", True)),
        dtype=str(model_args.get("dtype", "auto")),
        gpu_memory_utilization=float(model_args.get("gpu_memory_utilization", cfg.gpu_memory_utilization)),
        max_model_len=cfg.max_seq_length,
        seed=cfg.seed,
        enable_lora=bool(lora_path),
    )
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("cold_start", 1, lora_path, base_model_name=cfg.base_model_name)
    return llm, lora_request


def score_pending_rows(
    pending_rows: list[dict[str, Any]],
    *,
    cfg,
    batch_size: int,
    sampling: dict[str, Any],
) -> list[dict[str, Any]]:
    if not pending_rows:
        return []

    from vllm import SamplingParams

    llm, lora_request = init_vllm(cfg)
    sampling_params = SamplingParams(
        n=sampling["num_samples_per_problem"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_tokens=sampling["max_new_tokens"],
        repetition_penalty=1.0,
    )
    model_id = build_score_model_id(cfg)
    scored_rows: list[dict[str, Any]] = []

    for start in range(0, len(pending_rows), batch_size):
        batch = pending_rows[start : start + batch_size]
        prompts = [resolve_prompt(row) for row in batch]
        outputs = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=True,
            lora_request=lora_request,
        )
        for row, prompt, output in zip(batch, prompts, outputs, strict=True):
            completions: list[str] = []
            token_counts: list[int] = []
            for candidate in output.outputs:
                completions.append(candidate.text)
                token_counts.append(len(candidate.token_ids))
            metrics = summarize_trial_results(completions, token_counts, str(row["final_answer"]))
            scored_rows.append(
                build_scored_record(
                    row,
                    prompt=prompt,
                    model_id=model_id,
                    sampling=sampling,
                    metrics=metrics,
                )
            )
    return scored_rows


def main() -> None:
    args = parse_args()
    cfg = load_grpo_config(args.config)
    sampling = build_sampling_spec(args)
    candidates = read_jsonl(args.candidate_dataset)
    output_path = ensure_parent(args.output_scored)
    existing_index = load_existing_index(output_path, enabled=args.resume)
    model_id = build_score_model_id(cfg)

    final_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    for row in candidates:
        prompt = resolve_prompt(row)
        cache_key = build_cache_key(
            row,
            prompt=prompt,
            model_id=model_id,
            sampling=sampling,
        )
        cached = existing_index.get(cache_key)
        if cached is not None:
            final_rows.append(cached)
        else:
            pending_rows.append(row)

    if pending_rows:
        final_rows.extend(
            score_pending_rows(
                pending_rows,
                cfg=cfg,
                batch_size=args.batch_size,
                sampling=sampling,
            )
        )

    row_by_key = {str(row["score_cache_key"]): row for row in final_rows}
    ordered_rows: list[dict[str, Any]] = []
    for row in candidates:
        prompt = resolve_prompt(row)
        key = build_cache_key(
            row,
            prompt=prompt,
            model_id=model_id,
            sampling=sampling,
        )
        ordered_rows.append(row_by_key[key])

    write_jsonl(output_path, ordered_rows)
    reused = len(candidates) - len(pending_rows)
    print(f"候选题总数: {len(candidates)}")
    print(f"缓存命中: {reused}")
    print(f"新打分题数: {len(pending_rows)}")
    print(f"输出 scored 工件: {output_path}")


if __name__ == "__main__":
    main()
