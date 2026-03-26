from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from post_train.answers import evaluate_prediction
from post_train.data import load_dataset_rows, make_eval_record, sample_rows
from post_train.eval import resolve_model_args
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.prompts import build_eval_prompt
from post_train.schemas import PreferenceRecord


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def _same_source(row: dict[str, Any], source_name: str) -> bool:
    meta = row.get("meta", {})
    source = str(meta.get("source", "") or "")
    source_dataset = str(meta.get("source_dataset", "") or "")
    return source == source_name or source_dataset == source_name


def load_excluded_ids(paths: list[str | Path], *, source_name: str) -> set[str]:
    excluded_ids: set[str] = set()
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            continue
        for row in read_jsonl(candidate):
            if _same_source(row, source_name):
                excluded_ids.add(str(row.get("id", "")))
    return excluded_ids


def build_query_pool(cfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = load_dataset_rows(
        cfg.query_dataset,
        split=cfg.query_split,
        config_name=cfg.query_config_name,
        cache_dir=cfg.cache_dir,
    )
    eval_rows = [
        make_eval_record(row, index, source=cfg.query_source)
        for index, row in enumerate(raw_rows)
    ]
    excluded_ids = load_excluded_ids(cfg.exclude_paths, source_name=cfg.query_source)
    available_rows = [
        row for row in eval_rows if str(row.get("id", "")) not in excluded_ids
    ]
    query_rows = sample_rows(available_rows, sample_size=cfg.query_count, seed=cfg.seed)
    return query_rows, {
        "raw_rows": len(raw_rows),
        "excluded_ids": len(excluded_ids),
        "available_rows": len(available_rows),
        "sampled_queries": len(query_rows),
    }


def _output_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def sample_candidate_responses(
    query_rows: list[dict[str, Any]],
    cfg,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_args = resolve_model_args(
        cfg.generation_model,
        cfg.base_model_name,
        backend="vllm",
        max_length=cfg.max_model_length,
        device="cuda",
        attn_implementation="sdpa",
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_lora_rank=cfg.max_lora_rank,
    )
    prompts = [build_eval_prompt(str(row["question"])) for row in query_rows]

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
    sampling_params = SamplingParams(
        n=cfg.responses_per_query,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        stop=["</s>", "<|im_end|>"],
    )
    lora_request = None
    if lora_path is not None:
        lora_request = LoRARequest("simpo_adapter", 1, lora_path)

    started_at = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )
    total_seconds = time.perf_counter() - started_at

    sample_rows_out: list[dict[str, Any]] = []
    for row, prompt, output in zip(query_rows, prompts, outputs, strict=True):
        responses: list[dict[str, Any]] = []
        for candidate in output.outputs:
            text = candidate.text
            verdict = evaluate_prediction(
                text, str(row.get("final_answer", "")), require_boxed=True
            )
            responses.append(
                {
                    "text": text,
                    "predicted_answer": str(verdict.get("parsed_answer", "")),
                    "boxed": bool(verdict["boxed"]),
                    "parse_ok": bool(verdict["parse_ok"]),
                    "correct": bool(verdict["correct"]),
                    "output_tokens": _output_tokens(text),
                }
            )
        sample_rows_out.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "final_answer": str(row.get("final_answer", "")),
                "prompt": prompt,
                "responses": responses,
                "meta": row.get("meta", {}),
            }
        )

    return sample_rows_out, {
        "model": cfg.generation_model,
        "responses_per_query": cfg.responses_per_query,
        "sample_count": len(sample_rows_out),
        "total_seconds": total_seconds,
    }


def _best_correct(responses: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        responses,
        key=lambda row: (
            not bool(row["boxed"]),
            not bool(row["parse_ok"]),
            int(row["output_tokens"]),
        ),
    )


def _worst_incorrect(responses: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        responses,
        key=lambda row: (
            bool(row["boxed"]),
            bool(row["parse_ok"]),
            -int(row["output_tokens"]),
        ),
    )


def _has_length_gap(chosen: dict[str, Any], rejected: dict[str, Any], cfg) -> bool:
    chosen_tokens = int(chosen["output_tokens"])
    rejected_tokens = int(rejected["output_tokens"])
    absolute_gap = rejected_tokens - chosen_tokens
    if absolute_gap >= cfg.min_length_gap_tokens:
        return True
    if rejected_tokens <= 0:
        return False
    return (absolute_gap / rejected_tokens) >= cfg.min_length_gap_ratio


def build_preference_pairs(
    raw_samples: list[dict[str, Any]],
    cfg,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    report = {
        "correct_vs_incorrect_pairs": 0,
        "correct_vs_correct_pairs": 0,
        "skipped_all_correct_but_small_gap": 0,
        "skipped_all_correct_without_pair": 0,
        "skipped_without_correct": 0,
        "skipped_empty_or_duplicate_pair": 0,
        "final_pair_count": 0,
        "target_pair_count": cfg.target_pair_count,
    }

    for sample in raw_samples:
        if len(pairs) >= cfg.target_pair_count:
            break

        responses = list(sample.get("responses", []))
        correct_responses = [row for row in responses if bool(row.get("correct"))]
        incorrect_responses = [row for row in responses if not bool(row.get("correct"))]

        pair_type = ""
        chosen: dict[str, Any] | None = None
        rejected: dict[str, Any] | None = None

        if correct_responses and incorrect_responses:
            chosen = _best_correct(correct_responses)
            rejected = _worst_incorrect(incorrect_responses)
            pair_type = "correct_vs_incorrect"
            report["correct_vs_incorrect_pairs"] += 1
        elif len(correct_responses) >= 2:
            ordered = sorted(
                correct_responses,
                key=lambda row: (
                    not bool(row["boxed"]),
                    not bool(row["parse_ok"]),
                    int(row["output_tokens"]),
                ),
            )
            chosen = ordered[0]
            rejected = ordered[-1]
            if chosen is rejected or not _has_length_gap(chosen, rejected, cfg):
                report["skipped_all_correct_but_small_gap"] += 1
                continue
            pair_type = "correct_vs_correct"
            report["correct_vs_correct_pairs"] += 1
        elif correct_responses:
            report["skipped_all_correct_without_pair"] += 1
            continue
        else:
            report["skipped_without_correct"] += 1
            continue

        if chosen is None or rejected is None or chosen["text"] == rejected["text"]:
            report["skipped_empty_or_duplicate_pair"] += 1
            if pair_type == "correct_vs_incorrect":
                report["correct_vs_incorrect_pairs"] -= 1
            elif pair_type == "correct_vs_correct":
                report["correct_vs_correct_pairs"] -= 1
            continue

        record = PreferenceRecord(
            id=str(sample.get("id", "")),
            prompt=str(sample.get("prompt", "")),
            chosen=str(chosen["text"]),
            rejected=str(rejected["text"]),
            meta={
                "question_id": str(sample.get("id", "")),
                "source_dataset": cfg.query_dataset,
                "pair_type": pair_type,
                "chosen_correct": bool(chosen["correct"]),
                "rejected_correct": bool(rejected["correct"]),
                "chosen_output_tokens": int(chosen["output_tokens"]),
                "rejected_output_tokens": int(rejected["output_tokens"]),
            },
        )
        pairs.append(record.model_dump())

    report["final_pair_count"] = len(pairs)
    return pairs, report


def build_pilot_pairs(
    pair_rows: list[dict[str, Any]],
    cfg,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_count = min(cfg.pilot_pair_count, len(pair_rows))
    correct_vs_incorrect = [
        row for row in pair_rows if str(row.get("meta", {}).get("pair_type", "")) == "correct_vs_incorrect"
    ]
    correct_vs_correct = [
        row for row in pair_rows if str(row.get("meta", {}).get("pair_type", "")) == "correct_vs_correct"
    ]

    pilot_rows: list[dict[str, Any]] = []
    if target_count > 0 and correct_vs_incorrect:
        pilot_rows.extend(
            sample_rows(
                correct_vs_incorrect,
                sample_size=min(target_count, len(correct_vs_incorrect)),
                seed=cfg.seed,
            )
        )

    remaining = target_count - len(pilot_rows)
    if remaining > 0 and correct_vs_correct:
        pilot_rows.extend(
            sample_rows(
                correct_vs_correct,
                sample_size=min(remaining, len(correct_vs_correct)),
                seed=cfg.seed + 1,
            )
        )

    report = {
        "target_pilot_pair_count": cfg.pilot_pair_count,
        "final_pilot_pair_count": len(pilot_rows),
        "correct_vs_incorrect_pairs": sum(
            1 for row in pilot_rows if str(row.get("meta", {}).get("pair_type", "")) == "correct_vs_incorrect"
        ),
        "correct_vs_correct_pairs": sum(
            1 for row in pilot_rows if str(row.get("meta", {}).get("pair_type", "")) == "correct_vs_correct"
        ),
    }
    return pilot_rows, report


def prepare_simpo_dataset(cfg) -> dict[str, Any]:
    query_rows, query_report = build_query_pool(cfg)
    write_jsonl(cfg.query_output_path, query_rows)

    raw_samples, sampling_report = sample_candidate_responses(query_rows, cfg)
    write_jsonl(cfg.raw_samples_output_path, raw_samples)

    pair_rows, pair_report = build_preference_pairs(raw_samples, cfg)
    write_jsonl(cfg.pair_output_path, pair_rows)
    pilot_rows, pilot_report = build_pilot_pairs(pair_rows, cfg)
    write_jsonl(cfg.pilot_output_path, pilot_rows)

    report = {
        "query_pool": query_report,
        "sampling": sampling_report,
        "pairing": pair_report,
        "pilot": pilot_report,
    }
    _write_json(cfg.report_path, report)
    return {
        "query_output_path": str(cfg.query_output_path),
        "raw_samples_output_path": str(cfg.raw_samples_output_path),
        "pair_output_path": str(cfg.pair_output_path),
        "pilot_output_path": str(cfg.pilot_output_path),
        "report_path": str(cfg.report_path),
        "report": report,
    }
