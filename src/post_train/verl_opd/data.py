from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from post_train.io import ensure_parent, read_jsonl
from post_train.prompts import build_eval_prompt
from post_train.rollout.math_sft import build_math_query_pool


DEFAULT_CANDIDATE_SOURCE = Path("data/on_policy_loop/query_strategy/candidate_pool.jsonl")
DEFAULT_DAPO_SOURCE = Path(
    "/mnt/dataY/fsw/cache/huggingface/hub/datasets--open-r1--DAPO-Math-17k-Processed/"
    "snapshots/31dd309567e3da778038cc87d868b6097a3ccf68/all/train-00000-of-00001.parquet"
)
DEFAULT_SMOKE_OUTPUT_DIR = Path("../verl/data/opd_dapo17k/smoke128")
DEFAULT_MIX_OUTPUT_DIR = Path("../verl/data/opd_mix/candidate1_dapo4_1k")
DEFAULT_MATH_LEVEL_OUTPUT_DIR = Path("../verl/data/opd_math/hendrycks_l34_prompt256_1k_seed42")
DEFAULT_QWEN3_1P7B_TOKENIZER = Path(
    "/mnt/dataY/fsw/cache/huggingface/hub/models--Qwen--Qwen3-1.7B/"
    "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _chat_prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_eval_prompt(question)}]


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _prompt_token_length(
    tokenizer: Any,
    question: str,
    *,
    enable_thinking: bool,
) -> int:
    messages = _chat_prompt(question)
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return len(token_ids)


def _candidate_row(row: dict[str, Any], *, source_path: str | Path, source_index: int) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError(f"candidate 第 {source_index} 行缺少 question。")
    answer = str(row.get("final_answer", "")).strip()
    if not answer:
        raise ValueError(f"candidate 第 {source_index} 行缺少 final_answer。")
    meta = dict(row.get("meta", {}))
    return {
        "data_source": "math_dapo",
        "prompt": _chat_prompt(question),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": str(row.get("id", source_index)),
            "source": "opd_candidate",
            "source_path": str(source_path),
            "source_index": source_index,
            "question": question,
            "final_answer": answer,
            "meta": meta,
        },
    }


def _load_dapo_rows(path: str | Path) -> list[dict[str, Any]]:
    dataset = Dataset.from_parquet(str(path))
    return [dict(row) for row in dataset]


def _reward_model(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("reward_model")
    if isinstance(value, dict):
        return dict(value)
    return {"style": "rule", "ground_truth": str(row.get("final_answer", ""))}


def _dapo_question(row: dict[str, Any], *, source_index: int) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    source_prompt = row.get("source_prompt")
    if isinstance(source_prompt, list) and source_prompt:
        first = source_prompt[0]
        if isinstance(first, dict) and str(first.get("content", "")).strip():
            return str(first["content"]).strip()
    raise ValueError(f"DAPO 第 {source_index} 行缺少可用题面。")


def _dapo_row(row: dict[str, Any], *, source_path: str | Path, source_index: int) -> dict[str, Any]:
    question = _dapo_question(row, source_index=source_index)
    reward_model = _reward_model(row)
    extra = row.get("extra_info")
    extra_info = dict(extra) if isinstance(extra, dict) else {}
    extra_info.update(
        {
            "source": "opd_dapo",
            "source_path": str(source_path),
            "source_index": source_index,
            "question": question,
        }
    )
    return {
        "data_source": "math_dapo",
        "prompt": _chat_prompt(question),
        "ability": str(row.get("ability") or "math"),
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def _math_row(row: dict[str, Any], *, prompt_token_length: int) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("MATH 样本缺少 question。")
    answer = str(row.get("final_answer", "")).strip()
    if not answer:
        raise ValueError(f"MATH 样本缺少 final_answer：id={row.get('id')}")
    meta = dict(row.get("meta", {}))
    return {
        "data_source": "math_dapo",
        "prompt": _chat_prompt(question),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": str(row.get("id", "")),
            "source": "opd_math_hendrycks",
            "source_dataset": str(meta.get("source_dataset", "")),
            "source_split": str(meta.get("source_split", "")),
            "source_index": int(meta.get("source_index", -1)),
            "source_config": str(meta.get("source_config", "")),
            "level": int(meta.get("level", -1)),
            "type": str(meta.get("type", "")),
            "question": question,
            "final_answer": answer,
            "prompt_token_length": prompt_token_length,
        },
    }


def _write_parquet(rows: list[dict[str, Any]], output_dir: str | Path) -> Path:
    if not rows:
        raise ValueError("不能写入空 OPD 数据集。")
    output_path = ensure_parent(Path(output_dir) / "train.parquet")
    Dataset.from_list(rows).to_parquet(str(output_path))
    return output_path


def build_smoke_dataset(
    *,
    dapo_source: str | Path = DEFAULT_DAPO_SOURCE,
    output_dir: str | Path = DEFAULT_SMOKE_OUTPUT_DIR,
    sample_size: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    dapo_rows = _load_dapo_rows(dapo_source)
    if sample_size > len(dapo_rows):
        raise ValueError(f"DAPO 数量不足：需要 {sample_size}，实际只有 {len(dapo_rows)}")
    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(dapo_rows)), sample_size)
    rows = [
        _dapo_row(dapo_rows[index], source_path=dapo_source, source_index=index)
        for index in sampled_indices
    ]
    rng.shuffle(rows)
    train_path = _write_parquet(rows, output_dir)
    report = {
        "mode": "smoke",
        "dapo_source": str(dapo_source),
        "sample_size": sample_size,
        "seed": seed,
        "counts": {"opd_dapo": len(rows)},
        "outputs": {"train": str(train_path)},
    }
    _write_json(Path(output_dir) / "report.json", report)
    return report


def build_mix_dataset(
    *,
    candidate_source: str | Path = DEFAULT_CANDIDATE_SOURCE,
    dapo_source: str | Path = DEFAULT_DAPO_SOURCE,
    output_dir: str | Path = DEFAULT_MIX_OUTPUT_DIR,
    candidate_size: int = 200,
    dapo_size: int = 800,
    seed: int = 42,
) -> dict[str, Any]:
    candidate_rows = read_jsonl(candidate_source)
    dapo_rows = _load_dapo_rows(dapo_source)
    if candidate_size > len(candidate_rows):
        raise ValueError(f"candidate 数量不足：需要 {candidate_size}，实际只有 {len(candidate_rows)}")
    if dapo_size > len(dapo_rows):
        raise ValueError(f"DAPO 数量不足：需要 {dapo_size}，实际只有 {len(dapo_rows)}")

    rng = random.Random(seed)
    candidate_indices = rng.sample(range(len(candidate_rows)), candidate_size)
    rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for index in candidate_indices:
        row = _candidate_row(candidate_rows[index], source_path=candidate_source, source_index=index)
        question_key = _normalize_question(str(row["extra_info"]["question"]))
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        rows.append(row)

    dapo_indices = list(range(len(dapo_rows)))
    rng.shuffle(dapo_indices)
    dapo_added = 0
    for index in dapo_indices:
        if dapo_added >= dapo_size:
            break
        question = _dapo_question(dapo_rows[index], source_index=index)
        question_key = _normalize_question(question)
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        rows.append(_dapo_row(dapo_rows[index], source_path=dapo_source, source_index=index))
        dapo_added += 1

    if len(rows) != candidate_size + dapo_size:
        raise ValueError(
            "混合数据数量不足："
            f"candidate={len(rows) - dapo_added}/{candidate_size}, dapo={dapo_added}/{dapo_size}"
        )

    rng.shuffle(rows)
    train_path = _write_parquet(rows, output_dir)
    report = {
        "mode": "mix",
        "candidate_source": str(candidate_source),
        "dapo_source": str(dapo_source),
        "candidate_size": candidate_size,
        "dapo_size": dapo_size,
        "total_size": len(rows),
        "seed": seed,
        "counts": {
            "opd_candidate": sum(1 for row in rows if row["extra_info"]["source"] == "opd_candidate"),
            "opd_dapo": sum(1 for row in rows if row["extra_info"]["source"] == "opd_dapo"),
        },
        "outputs": {"train": str(train_path)},
    }
    _write_json(Path(output_dir) / "report.json", report)
    return report


def build_math_level_dataset(
    *,
    output_dir: str | Path = DEFAULT_MATH_LEVEL_OUTPUT_DIR,
    dataset_name: str = "EleutherAI/hendrycks_math",
    split: str = "train",
    levels: list[int] | None = None,
    sample_size: int = 1000,
    seed: int = 42,
    max_prompt_tokens: int = 256,
    tokenizer_name: str | Path = DEFAULT_QWEN3_1P7B_TOKENIZER,
    chat_template_enable_thinking: bool = False,
    cache_dir: str | None = "/mnt/dataY/fsw/cache/huggingface",
) -> dict[str, Any]:
    selected_levels = levels or [3, 4]
    query_rows, pool_report = build_math_query_pool(
        dataset_name=dataset_name,
        split=split,
        sample_size=None,
        seed=seed,
        levels=selected_levels,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name), trust_remote_code=True)

    prompt_lengths: list[int] = []
    filtered_rows: list[tuple[dict[str, Any], int]] = []
    filtered_missing_answer = 0
    filtered_over_prompt = 0
    for row in query_rows:
        if not str(row.get("final_answer", "")).strip():
            filtered_missing_answer += 1
            continue
        prompt_len = _prompt_token_length(
            tokenizer,
            str(row["question"]),
            enable_thinking=chat_template_enable_thinking,
        )
        prompt_lengths.append(prompt_len)
        if prompt_len > max_prompt_tokens:
            filtered_over_prompt += 1
            continue
        filtered_rows.append((row, prompt_len))

    if sample_size > len(filtered_rows):
        raise ValueError(
            "MATH level 数据数量不足："
            f"需要 {sample_size}，level 过滤后 {len(query_rows)}，"
            f"prompt_token<={max_prompt_tokens} 后 {len(filtered_rows)}。"
        )

    rng = random.Random(seed)
    sampled = rng.sample(filtered_rows, sample_size)
    rows = [_math_row(row, prompt_token_length=prompt_len) for row, prompt_len in sampled]
    rng.shuffle(rows)

    train_path = _write_parquet(rows, output_dir)
    sampled_lengths = [int(row["extra_info"]["prompt_token_length"]) for row in rows]
    report = {
        "mode": "math-levels",
        "dataset_name": dataset_name,
        "split": split,
        "levels": selected_levels,
        "sample_size": sample_size,
        "seed": seed,
        "max_prompt_tokens": max_prompt_tokens,
        "tokenizer_name": str(tokenizer_name),
        "chat_template_enable_thinking": chat_template_enable_thinking,
        "pool_report": pool_report,
        "counts": {
            "level_filtered": len(query_rows),
            "filtered_missing_answer": filtered_missing_answer,
            "filtered_over_prompt_tokens": filtered_over_prompt,
            "eligible_after_prompt_filter": len(filtered_rows),
            "written": len(rows),
        },
        "prompt_token_length": {
            "all_level_filtered": {
                "min": min(prompt_lengths) if prompt_lengths else 0,
                "mean": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0,
                "p50": _percentile(prompt_lengths, 0.50),
                "p90": _percentile(prompt_lengths, 0.90),
                "p95": _percentile(prompt_lengths, 0.95),
                "max": max(prompt_lengths) if prompt_lengths else 0,
            },
            "sampled": {
                "min": min(sampled_lengths) if sampled_lengths else 0,
                "mean": sum(sampled_lengths) / len(sampled_lengths) if sampled_lengths else 0.0,
                "p50": _percentile(sampled_lengths, 0.50),
                "p90": _percentile(sampled_lengths, 0.90),
                "p95": _percentile(sampled_lengths, 0.95),
                "max": max(sampled_lengths) if sampled_lengths else 0,
            },
        },
        "outputs": {"train": str(train_path)},
    }
    _write_json(Path(output_dir) / "report.json", report)
    return report
