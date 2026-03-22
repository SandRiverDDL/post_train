from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rl.answers import ensure_boxed_final_answer
from rl.data import GSM8K_ANSWER_RE, clean_completion_for_protocol, make_record, sample_numinamath, to_sft_record
from rl.grpo import build_grpo_record, response_token_length


def infer_source(dataset_name: str, explicit_source: str) -> str:
    if explicit_source != "auto":
        return explicit_source

    lowered = dataset_name.lower()
    if "gsm8k" in lowered:
        return "gsm8k"
    return "numinamath"


def load_train_dataset(dataset_name: str, dataset_config: str | None, split: str, cache_dir: str | None):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def build_train_record(row: dict[str, object], index: int, source: str) -> dict[str, object]:
    if source == "numinamath":
        return to_sft_record(row, index)

    if source == "gsm8k":
        record = make_record(row, index, source, include_solution=False)
        raw_answer = str(row.get("answer", "")).strip()
        match = GSM8K_ANSWER_RE.search(raw_answer)
        final_answer = match.group(1).strip() if match else str(record["final_answer"])
        record["final_answer"] = final_answer
        record["solution"] = ensure_boxed_final_answer(raw_answer, final_answer)
        record["problem_type"] = "Arithmetic"
        return record

    return make_record(row, index, source, include_solution=True)


def sample_rows(
    rows: list[dict[str, object]],
    rng: random.Random,
    total_samples: int,
    source: str,
) -> list[dict[str, object]]:
    if total_samples <= 0:
        raise ValueError("num_samples 必须大于 0")

    if source == "numinamath":
        return sample_numinamath(rows, rng, total_samples)

    shuffled = list(rows)
    rng.shuffle(shuffled)
    if len(shuffled) < total_samples:
        raise ValueError(f"{source} 过滤后样本不足：{len(shuffled)} < {total_samples}")
    return shuffled[:total_samples]


def build_grpo_candidates(
    dataset,
    *,
    source: str,
    tokenizer,
    min_response_tokens: int,
    max_response_tokens: int,
    prompt_version: str = "v1",
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for index in range(len(dataset)):
        row = build_train_record(dict(dataset[index]), index, source)
        reference_solution = clean_completion_for_protocol(str(row["solution"]))
        token_length = response_token_length(reference_solution, tokenizer)
        if token_length < min_response_tokens or token_length > max_response_tokens:
            continue
        candidates.append(
            build_grpo_record(
                row,
                prompt_version=prompt_version,
                response_tokens=token_length,
                reference_solution=reference_solution,
            )
        )
    return candidates


def select_rows_by_budget(
    rows: list[dict[str, Any]],
    *,
    target_size: int,
    stratify_by: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    if target_size > len(rows):
        raise ValueError(f"筛后样本不足：{len(rows)} < {target_size}")
    if not stratify_by:
        rng = random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:target_size]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_name = str(row.get(stratify_by, "Other"))
        groups.setdefault(group_name, []).append(row)

    total = len(rows)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for group_name, group_rows in groups.items():
        raw = target_size * len(group_rows) / total
        base = min(len(group_rows), int(raw))
        allocations[group_name] = base
        remainders.append((raw - base, group_name))

    assigned = sum(allocations.values())
    for _, group_name in sorted(remainders, reverse=True):
        if assigned >= target_size:
            break
        if allocations[group_name] >= len(groups[group_name]):
            continue
        allocations[group_name] += 1
        assigned += 1

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group_name in sorted(groups):
        group_rows = list(groups[group_name])
        rng.shuffle(group_rows)
        selected.extend(group_rows[: allocations[group_name]])

    if len(selected) < target_size:
        remaining = [row for row in rows if row not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_size - len(selected)])
    return selected[:target_size]
