from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from post_train.data import read_json_rows
from post_train.io import ensure_parent, read_jsonl

STRATEGY_UNIFORM_EPOCH = "uniform_epoch"
STRATEGY_MIXED_BOOTSTRAP_CANDIDATE = "mixed_bootstrap_candidate"
STRATEGY_CANDIDATE_RANDOM_MIX = "candidate_random_mix"
MIXED_SOURCE_BOOTSTRAP = "bootstrap_all_correct"
MIXED_SOURCE_CANDIDATE = "candidate_pool"
MIXED_SOURCE_RANDOM = "random_pool"
WHITESPACE_RE = re.compile(r"\s+")


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def normalize_question(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def build_query_epoch_rows(
    query_rows: list[dict[str, Any]],
    *,
    seed: int,
    epoch_index: int,
) -> list[dict[str, Any]]:
    shuffled = list(query_rows)
    random.Random(seed + epoch_index).shuffle(shuffled)
    return shuffled


def build_query_sampler_state(
    query_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if not query_rows:
        raise ValueError("query 池为空，无法启动 on-policy loop。")
    return {
        "seed": seed,
        "pool_size": len(query_rows),
        "epoch_index": 0,
        "epoch_offset": 0,
        "epoch_rows": build_query_epoch_rows(query_rows, seed=seed, epoch_index=0),
    }


def dump_query_sampler_state(sampler_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if sampler_state is None:
        return None
    return {
        "seed": int(sampler_state["seed"]),
        "pool_size": int(sampler_state["pool_size"]),
        "epoch_index": int(sampler_state["epoch_index"]),
        "epoch_offset": int(sampler_state["epoch_offset"]),
    }


def load_query_sampler_state(
    query_rows: list[dict[str, Any]],
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    seed = int(payload["seed"])
    epoch_index = int(payload.get("epoch_index", 0))
    epoch_offset = int(payload.get("epoch_offset", 0))
    return {
        "seed": seed,
        "pool_size": len(query_rows),
        "epoch_index": epoch_index,
        "epoch_offset": epoch_offset,
        "epoch_rows": build_query_epoch_rows(query_rows, seed=seed, epoch_index=epoch_index),
    }


def sample_round_queries(
    query_rows: list[dict[str, Any]],
    sampler_state: dict[str, Any],
    *,
    requested_count: int,
    skip_ids: set[str] | None = None,
    skip_questions: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not query_rows:
        raise ValueError("query 池为空，无法为当前轮采样。")
    skip_ids = skip_ids or set()
    skip_questions = skip_questions or set()
    eligible_rows = [
        row
        for row in query_rows
        if str(row.get("id", "")) not in skip_ids
        and normalize_question(str(row.get("question", ""))) not in skip_questions
    ]
    effective_query_count = min(requested_count, len(eligible_rows))
    if effective_query_count <= 0:
        raise ValueError("requested_count 必须大于 0。")

    start_epoch_index = int(sampler_state["epoch_index"])
    start_epoch_offset = int(sampler_state["epoch_offset"])
    sampled_rows: list[dict[str, Any]] = []
    crossed_epoch = False
    seen_keys: set[tuple[str, str]] = set()

    while len(sampled_rows) < effective_query_count:
        epoch_rows = sampler_state["epoch_rows"]
        while int(sampler_state["epoch_offset"]) < len(epoch_rows) and len(sampled_rows) < effective_query_count:
            epoch_offset = int(sampler_state["epoch_offset"])
            row = epoch_rows[epoch_offset]
            sampler_state["epoch_offset"] = epoch_offset + 1
            row_id = str(row.get("id", ""))
            row_question = normalize_question(str(row.get("question", "")))
            key = (row_id, row_question)
            if row_id in skip_ids or row_question in skip_questions or key in seen_keys:
                continue
            sampled_rows.append(row)
            seen_keys.add(key)
        if len(sampled_rows) >= effective_query_count:
            break
        crossed_epoch = True
        sampler_state["epoch_index"] = int(sampler_state["epoch_index"]) + 1
        sampler_state["epoch_rows"] = build_query_epoch_rows(
            query_rows,
            seed=int(sampler_state["seed"]),
            epoch_index=int(sampler_state["epoch_index"]),
        )
        sampler_state["epoch_offset"] = 0

    return sampled_rows, {
        "query_pool_size": len(query_rows),
        "eligible_query_count": len(eligible_rows),
        "requested_query_count": requested_count,
        "effective_query_count": effective_query_count,
        "crossed_epoch": crossed_epoch,
        "epoch_index_start": start_epoch_index,
        "epoch_offset_start": start_epoch_offset,
        "epoch_index_end": int(sampler_state["epoch_index"]),
        "epoch_offset_end": int(sampler_state["epoch_offset"]),
    }


def _copy_query_row(row: dict[str, Any], *, source_name: str) -> dict[str, Any]:
    meta = dict(row.get("meta", {}))
    meta["on_policy_query_strategy_source"] = source_name
    return {
        "id": str(row.get("id", "")),
        "question": str(row.get("question", "")),
        "final_answer": str(row.get("final_answer", "")),
        "meta": meta,
    }


def load_bootstrap_all_correct_pool(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        response_summary = dict(row.get("response_summary", {}))
        response_count = int(response_summary.get("response_count", 0))
        correct_count = int(response_summary.get("correct_count", 0))
        if response_count > 0 and correct_count == response_count:
            selected_rows.append(_copy_query_row(row, source_name=MIXED_SOURCE_BOOTSTRAP))
    return selected_rows


def load_candidate_query_pool(path: str | Path) -> list[dict[str, Any]]:
    return [
        _copy_query_row(row, source_name=MIXED_SOURCE_CANDIDATE)
        for row in read_json_rows(path)
    ]


def _strategy_artifact_dir(data_base_dir: str | Path) -> Path:
    return Path(data_base_dir) / "query_strategy"


def _sample_or_empty(
    query_rows: list[dict[str, Any]],
    sampler_state: dict[str, Any] | None,
    *,
    requested_count: int,
    skip_ids: set[str] | None = None,
    skip_questions: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sampler_state is None or requested_count <= 0 or not query_rows:
        return [], {
            "query_pool_size": len(query_rows),
            "eligible_query_count": 0,
            "requested_query_count": requested_count,
            "effective_query_count": 0,
            "crossed_epoch": False,
            "epoch_index_start": 0,
            "epoch_offset_start": 0,
            "epoch_index_end": 0,
            "epoch_offset_end": 0,
        }
    skip_ids = skip_ids or set()
    skip_questions = skip_questions or set()
    eligible_rows = [
        row
        for row in query_rows
        if str(row.get("id", "")) not in skip_ids
        and normalize_question(str(row.get("question", ""))) not in skip_questions
    ]
    if not eligible_rows:
        return [], {
            "query_pool_size": len(query_rows),
            "eligible_query_count": 0,
            "requested_query_count": requested_count,
            "effective_query_count": 0,
            "crossed_epoch": False,
            "epoch_index_start": int(sampler_state["epoch_index"]),
            "epoch_offset_start": int(sampler_state["epoch_offset"]),
            "epoch_index_end": int(sampler_state["epoch_index"]),
            "epoch_offset_end": int(sampler_state["epoch_offset"]),
        }
    return sample_round_queries(
        query_rows,
        sampler_state,
        requested_count=requested_count,
        skip_ids=skip_ids,
        skip_questions=skip_questions,
    )
