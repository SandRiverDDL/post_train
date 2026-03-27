from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from post_train.data import read_json_rows
from post_train.io import ensure_parent, read_jsonl, write_jsonl

STRATEGY_UNIFORM_EPOCH = "uniform_epoch"
STRATEGY_MIXED_BOOTSTRAP_CANDIDATE = "mixed_bootstrap_candidate"
MIXED_SOURCE_BOOTSTRAP = "bootstrap_all_correct"
MIXED_SOURCE_CANDIDATE = "candidate_pool"
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


def build_mixed_query_strategy(
    *,
    bootstrap_all_correct_raw_samples: str | Path,
    candidate_query_file: str | Path,
    data_base_dir: str | Path,
    seed: int,
    bootstrap_ratio: float,
    candidate_ratio: float,
    candidate_freeze_all_correct_hits: int,
) -> dict[str, Any]:
    bootstrap_rows = load_bootstrap_all_correct_pool(bootstrap_all_correct_raw_samples)
    candidate_rows = load_candidate_query_pool(candidate_query_file)
    if not bootstrap_rows and not candidate_rows:
        raise ValueError("mixed query strategy 没有可用 query。")
    candidate_stats = {
        str(row.get("id", "")): {
            "sample_hits": 0,
            "all_correct_hits": 0,
            "frozen": False,
            "frozen_after_round": None,
        }
        for row in candidate_rows
    }
    strategy_dir = _strategy_artifact_dir(data_base_dir)
    state = {
        "strategy_name": STRATEGY_MIXED_BOOTSTRAP_CANDIDATE,
        "seed": seed,
        "bootstrap_ratio": bootstrap_ratio,
        "candidate_ratio": candidate_ratio,
        "candidate_freeze_all_correct_hits": candidate_freeze_all_correct_hits,
        "bootstrap_all_correct_raw_samples": str(bootstrap_all_correct_raw_samples),
        "candidate_query_file": str(candidate_query_file),
        "artifact_dir": strategy_dir,
        "bootstrap_pool_path": strategy_dir / "bootstrap_all_correct_pool.jsonl",
        "candidate_pool_path": strategy_dir / "candidate_pool.jsonl",
        "state_path": strategy_dir / "state.json",
        "bootstrap_rows": bootstrap_rows,
        "candidate_rows": candidate_rows,
        "bootstrap_sampler_state": build_query_sampler_state(bootstrap_rows, seed=seed) if bootstrap_rows else None,
        "candidate_sampler_state": build_query_sampler_state(candidate_rows, seed=seed + 1000) if candidate_rows else None,
        "candidate_stats": candidate_stats,
        "candidate_frozen_ids": set(),
        "last_updated_round": 0,
    }
    persist_mixed_query_strategy_state(state)
    return state


def persist_mixed_query_strategy_state(state: dict[str, Any]) -> Path:
    artifact_dir = ensure_parent(Path(state["artifact_dir"]) / ".placeholder").parent
    write_jsonl(state["bootstrap_pool_path"], state["bootstrap_rows"])
    write_jsonl(state["candidate_pool_path"], state["candidate_rows"])
    payload = {
        "strategy_name": state["strategy_name"],
        "seed": state["seed"],
        "bootstrap_ratio": state["bootstrap_ratio"],
        "candidate_ratio": state["candidate_ratio"],
        "candidate_freeze_all_correct_hits": state["candidate_freeze_all_correct_hits"],
        "bootstrap_all_correct_raw_samples": state["bootstrap_all_correct_raw_samples"],
        "candidate_query_file": state["candidate_query_file"],
        "bootstrap_pool_path": str(state["bootstrap_pool_path"]),
        "candidate_pool_path": str(state["candidate_pool_path"]),
        "bootstrap_pool_size": len(state["bootstrap_rows"]),
        "candidate_pool_size": len(state["candidate_rows"]),
        "candidate_frozen_ids": sorted(state["candidate_frozen_ids"]),
        "candidate_frozen_count": len(state["candidate_frozen_ids"]),
        "candidate_stats": state["candidate_stats"],
        "bootstrap_sampler_state": dump_query_sampler_state(state["bootstrap_sampler_state"]),
        "candidate_sampler_state": dump_query_sampler_state(state["candidate_sampler_state"]),
        "last_updated_round": int(state.get("last_updated_round", 0)),
    }
    return _write_json(state["state_path"], payload)


def load_mixed_query_strategy_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    bootstrap_pool_path = Path(payload["bootstrap_pool_path"])
    candidate_pool_path = Path(payload["candidate_pool_path"])
    bootstrap_rows = read_jsonl(bootstrap_pool_path)
    candidate_rows = read_jsonl(candidate_pool_path)
    return {
        "strategy_name": payload["strategy_name"],
        "seed": int(payload["seed"]),
        "bootstrap_ratio": float(payload["bootstrap_ratio"]),
        "candidate_ratio": float(payload["candidate_ratio"]),
        "candidate_freeze_all_correct_hits": int(payload["candidate_freeze_all_correct_hits"]),
        "bootstrap_all_correct_raw_samples": str(payload["bootstrap_all_correct_raw_samples"]),
        "candidate_query_file": str(payload["candidate_query_file"]),
        "artifact_dir": state_path.parent,
        "bootstrap_pool_path": bootstrap_pool_path,
        "candidate_pool_path": candidate_pool_path,
        "state_path": state_path,
        "bootstrap_rows": bootstrap_rows,
        "candidate_rows": candidate_rows,
        "bootstrap_sampler_state": load_query_sampler_state(bootstrap_rows, payload.get("bootstrap_sampler_state")),
        "candidate_sampler_state": load_query_sampler_state(candidate_rows, payload.get("candidate_sampler_state")),
        "candidate_stats": dict(payload.get("candidate_stats", {})),
        "candidate_frozen_ids": set(payload.get("candidate_frozen_ids", [])),
        "last_updated_round": int(payload.get("last_updated_round", 0)),
    }


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


def sample_mixed_round_queries(
    state: dict[str, Any],
    *,
    requested_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bootstrap_target = round(requested_count * float(state["bootstrap_ratio"]))
    candidate_target = requested_count - bootstrap_target

    bootstrap_rows, bootstrap_report = _sample_or_empty(
        state["bootstrap_rows"],
        state["bootstrap_sampler_state"],
        requested_count=bootstrap_target,
    )
    selected_questions = {
        normalize_question(str(row.get("question", "")))
        for row in bootstrap_rows
    }
    candidate_rows, candidate_report = _sample_or_empty(
        state["candidate_rows"],
        state["candidate_sampler_state"],
        requested_count=candidate_target,
        skip_ids=set(state["candidate_frozen_ids"]),
        skip_questions=selected_questions,
    )
    selected_questions.update(
        normalize_question(str(row.get("question", "")))
        for row in candidate_rows
    )

    bootstrap_shortage = bootstrap_target - len(bootstrap_rows)
    candidate_shortage = candidate_target - len(candidate_rows)
    bootstrap_fill_rows: list[dict[str, Any]] = []
    candidate_fill_rows: list[dict[str, Any]] = []
    bootstrap_fill_report: dict[str, Any] | None = None
    candidate_fill_report: dict[str, Any] | None = None

    if bootstrap_shortage > 0:
        candidate_fill_rows, candidate_fill_report = _sample_or_empty(
            state["candidate_rows"],
            state["candidate_sampler_state"],
            requested_count=bootstrap_shortage,
            skip_ids=set(state["candidate_frozen_ids"]),
            skip_questions=selected_questions,
        )
        selected_questions.update(
            normalize_question(str(row.get("question", "")))
            for row in candidate_fill_rows
        )

    if candidate_shortage > 0:
        bootstrap_fill_rows, bootstrap_fill_report = _sample_or_empty(
            state["bootstrap_rows"],
            state["bootstrap_sampler_state"],
            requested_count=candidate_shortage,
            skip_questions=selected_questions,
        )
        selected_questions.update(
            normalize_question(str(row.get("question", "")))
            for row in bootstrap_fill_rows
        )

    sampled_rows = bootstrap_rows + candidate_rows + candidate_fill_rows + bootstrap_fill_rows
    shortage = requested_count - len(sampled_rows)
    return sampled_rows, {
        "query_strategy": STRATEGY_MIXED_BOOTSTRAP_CANDIDATE,
        "requested_query_count": requested_count,
        "effective_query_count": len(sampled_rows),
        "query_pool_size": len(state["bootstrap_rows"]) + len(state["candidate_rows"]),
        "source_mix": {
            "bootstrap_target": bootstrap_target,
            "candidate_target": candidate_target,
            "bootstrap_sampled": len(bootstrap_rows) + len(bootstrap_fill_rows),
            "candidate_sampled": len(candidate_rows) + len(candidate_fill_rows),
            "bootstrap_pool_size": len(state["bootstrap_rows"]),
            "candidate_pool_size": len(state["candidate_rows"]),
            "candidate_frozen_count": len(state["candidate_frozen_ids"]),
            "filled_from_bootstrap": len(bootstrap_fill_rows),
            "filled_from_candidate": len(candidate_fill_rows),
            "quota_shortage": shortage,
        },
        "bootstrap_sampling": bootstrap_report,
        "candidate_sampling": candidate_report,
        "bootstrap_fill_sampling": bootstrap_fill_report or {},
        "candidate_fill_sampling": candidate_fill_report or {},
    }


def update_mixed_query_strategy(
    state: dict[str, Any],
    *,
    raw_samples_path: str | Path,
    round_index: int,
) -> dict[str, Any]:
    raw_samples = read_jsonl(raw_samples_path)
    newly_frozen: list[str] = []
    candidate_sampled = 0
    candidate_all_correct = 0

    for sample in raw_samples:
        meta = dict(sample.get("meta", {}))
        if meta.get("on_policy_query_strategy_source") != MIXED_SOURCE_CANDIDATE:
            continue
        sample_id = str(sample.get("id", ""))
        if sample_id not in state["candidate_stats"]:
            continue
        candidate_sampled += 1
        response_summary = dict(sample.get("response_summary", {}))
        response_count = int(response_summary.get("response_count", 0))
        correct_count = int(response_summary.get("correct_count", 0))
        is_all_correct = response_count > 0 and correct_count == response_count
        stats = state["candidate_stats"][sample_id]
        stats["sample_hits"] += 1
        if is_all_correct:
            candidate_all_correct += 1
            stats["all_correct_hits"] += 1
        if not stats["frozen"] and int(stats["all_correct_hits"]) >= int(state["candidate_freeze_all_correct_hits"]):
            stats["frozen"] = True
            stats["frozen_after_round"] = round_index
            state["candidate_frozen_ids"].add(sample_id)
            newly_frozen.append(sample_id)

    state["last_updated_round"] = round_index
    persist_mixed_query_strategy_state(state)
    return {
        "candidate_sampled": candidate_sampled,
        "candidate_all_correct_this_round": candidate_all_correct,
        "candidate_newly_frozen": len(newly_frozen),
        "candidate_newly_frozen_ids": newly_frozen,
        "candidate_frozen_count": len(state["candidate_frozen_ids"]),
        "strategy_state_path": str(state["state_path"]),
    }
