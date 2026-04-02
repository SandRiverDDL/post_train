from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_train.io import ensure_parent, read_jsonl, write_jsonl

from .strategy_common import (
    STRATEGY_MIXED_BOOTSTRAP_CANDIDATE,
    _sample_or_empty,
    _strategy_artifact_dir,
    _write_json,
    build_query_sampler_state,
    dump_query_sampler_state,
    load_bootstrap_all_correct_pool,
    load_candidate_query_pool,
    load_query_sampler_state,
    normalize_question,
)


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
    ensure_parent(Path(state["artifact_dir"]) / ".placeholder")
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
        if meta.get("on_policy_query_strategy_source") != "candidate_pool":
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
