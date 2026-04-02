from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_train.io import ensure_parent, read_jsonl, write_jsonl

from .strategy_common import (
    MIXED_SOURCE_RANDOM,
    STRATEGY_CANDIDATE_RANDOM_MIX,
    _copy_query_row,
    _sample_or_empty,
    _strategy_artifact_dir,
    _write_json,
    build_query_sampler_state,
    dump_query_sampler_state,
    load_candidate_query_pool,
    load_query_sampler_state,
    normalize_question,
)


def build_candidate_random_query_strategy(
    *,
    random_query_rows: list[dict[str, Any]],
    candidate_query_file: str | Path,
    data_base_dir: str | Path,
    seed: int,
    candidate_ratio: float,
    random_ratio: float,
) -> dict[str, Any]:
    random_rows = [_copy_query_row(row, source_name=MIXED_SOURCE_RANDOM) for row in random_query_rows]
    candidate_rows = load_candidate_query_pool(candidate_query_file)
    if not random_rows and not candidate_rows:
        raise ValueError("candidate_random_mix 没有可用 query。")
    strategy_dir = _strategy_artifact_dir(data_base_dir)
    state = {
        "strategy_name": STRATEGY_CANDIDATE_RANDOM_MIX,
        "seed": seed,
        "candidate_ratio": candidate_ratio,
        "random_ratio": random_ratio,
        "artifact_dir": strategy_dir,
        "random_pool_path": strategy_dir / "random_pool.jsonl",
        "candidate_pool_path": strategy_dir / "candidate_pool.jsonl",
        "state_path": strategy_dir / "state.json",
        "random_rows": random_rows,
        "candidate_rows": candidate_rows,
        "random_sampler_state": build_query_sampler_state(random_rows, seed=seed) if random_rows else None,
        "candidate_sampler_state": build_query_sampler_state(candidate_rows, seed=seed + 1000) if candidate_rows else None,
    }
    persist_candidate_random_query_strategy_state(state)
    return state


def persist_candidate_random_query_strategy_state(state: dict[str, Any]) -> Path:
    ensure_parent(Path(state["artifact_dir"]) / ".placeholder")
    write_jsonl(state["random_pool_path"], state["random_rows"])
    write_jsonl(state["candidate_pool_path"], state["candidate_rows"])
    payload = {
        "strategy_name": state["strategy_name"],
        "seed": int(state["seed"]),
        "candidate_ratio": float(state["candidate_ratio"]),
        "random_ratio": float(state["random_ratio"]),
        "random_pool_path": str(state["random_pool_path"]),
        "candidate_pool_path": str(state["candidate_pool_path"]),
        "random_pool_size": len(state["random_rows"]),
        "candidate_pool_size": len(state["candidate_rows"]),
        "random_sampler_state": dump_query_sampler_state(state["random_sampler_state"]),
        "candidate_sampler_state": dump_query_sampler_state(state["candidate_sampler_state"]),
    }
    return _write_json(state["state_path"], payload)


def load_candidate_random_query_strategy_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    random_pool_path = Path(payload["random_pool_path"])
    candidate_pool_path = Path(payload["candidate_pool_path"])
    random_rows = read_jsonl(random_pool_path)
    candidate_rows = read_jsonl(candidate_pool_path)
    return {
        "strategy_name": payload["strategy_name"],
        "seed": int(payload["seed"]),
        "candidate_ratio": float(payload["candidate_ratio"]),
        "random_ratio": float(payload["random_ratio"]),
        "artifact_dir": state_path.parent,
        "random_pool_path": random_pool_path,
        "candidate_pool_path": candidate_pool_path,
        "state_path": state_path,
        "random_rows": random_rows,
        "candidate_rows": candidate_rows,
        "random_sampler_state": load_query_sampler_state(random_rows, payload.get("random_sampler_state")),
        "candidate_sampler_state": load_query_sampler_state(candidate_rows, payload.get("candidate_sampler_state")),
    }


def sample_candidate_random_round_queries(
    state: dict[str, Any],
    *,
    requested_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_target = round(requested_count * float(state["candidate_ratio"]))
    random_target = requested_count - candidate_target

    candidate_rows, candidate_report = _sample_or_empty(
        state["candidate_rows"],
        state["candidate_sampler_state"],
        requested_count=candidate_target,
    )
    selected_questions = {
        normalize_question(str(row.get("question", "")))
        for row in candidate_rows
    }
    random_rows, random_report = _sample_or_empty(
        state["random_rows"],
        state["random_sampler_state"],
        requested_count=random_target,
        skip_questions=selected_questions,
    )
    selected_questions.update(
        normalize_question(str(row.get("question", "")))
        for row in random_rows
    )

    candidate_shortage = candidate_target - len(candidate_rows)
    random_shortage = random_target - len(random_rows)
    random_fill_rows: list[dict[str, Any]] = []
    candidate_fill_rows: list[dict[str, Any]] = []
    random_fill_report: dict[str, Any] | None = None
    candidate_fill_report: dict[str, Any] | None = None

    if candidate_shortage > 0:
        random_fill_rows, random_fill_report = _sample_or_empty(
            state["random_rows"],
            state["random_sampler_state"],
            requested_count=candidate_shortage,
            skip_questions=selected_questions,
        )
        selected_questions.update(
            normalize_question(str(row.get("question", "")))
            for row in random_fill_rows
        )

    if random_shortage > 0:
        candidate_fill_rows, candidate_fill_report = _sample_or_empty(
            state["candidate_rows"],
            state["candidate_sampler_state"],
            requested_count=random_shortage,
            skip_questions=selected_questions,
        )
        selected_questions.update(
            normalize_question(str(row.get("question", "")))
            for row in candidate_fill_rows
        )

    sampled_rows = candidate_rows + random_rows + random_fill_rows + candidate_fill_rows
    shortage = requested_count - len(sampled_rows)
    return sampled_rows, {
        "query_strategy": STRATEGY_CANDIDATE_RANDOM_MIX,
        "requested_query_count": requested_count,
        "effective_query_count": len(sampled_rows),
        "query_pool_size": len(state["candidate_rows"]) + len(state["random_rows"]),
        "source_mix": {
            "candidate_target": candidate_target,
            "random_target": random_target,
            "candidate_sampled": len(candidate_rows) + len(candidate_fill_rows),
            "random_sampled": len(random_rows) + len(random_fill_rows),
            "candidate_pool_size": len(state["candidate_rows"]),
            "random_pool_size": len(state["random_rows"]),
            "filled_from_candidate": len(candidate_fill_rows),
            "filled_from_random": len(random_fill_rows),
            "quota_shortage": shortage,
        },
        "candidate_sampling": candidate_report,
        "random_sampling": random_report,
        "candidate_fill_sampling": candidate_fill_report or {},
        "random_fill_sampling": random_fill_report or {},
    }
