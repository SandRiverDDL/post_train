from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from post_train.io import read_jsonl, write_jsonl
from post_train.on_policy.query_strategy import normalize_question, sample_round_queries

ANCHOR_SOURCE = "stage1_anchor"
MIXED_SOURCE = "mixed_retained"


def load_anchor_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    anchor_rows: list[dict[str, Any]] = []
    for row in rows:
        meta = dict(row.get("meta", {}))
        meta["on_policy_trainset_source"] = ANCHOR_SOURCE
        anchor_rows.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "solution": str(row.get("solution", "")),
                "final_answer": str(row.get("final_answer", "")),
                "meta": meta,
            }
        )
    return anchor_rows


def build_mixed_plus_anchor_dataset(
    *,
    mixed_retained_path: str | Path,
    anchor_rows: list[dict[str, Any]],
    anchor_sampler_state: dict[str, Any],
    anchor_share: float,
    output_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    mixed_rows = read_jsonl(mixed_retained_path)
    if not mixed_rows:
        raise ValueError("mixed retained 数据为空，无法构造 mixed_plus_anchor 训练集。")

    mixed_rows_out: list[dict[str, Any]] = []
    mixed_ids: set[str] = set()
    mixed_questions: set[str] = set()
    for row in mixed_rows:
        meta = dict(row.get("meta", {}))
        meta["on_policy_trainset_source"] = MIXED_SOURCE
        copied = dict(row)
        copied["meta"] = meta
        mixed_rows_out.append(copied)
        mixed_ids.add(str(row.get("id", "")))
        mixed_questions.add(normalize_question(str(row.get("question", ""))))

    mixed_count = len(mixed_rows_out)
    anchor_target = math.ceil((mixed_count * anchor_share) / (1.0 - anchor_share))
    try:
        anchor_rows_out, anchor_sampling_report = sample_round_queries(
            anchor_rows,
            anchor_sampler_state,
            requested_count=anchor_target,
            skip_ids=mixed_ids,
            skip_questions=mixed_questions,
        )
    except ValueError:
        anchor_rows_out = []
        anchor_sampling_report = {
            "query_pool_size": len(anchor_rows),
            "eligible_query_count": 0,
            "requested_query_count": anchor_target,
            "effective_query_count": 0,
            "crossed_epoch": False,
            "epoch_index_start": int(anchor_sampler_state["epoch_index"]),
            "epoch_offset_start": int(anchor_sampler_state["epoch_offset"]),
            "epoch_index_end": int(anchor_sampler_state["epoch_index"]),
            "epoch_offset_end": int(anchor_sampler_state["epoch_offset"]),
        }
    final_rows = mixed_rows_out + anchor_rows_out
    write_jsonl(output_path, final_rows)
    return Path(output_path), {
        "train_selector": "mixed_plus_anchor",
        "mixed_count": mixed_count,
        "anchor_target": anchor_target,
        "anchor_count": len(anchor_rows_out),
        "train_sample_count": len(final_rows),
        "anchor_share": anchor_share,
        "anchor_sampling": anchor_sampling_report,
    }
