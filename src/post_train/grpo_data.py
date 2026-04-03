from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from datasets import Dataset

from post_train.data import read_json_rows
from post_train.io import ensure_parent, write_jsonl
from post_train.on_policy_query_strategy import normalize_question
from post_train.prompts import build_sft_prompt
from post_train.schemas import GRPORecord


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _sample_rows(rows: list[dict[str, Any]], *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size >= len(rows):
        return list(rows)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:sample_size]


def load_rd211_candidate_rows(cfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from post_train.data import load_dataset_rows

    kept_rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    duplicate_question_count = 0
    raw_rows = load_dataset_rows(cfg.rd211_dataset, split=cfg.dataset_split, cache_dir=cfg.cache_dir)
    solve_rate_kept = 0
    solve_rate_filtered = 0

    for index, row in enumerate(raw_rows):
        solve_rate = float(row["llama8b_solve_rate"])
        if not (cfg.solve_rate_lower < solve_rate < cfg.solve_rate_upper):
            solve_rate_filtered += 1
            continue
        solve_rate_kept += 1
        question = str(row.get("problem", "")).strip()
        question_key = normalize_question(question)
        if not question_key or question_key in seen_questions:
            duplicate_question_count += 1
            continue
        seen_questions.add(question_key)
        kept_rows.append(
            GRPORecord(
                id=f"{cfg.dataset_split}:{index}",
                prompt=build_sft_prompt(question),
                question=question,
                final_answer=str(row.get("answer", "")).strip(),
                meta={
                    "source": cfg.rd211_dataset,
                    "source_split": cfg.dataset_split,
                    "source_domain": str(row.get("domain", "")),
                    "source_dataset_row_source": str(row.get("source", "")),
                    "llama8b_solve_rate": solve_rate,
                },
            ).model_dump()
        )

    return kept_rows, {
        "dataset": cfg.rd211_dataset,
        "dataset_split": cfg.dataset_split,
        "raw_count": len(raw_rows),
        "solve_rate_kept": solve_rate_kept,
        "solve_rate_filtered": solve_rate_filtered,
        "solve_rate_lower": cfg.solve_rate_lower,
        "solve_rate_upper": cfg.solve_rate_upper,
        "kept_after_exact_dedup": len(kept_rows),
        "duplicate_question_count": duplicate_question_count,
    }


def load_anchor_rows(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = read_json_rows(path)
    kept_rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    duplicate_question_count = 0

    for index, row in enumerate(raw_rows):
        question = str(row.get("question", row.get("problem", ""))).strip()
        final_answer = str(row.get("final_answer", row.get("answer", ""))).strip()
        question_key = normalize_question(question)
        if not question_key:
            continue
        if question_key in seen_questions:
            duplicate_question_count += 1
            continue
        seen_questions.add(question_key)
        kept_rows.append(
            GRPORecord(
                id=str(row.get("id", row.get("problem_id", index))),
                prompt=build_sft_prompt(question),
                question=question,
                final_answer=final_answer,
                meta={
                    **dict(row.get("meta", {})),
                    "source": str(row.get("meta", {}).get("source", path)),
                    "anchor_dataset_path": str(path),
                },
            ).model_dump()
        )

    return kept_rows, {
        "path": str(path),
        "raw_count": len(raw_rows),
        "kept_after_exact_dedup": len(kept_rows),
        "duplicate_question_count": duplicate_question_count,
    }


def _best_mix_counts(*, rd211_count: int, anchor_count: int, anchor_share: float) -> tuple[int, int]:
    if anchor_share <= 0.0:
        return rd211_count, 0
    best_rd_count = 0
    best_anchor_count = 0
    best_error = float("inf")
    best_total = -1
    for anchor_sample_count in range(1, anchor_count + 1):
        ideal_rd_count = anchor_sample_count * (1.0 - anchor_share) / anchor_share
        for candidate_rd_count in {int(ideal_rd_count), int(ideal_rd_count) + 1}:
            if candidate_rd_count <= 0 or candidate_rd_count > rd211_count:
                continue
            total_count = anchor_sample_count + candidate_rd_count
            realized_share = anchor_sample_count / total_count
            share_error = abs(realized_share - anchor_share)
            if total_count > best_total or (total_count == best_total and share_error < best_error):
                best_total = total_count
                best_error = share_error
                best_rd_count = candidate_rd_count
                best_anchor_count = anchor_sample_count
    if best_total < 0:
        raise ValueError("无法在当前 rd211 与 anchor 规模下构造混合数据。")
    return best_rd_count, best_anchor_count


def build_grpo_dataset(cfg) -> tuple[Path, dict[str, Any]]:
    rd211_rows, rd211_report = load_rd211_candidate_rows(cfg)
    anchor_rows, anchor_report = load_anchor_rows(cfg.anchor_dataset_path)

    rd211_questions = {normalize_question(str(row["question"])) for row in rd211_rows}
    deduped_anchor_rows = [
        row
        for row in anchor_rows
        if normalize_question(str(row["question"])) not in rd211_questions
    ]
    cross_pool_duplicate_count = len(anchor_rows) - len(deduped_anchor_rows)

    rd211_sample_count, anchor_sample_count = _best_mix_counts(
        rd211_count=len(rd211_rows),
        anchor_count=len(deduped_anchor_rows),
        anchor_share=cfg.anchor_share,
    )
    sampled_rd211_rows = _sample_rows(rd211_rows, sample_size=rd211_sample_count, seed=cfg.seed)
    sampled_anchor_rows = _sample_rows(deduped_anchor_rows, sample_size=anchor_sample_count, seed=cfg.seed + 1)
    final_rows = sampled_rd211_rows + sampled_anchor_rows
    final_rows = _sample_rows(final_rows, sample_size=len(final_rows), seed=cfg.seed + 2)

    output_path = write_jsonl(cfg.output_path, final_rows)
    report = {
        "train_dataset": str(output_path),
        "train_count": len(final_rows),
        "anchor_share_target": cfg.anchor_share,
        "anchor_share_realized": (anchor_sample_count / len(final_rows)) if final_rows else 0.0,
        "rd211_count": len(sampled_rd211_rows),
        "anchor_count": len(sampled_anchor_rows),
        "cross_pool_duplicate_count": cross_pool_duplicate_count,
        "anchor_is_bottleneck": anchor_sample_count == len(deduped_anchor_rows),
        "rd211_dropped_for_ratio": len(rd211_rows) - rd211_sample_count,
        "rd211": rd211_report,
        "anchor": {
            **anchor_report,
            "kept_after_cross_pool_exact_dedup": len(deduped_anchor_rows),
        },
    }
    report_path = _write_json(cfg.report_path, report)
    return output_path, {
        "output_path": output_path,
        "report_path": report_path,
        "report": report,
    }


def load_grpo_dataset(path: str | Path) -> Dataset:
    rows = [GRPORecord.model_validate(row).model_dump() for row in read_json_rows(path)]
    return Dataset.from_list(rows)


def select_grpo_train_subset(
    dataset: Dataset,
    *,
    max_train_samples: int | None,
    seed: int,
    mode: str,
) -> Dataset:
    if max_train_samples is None or max_train_samples >= len(dataset):
        return dataset

    if mode == "head":
        return dataset.select(range(max_train_samples))
    if mode != "fixed_random":
        raise ValueError(f"未知的 GRPO 训练子集模式: {mode}")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return dataset.select(indices[:max_train_samples])
