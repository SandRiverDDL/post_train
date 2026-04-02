from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from post_train.config import OnPolicyLoopConfig
from post_train.io import ensure_parent

from .loop_paths import build_round_paths
from .strategy_common import normalize_question

FINISHED_STOP_REASONS = {
    "no_improvement",
    "max_rounds_reached",
    "completed",
    "insufficient_retained_data",
}


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def history_path(cfg: OnPolicyLoopConfig) -> Path:
    return cfg.round_base_dir / "history.json"


def final_summary_path(cfg: OnPolicyLoopConfig) -> Path:
    return cfg.round_base_dir / "final_summary.json"


def remove_loop_dirs(cfg: OnPolicyLoopConfig) -> None:
    for path in (cfg.round_base_dir, cfg.data_base_dir):
        if path.exists():
            shutil.rmtree(path)


def prepare_loop_run(
    cfg: OnPolicyLoopConfig,
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用。")
    history = history_path(cfg)
    final_summary = final_summary_path(cfg)
    has_existing_dirs = cfg.round_base_dir.exists() or cfg.data_base_dir.exists()
    if overwrite:
        remove_loop_dirs(cfg)
        return {
            "mode": "fresh",
            "history_path": history,
            "final_summary_path": final_summary,
        }
    if not has_existing_dirs:
        return {
            "mode": "fresh",
            "history_path": history,
            "final_summary_path": final_summary,
        }
    if not resume:
        raise ValueError(
            "自动循环输出目录已存在，请显式使用 --resume 继续未完成 run，或使用 --overwrite 重开。"
        )
    if history.exists():
        history_payload = read_json(history)
        stop_reason = str(history_payload.get("stop_reason", ""))
        if stop_reason in FINISHED_STOP_REASONS:
            raise ValueError(
                f"自动循环已经自然结束（stop_reason={stop_reason}），不能用 --resume 继续；请使用 --overwrite 重开。"
            )
        return {
            "mode": "resume",
            "history_path": history,
            "final_summary_path": final_summary,
            "history": history_payload,
        }
    if final_summary.exists():
        final_summary_payload = read_json(final_summary)
        stop_reason = str(final_summary_payload.get("stop_reason", ""))
        if stop_reason in FINISHED_STOP_REASONS:
            raise ValueError(
                f"自动循环已经自然结束（stop_reason={stop_reason}），不能用 --resume 继续；请使用 --overwrite 重开。"
            )
    raise ValueError("检测到已有 on-policy loop 目录，但缺少可恢复的 history.json；请使用 --overwrite 重开。")


def history_payload(
    *,
    loop_cfg: OnPolicyLoopConfig,
    rounds: list[dict[str, Any]],
    best_round_index: int | None,
    best_holdout_accuracy: float | None,
    best_model_path: str | None,
    current_model: str,
    no_improve_rounds: int,
    stop_reason: str | None,
    query_sampler_state: dict[str, Any] | None,
    anchor_sampler_state: dict[str, Any] | None,
    query_strategy_state_path: str | None,
) -> dict[str, Any]:
    return {
        "seed_model": loop_cfg.seed_model,
        "stop_dataset": str(loop_cfg.stop_dataset),
        "patience": loop_cfg.patience,
        "min_delta": loop_cfg.min_delta,
        "max_rounds": loop_cfg.max_rounds,
        "query_strategy": loop_cfg.query_strategy,
        "best_round_index": best_round_index,
        "best_holdout_accuracy": best_holdout_accuracy,
        "best_model_path": best_model_path or "",
        "current_model": current_model,
        "no_improve_rounds": no_improve_rounds,
        "last_completed_round": len(rounds),
        "stop_reason": stop_reason or "",
        "query_sampler_state": query_sampler_state,
        "anchor_sampler_state": anchor_sampler_state,
        "query_strategy_state_path": query_strategy_state_path or "",
        "rounds": rounds,
    }


def load_seen_query_questions(loop_cfg: OnPolicyLoopConfig, *, completed_rounds: int) -> set[str]:
    seen_questions: set[str] = set()
    for round_index in range(1, completed_rounds + 1):
        query_path = build_round_paths(loop_cfg, round_index)["query_output_path"]
        if not query_path.exists():
            continue
        rows = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            question = normalize_question(str(row.get("question", "")))
            if question:
                seen_questions.add(question)
    return seen_questions
