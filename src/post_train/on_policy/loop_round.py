from __future__ import annotations

from pathlib import Path
from typing import Any

from post_train.config import EvalConfig, EvalTaskConfig, OnPolicyDataConfig, OnPolicyLoopConfig, SFTTrainConfig
from post_train.eval import run_eval_task
from post_train.on_policy.data import prepare_on_policy_sft_dataset_from_queries
from post_train.on_policy.trainset import build_mixed_plus_anchor_dataset
from post_train.sft import train_sft
from post_train.sft_selection import run_checkpoint_selection

from .loop_paths import build_round_train_config
from .strategy_candidate_random import (
    persist_candidate_random_query_strategy_state,
    sample_candidate_random_round_queries,
)
from .strategy_common import STRATEGY_CANDIDATE_RANDOM_MIX, STRATEGY_MIXED_BOOTSTRAP_CANDIDATE, normalize_question, sample_round_queries
from .strategy_mixed import sample_mixed_round_queries, update_mixed_query_strategy


def _effective_eval_setting(
    loop_value: int | str | None,
    default_value: int | str | None,
) -> int | str | None:
    if loop_value is None:
        return default_value
    return loop_value


def evaluate_single_dataset(
    *,
    model_name: str,
    dataset_path: str | Path,
    eval_cfg: EvalConfig,
    output_dir: str | Path,
    backend: str | None = None,
    batch_size: int | str | None = None,
    max_new_tokens: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    resolved_backend = backend or eval_cfg.backend
    if resolved_backend != "vllm":
        raise ValueError("当前评测只支持 backend=vllm。")
    resolved_batch_size = _effective_eval_setting(batch_size, eval_cfg.batch_size)
    resolved_max_new_tokens = int(
        _effective_eval_setting(max_new_tokens, eval_cfg.max_new_tokens) or eval_cfg.max_new_tokens
    )
    task_name = Path(dataset_path).stem.replace("-", "_")
    task = EvalTaskConfig(
        name=task_name,
        dataset_path=Path(dataset_path),
        output_path=Path(output_dir) / task_name,
    )
    run_result = run_eval_task(
        model_name=model_name,
        task=task,
        eval_cfg=eval_cfg.model_copy(update={"backend": resolved_backend}),
        batch_size=resolved_batch_size,
        max_new_tokens=resolved_max_new_tokens,
        limit=limit,
        output_dir=output_dir,
    )
    return {
        "result": run_result["result"],
        "result_path": str(run_result["result_path"]),
        "raw_result_path": str(run_result["raw_result_path"]),
    }


def is_improved(current: float, best_so_far: float | None, *, min_delta: float) -> bool:
    if best_so_far is None:
        return True
    return current > (best_so_far + min_delta)


def sample_queries_for_round(
    *,
    cfg: OnPolicyLoopConfig,
    query_rows: list[dict[str, Any]],
    query_pool_report: dict[str, Any],
    query_sampler_state: dict[str, Any] | None,
    mixed_strategy_state: dict[str, Any] | None,
    candidate_random_strategy_state: dict[str, Any] | None,
    seen_query_questions: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if cfg.query_strategy == STRATEGY_MIXED_BOOTSTRAP_CANDIDATE:
        if mixed_strategy_state is None:
            raise ValueError("mixed strategy state 未初始化。")
        round_queries, sampler_report = sample_mixed_round_queries(
            mixed_strategy_state,
            requested_count=cfg.round_query_count,
        )
    elif cfg.query_strategy == STRATEGY_CANDIDATE_RANDOM_MIX:
        if candidate_random_strategy_state is None:
            raise ValueError("candidate_random strategy state 未初始化。")
        round_queries, sampler_report = sample_candidate_random_round_queries(
            candidate_random_strategy_state,
            requested_count=cfg.round_query_count,
        )
    else:
        if query_sampler_state is None:
            raise ValueError("uniform sampler state 未初始化。")
        round_queries, sampler_report = sample_round_queries(
            query_rows,
            query_sampler_state,
            requested_count=cfg.round_query_count,
        )
    round_query_report = dict(query_pool_report)
    round_query_report["sampled_queries"] = len(round_queries)
    round_query_report["effective_query_count"] = sampler_report["effective_query_count"]
    round_query_report["round_query_count"] = cfg.round_query_count
    round_query_report["query_strategy"] = cfg.query_strategy
    current_round_questions = {
        normalize_question(str(row.get("question", "")))
        for row in round_queries
    }
    reused_query_count = len(current_round_questions & seen_query_questions)
    round_query_report["query_reuse_ratio"] = (
        reused_query_count / len(current_round_questions)
        if current_round_questions
        else 0.0
    )
    round_query_report["reused_query_count"] = reused_query_count
    return round_queries, sampler_report, round_query_report


def build_round_trainset(
    *,
    cfg: OnPolicyLoopConfig,
    round_paths: dict[str, Path],
    data_result: dict[str, Any],
    anchor_rows: list[dict[str, Any]],
    anchor_sampler_state: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    if cfg.train_selector == "mixed_plus_anchor":
        if anchor_sampler_state is None:
            raise ValueError("anchor sampler state 未初始化。")
        return build_mixed_plus_anchor_dataset(
            mixed_retained_path=data_result["retained_output_paths"]["mixed_only_shortest"],
            anchor_rows=anchor_rows,
            anchor_sampler_state=anchor_sampler_state,
            anchor_share=cfg.anchor_share,
            output_path=round_paths["train_dataset_path"],
        )
    train_dataset_path = Path(data_result["retained_output_path"])
    retained_rows = int(data_result["report"]["retained"]["kept"])
    return train_dataset_path, {
        "train_selector": "primary_retained",
        "mixed_count": retained_rows,
        "anchor_target": 0,
        "anchor_count": 0,
        "train_sample_count": retained_rows,
        "anchor_share": 0.0,
        "anchor_sampling": {},
    }


def train_and_select_round_model(
    *,
    cfg: OnPolicyLoopConfig,
    base_train_cfg: SFTTrainConfig,
    eval_cfg: EvalConfig,
    round_index: int,
    round_paths: dict[str, Path],
    current_model: str,
    train_dataset_path: Path,
    trainset_report: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    train_cfg = build_round_train_config(
        base_train_cfg,
        cfg,
        round_index=round_index,
        model_name=current_model,
        train_dataset=train_dataset_path,
        train_sample_count=int(trainset_report["train_sample_count"]),
    )
    train_output_dir = train_sft(train_cfg)
    if cfg.checkpoint_selection_enabled:
        selection_result = run_checkpoint_selection(
            train_output_dir=train_output_dir,
            dataset_path=cfg.stop_dataset,
            eval_cfg=eval_cfg,
            backend=cfg.backend,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_new_tokens,
            limit=cfg.limit,
        )
        best_checkpoint = selection_result["best"]
        return Path(train_output_dir), {
            "selected_model_path": str(best_checkpoint["checkpoint_path"]),
            "selected_global_step": int(best_checkpoint["global_step"]),
            "holdout_metrics": best_checkpoint["metrics"],
            "holdout_result_path": str(best_checkpoint["result_path"]),
            "holdout_raw_result_path": str(best_checkpoint["raw_result_path"]),
            "checkpoint_ranking_path": str(selection_result["ranking_path"]),
            "checkpoint_best_path": str(selection_result["best_path"]),
        }
    holdout_eval = evaluate_single_dataset(
        model_name=str(train_output_dir),
        dataset_path=cfg.stop_dataset,
        eval_cfg=eval_cfg,
        output_dir=round_paths["output_dir"] / "holdout_eval",
        backend=cfg.backend,
        batch_size=cfg.batch_size,
        max_new_tokens=cfg.max_new_tokens,
        limit=cfg.limit,
    )
    return Path(train_output_dir), {
        "selected_model_path": str(train_output_dir),
        "selected_global_step": 0,
        "holdout_metrics": holdout_eval["result"]["metrics"],
        "holdout_result_path": str(holdout_eval["result_path"]),
        "holdout_raw_result_path": str(holdout_eval["raw_result_path"]),
        "checkpoint_ranking_path": "",
        "checkpoint_best_path": "",
    }


def update_round_strategy(
    *,
    cfg: OnPolicyLoopConfig,
    mixed_strategy_state: dict[str, Any] | None,
    candidate_random_strategy_state: dict[str, Any] | None,
    data_result: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    if cfg.query_strategy == STRATEGY_MIXED_BOOTSTRAP_CANDIDATE:
        if mixed_strategy_state is None:
            raise ValueError("mixed strategy state 未初始化。")
        return update_mixed_query_strategy(
            mixed_strategy_state,
            raw_samples_path=data_result["raw_samples_output_path"],
            round_index=round_index,
        )
    if cfg.query_strategy == STRATEGY_CANDIDATE_RANDOM_MIX:
        if candidate_random_strategy_state is None:
            raise ValueError("candidate_random strategy state 未初始化。")
        persist_candidate_random_query_strategy_state(candidate_random_strategy_state)
        return {
            "candidate_pool_size": len(candidate_random_strategy_state["candidate_rows"]),
            "random_pool_size": len(candidate_random_strategy_state["random_rows"]),
        }
    return {}


def build_round_summary(
    *,
    cfg: OnPolicyLoopConfig,
    round_index: int,
    round_name: str,
    current_model: str,
    data_result: dict[str, Any],
    train_dataset_path: Path,
    trainset_report: dict[str, Any],
    sampler_report: dict[str, Any],
    strategy_update: dict[str, Any],
    train_output_dir: Path,
    selection_info: dict[str, Any],
    holdout_accuracy: float,
    improved: bool,
    best_holdout_accuracy: float | None,
    no_improve_rounds: int,
) -> dict[str, Any]:
    retained_report = data_result["report"]["retained"]
    return {
        "round_index": round_index,
        "round_name": round_name,
        "generation_model": current_model,
        "query_strategy": cfg.query_strategy,
        "primary_selector": str(data_result["report"].get("primary_selector", "")),
        "train_selector": str(trainset_report["train_selector"]),
        "round_query_count": cfg.round_query_count,
        "retained_count": int(retained_report["kept"]),
        "retained_ratio": float(retained_report["retained_ratio"]),
        "mixed_count": int(trainset_report["mixed_count"]),
        "anchor_count": int(trainset_report["anchor_count"]),
        "anchor_target": int(trainset_report["anchor_target"]),
        "train_sample_count": int(trainset_report["train_sample_count"]),
        "train_dataset_path": str(train_dataset_path),
        "query_sampling": sampler_report,
        "trainset_sampling": trainset_report,
        "query_strategy_update": strategy_update,
        "round_model_path": selection_info["selected_model_path"],
        "round_output_dir": str(train_output_dir),
        "selected_checkpoint_path": selection_info["selected_model_path"],
        "selected_checkpoint_step": selection_info["selected_global_step"],
        "checkpoint_ranking_path": selection_info["checkpoint_ranking_path"],
        "checkpoint_best_path": selection_info["checkpoint_best_path"],
        "stop_dataset": str(cfg.stop_dataset),
        "holdout_accuracy": holdout_accuracy,
        "improved": improved,
        "best_holdout_accuracy_so_far": best_holdout_accuracy,
        "no_improve_rounds": no_improve_rounds,
        "holdout_result_path": selection_info["holdout_result_path"],
        "holdout_raw_result_path": selection_info["holdout_raw_result_path"],
        "data_report_path": str(data_result["report_path"]),
        "retained_output_paths": data_result.get("retained_output_paths", {}),
    }
