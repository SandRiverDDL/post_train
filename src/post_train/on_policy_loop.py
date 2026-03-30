from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from post_train.config import (
    EvalConfig,
    EvalTaskConfig,
    OnPolicyDataConfig,
    OnPolicyLoopConfig,
    SFTTrainConfig,
    load_eval_config,
    load_on_policy_data_config,
    load_sft_config,
)
from post_train.eval import (
    run_eval_task,
)
from post_train.io import ensure_parent
from post_train.on_policy_data import (
    InsufficientRetainedSamplesError,
    load_query_candidates,
    prepare_on_policy_sft_dataset_from_queries,
)
from post_train.on_policy_query_strategy import (
    STRATEGY_CANDIDATE_RANDOM_MIX,
    STRATEGY_MIXED_BOOTSTRAP_CANDIDATE,
    build_candidate_random_query_strategy,
    build_mixed_query_strategy,
    build_query_sampler_state,
    dump_query_sampler_state,
    load_candidate_random_query_strategy_state,
    load_mixed_query_strategy_state,
    load_query_sampler_state,
    persist_candidate_random_query_strategy_state,
    sample_candidate_random_round_queries,
    sample_mixed_round_queries,
    sample_round_queries,
    update_mixed_query_strategy,
    normalize_question,
)
from post_train.on_policy_trainset import (
    build_mixed_plus_anchor_dataset,
    load_anchor_rows,
)
from post_train.sft import train_sft
from post_train.sft_selection import run_checkpoint_selection

FINISHED_STOP_REASONS = {
    "no_improvement",
    "max_rounds_reached",
    "completed",
    "insufficient_retained_data",
}


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_round_name(round_index: int) -> str:
    return f"round{round_index}"


def build_round_paths(cfg: OnPolicyLoopConfig, round_index: int) -> dict[str, Path]:
    round_name = build_round_name(round_index)
    data_dir = cfg.data_base_dir / round_name
    output_dir = cfg.round_base_dir / round_name
    return {
        "round_name": Path(round_name),
        "data_dir": data_dir,
        "output_dir": output_dir,
        "query_output_path": data_dir / "query_pool.jsonl",
        "raw_samples_output_path": data_dir / "raw_samples.jsonl",
        "retained_output_path": data_dir / "train.jsonl",
        "train_dataset_path": data_dir / "train.mixed_plus_anchor.jsonl",
        "report_path": data_dir / "train.report.json",
        "holdout_output_path": output_dir / "holdout_eval" / "result.json",
        "holdout_raw_output_path": output_dir / "holdout_eval" / "raw.json",
        "round_summary_path": output_dir / "round_summary.json",
    }


def build_round_data_config(
    base_cfg: OnPolicyDataConfig,
    loop_cfg: OnPolicyLoopConfig,
    *,
    round_index: int,
    generation_model: str,
) -> OnPolicyDataConfig:
    paths = build_round_paths(loop_cfg, round_index)
    return base_cfg.model_copy(
        update={
            "round_name": build_round_name(round_index),
            "generation_model": generation_model,
            "query_output_path": paths["query_output_path"],
            "raw_samples_output_path": paths["raw_samples_output_path"],
            "retained_output_path": paths["retained_output_path"],
            "report_path": paths["report_path"],
        }
    )


def build_round_train_config(
    base_cfg: SFTTrainConfig,
    loop_cfg: OnPolicyLoopConfig,
    *,
    round_index: int,
    model_name: str,
    train_dataset: str | Path,
    train_sample_count: int,
) -> SFTTrainConfig:
    paths = build_round_paths(loop_cfg, round_index)
    effective_batch = max(1, int(base_cfg.batch_size) * int(base_cfg.gradient_accumulation_steps))
    estimated_total_steps = max(1, math.ceil((train_sample_count * float(loop_cfg.train_epochs)) / effective_batch))
    save_steps = max(1, estimated_total_steps // int(loop_cfg.checkpoint_target_count))
    return base_cfg.model_copy(
        update={
            "model_name": model_name,
            "train_dataset": Path(train_dataset),
            "output_dir": paths["output_dir"],
            "learning_rate": loop_cfg.train_learning_rate,
            "epochs": loop_cfg.train_epochs,
            "save_strategy": "steps" if loop_cfg.checkpoint_selection_enabled else "no",
            "save_steps": save_steps if loop_cfg.checkpoint_selection_enabled else None,
            "save_total_limit": (
                loop_cfg.checkpoint_save_total_limit
                if loop_cfg.checkpoint_selection_enabled
                else None
            ),
            "export_final_model": not loop_cfg.checkpoint_selection_enabled,
        }
    )


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


def _is_improved(current: float, best_so_far: float | None, *, min_delta: float) -> bool:
    if best_so_far is None:
        return True
    return current > (best_so_far + min_delta)


def _remove_loop_dirs(cfg: OnPolicyLoopConfig) -> None:
    for path in (cfg.round_base_dir, cfg.data_base_dir):
        if path.exists():
            shutil.rmtree(path)


def _history_path(cfg: OnPolicyLoopConfig) -> Path:
    return cfg.round_base_dir / "history.json"


def _final_summary_path(cfg: OnPolicyLoopConfig) -> Path:
    return cfg.round_base_dir / "final_summary.json"


def _prepare_loop_run(
    cfg: OnPolicyLoopConfig,
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用。")
    history_path = _history_path(cfg)
    final_summary_path = _final_summary_path(cfg)
    has_existing_dirs = cfg.round_base_dir.exists() or cfg.data_base_dir.exists()
    if overwrite:
        _remove_loop_dirs(cfg)
        return {
            "mode": "fresh",
            "history_path": history_path,
            "final_summary_path": final_summary_path,
        }
    if not has_existing_dirs:
        return {
            "mode": "fresh",
            "history_path": history_path,
            "final_summary_path": final_summary_path,
        }
    if not resume:
        raise ValueError(
            "自动循环输出目录已存在，请显式使用 --resume 继续未完成 run，或使用 --overwrite 重开。"
        )
    if history_path.exists():
        history = _read_json(history_path)
        stop_reason = str(history.get("stop_reason", ""))
        if stop_reason in FINISHED_STOP_REASONS:
            raise ValueError(
                f"自动循环已经自然结束（stop_reason={stop_reason}），不能用 --resume 继续；请使用 --overwrite 重开。"
            )
        return {
            "mode": "resume",
            "history_path": history_path,
            "final_summary_path": final_summary_path,
            "history": history,
        }
    if final_summary_path.exists():
        final_summary = _read_json(final_summary_path)
        stop_reason = str(final_summary.get("stop_reason", ""))
        if stop_reason in FINISHED_STOP_REASONS:
            raise ValueError(
                f"自动循环已经自然结束（stop_reason={stop_reason}），不能用 --resume 继续；请使用 --overwrite 重开。"
            )
    raise ValueError("检测到已有 on-policy loop 目录，但缺少可恢复的 history.json；请使用 --overwrite 重开。")


def _history_payload(
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


def _load_seen_query_questions(loop_cfg: OnPolicyLoopConfig, *, completed_rounds: int) -> set[str]:
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

def run_on_policy_loop(
    cfg: OnPolicyLoopConfig,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_state = _prepare_loop_run(cfg, resume=resume, overwrite=overwrite)
    base_data_cfg = load_on_policy_data_config(cfg.base_on_policy_data_config)
    base_train_cfg = load_sft_config(cfg.base_sft_config)
    eval_cfg = load_eval_config(cfg.eval_config)
    history_path = run_state["history_path"]
    final_summary_path = run_state["final_summary_path"]

    restored_history = run_state.get("history", {})
    anchor_rows: list[dict[str, Any]] = []
    anchor_sampler_state: dict[str, Any] | None = None
    if cfg.train_selector == "mixed_plus_anchor":
        if cfg.anchor_dataset_path is None:
            raise ValueError("mixed_plus_anchor 需要 anchor_dataset_path。")
        anchor_rows = load_anchor_rows(cfg.anchor_dataset_path)
        if run_state["mode"] == "resume":
            anchor_sampler_state = load_query_sampler_state(anchor_rows, restored_history.get("anchor_sampler_state"))
            if anchor_sampler_state is None:
                raise ValueError("history 缺少 anchor sampler state，无法安全 resume；请使用 --overwrite 重开。")
        else:
            anchor_sampler_state = build_query_sampler_state(anchor_rows, seed=base_data_cfg.seed + 2000)
    mixed_strategy_state: dict[str, Any] | None = None
    candidate_random_strategy_state: dict[str, Any] | None = None
    if cfg.query_strategy == STRATEGY_MIXED_BOOTSTRAP_CANDIDATE:
        if run_state["mode"] == "resume":
            strategy_state_path = restored_history.get("query_strategy_state_path") or (
                cfg.data_base_dir / "query_strategy" / "state.json"
            )
            mixed_strategy_state = load_mixed_query_strategy_state(strategy_state_path)
            if int(mixed_strategy_state.get("last_updated_round", 0)) != int(restored_history.get("last_completed_round", 0)):
                raise ValueError("mixed query strategy 状态与 history 不一致，无法安全 resume；请使用 --overwrite 重开。")
        else:
            mixed_strategy_state = build_mixed_query_strategy(
                bootstrap_all_correct_raw_samples=cfg.bootstrap_all_correct_raw_samples,
                candidate_query_file=cfg.candidate_query_file,
                data_base_dir=cfg.data_base_dir,
                seed=base_data_cfg.seed,
                bootstrap_ratio=cfg.bootstrap_ratio,
                candidate_ratio=cfg.candidate_ratio,
                candidate_freeze_all_correct_hits=cfg.candidate_freeze_all_correct_hits,
            )
        query_rows: list[dict[str, Any]] = []
        query_pool_report = {
            "query_strategy": cfg.query_strategy,
            "bootstrap_all_correct_raw_samples": str(cfg.bootstrap_all_correct_raw_samples),
            "candidate_query_file": str(cfg.candidate_query_file),
            "bootstrap_pool_size": len(mixed_strategy_state["bootstrap_rows"]),
            "candidate_pool_size": len(mixed_strategy_state["candidate_rows"]),
        }
        query_sampler_state = None
    elif cfg.query_strategy == STRATEGY_CANDIDATE_RANDOM_MIX:
        query_rows, query_pool_report = load_query_candidates(base_data_cfg)
        if run_state["mode"] == "resume":
            strategy_state_path = restored_history.get("query_strategy_state_path") or (
                cfg.data_base_dir / "query_strategy" / "state.json"
            )
            candidate_random_strategy_state = load_candidate_random_query_strategy_state(strategy_state_path)
        else:
            candidate_random_strategy_state = build_candidate_random_query_strategy(
                random_query_rows=query_rows,
                candidate_query_file=cfg.candidate_query_file,
                data_base_dir=cfg.data_base_dir,
                seed=base_data_cfg.seed,
                candidate_ratio=cfg.candidate_ratio,
                random_ratio=cfg.random_ratio,
            )
        query_pool_report = {
            "query_strategy": cfg.query_strategy,
            "candidate_query_file": str(cfg.candidate_query_file),
            "random_pool_size": len(candidate_random_strategy_state["random_rows"]),
            "candidate_pool_size": len(candidate_random_strategy_state["candidate_rows"]),
        }
        query_sampler_state = None
    else:
        query_rows, query_pool_report = load_query_candidates(base_data_cfg)
        if run_state["mode"] == "resume":
            query_sampler_state = load_query_sampler_state(query_rows, restored_history.get("query_sampler_state"))
            if query_sampler_state is None:
                raise ValueError("history 缺少 uniform sampler state，无法安全 resume；请使用 --overwrite 重开。")
        else:
            query_sampler_state = build_query_sampler_state(query_rows, seed=base_data_cfg.seed)

    current_model = str(restored_history.get("current_model") or cfg.seed_model)
    best_holdout_accuracy = (
        float(restored_history["best_holdout_accuracy"])
        if restored_history.get("best_holdout_accuracy") is not None
        else None
    )
    best_round_index = (
        int(restored_history["best_round_index"])
        if restored_history.get("best_round_index") is not None
        else None
    )
    best_model_path = str(restored_history.get("best_model_path") or "") or None
    no_improve_rounds = int(restored_history.get("no_improve_rounds", 0))
    stop_reason: str | None = None
    round_summaries: list[dict[str, Any]] = list(restored_history.get("rounds", []))
    start_round_index = int(restored_history.get("last_completed_round", 0)) + 1
    seen_query_questions = _load_seen_query_questions(cfg, completed_rounds=start_round_index - 1)

    for round_index in range(start_round_index, cfg.max_rounds + 1):
        round_name = build_round_name(round_index)
        round_paths = build_round_paths(cfg, round_index)
        try:
            data_cfg = build_round_data_config(
                base_data_cfg,
                cfg,
                round_index=round_index,
                generation_model=current_model,
            )
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
            data_result = prepare_on_policy_sft_dataset_from_queries(
                data_cfg,
                query_rows=round_queries,
                query_report=round_query_report,
            )
        except InsufficientRetainedSamplesError as exc:
            stop_reason = "insufficient_retained_data"
            round_summary = {
                "round_index": round_index,
                "round_name": round_name,
                "generation_model": current_model,
                "stop_reason": stop_reason,
                "error": str(exc),
            }
            _write_json(round_paths["round_summary_path"], round_summary)
            round_summaries.append(round_summary)
            break

        if cfg.train_selector == "mixed_plus_anchor":
            if anchor_sampler_state is None:
                raise ValueError("anchor sampler state 未初始化。")
            train_dataset_path, trainset_report = build_mixed_plus_anchor_dataset(
                mixed_retained_path=data_result["retained_output_paths"]["mixed_only_shortest"],
                anchor_rows=anchor_rows,
                anchor_sampler_state=anchor_sampler_state,
                anchor_share=cfg.anchor_share,
                output_path=round_paths["train_dataset_path"],
            )
        else:
            train_dataset_path = Path(data_result["retained_output_path"])
            retained_rows = int(data_result["report"]["retained"]["kept"])
            trainset_report = {
                "train_selector": "primary_retained",
                "mixed_count": retained_rows,
                "anchor_target": 0,
                "anchor_count": 0,
                "train_sample_count": retained_rows,
                "anchor_share": 0.0,
                "anchor_sampling": {},
            }

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
            selected_model_path = str(best_checkpoint["checkpoint_path"])
            selected_global_step = int(best_checkpoint["global_step"])
            holdout_metrics = best_checkpoint["metrics"]
            holdout_result_path = str(best_checkpoint["result_path"])
            holdout_raw_result_path = str(best_checkpoint["raw_result_path"])
            checkpoint_ranking_path = str(selection_result["ranking_path"])
            checkpoint_best_path = str(selection_result["best_path"])
        else:
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
            holdout_metrics = holdout_eval["result"]["metrics"]
            selected_model_path = str(train_output_dir)
            selected_global_step = 0
            holdout_result_path = str(holdout_eval["result_path"])
            holdout_raw_result_path = str(holdout_eval["raw_result_path"])
            checkpoint_ranking_path = ""
            checkpoint_best_path = ""
        holdout_accuracy = float(holdout_metrics.get("normalized_accuracy", 0.0))
        improved = _is_improved(
            holdout_accuracy,
            best_holdout_accuracy,
            min_delta=cfg.min_delta,
        )
        if improved:
            best_holdout_accuracy = holdout_accuracy
            best_round_index = round_index
            best_model_path = selected_model_path
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        if cfg.query_strategy == STRATEGY_MIXED_BOOTSTRAP_CANDIDATE:
            if mixed_strategy_state is None:
                raise ValueError("mixed strategy state 未初始化。")
            strategy_update = update_mixed_query_strategy(
                mixed_strategy_state,
                raw_samples_path=data_result["raw_samples_output_path"],
                round_index=round_index,
            )
        elif cfg.query_strategy == STRATEGY_CANDIDATE_RANDOM_MIX:
            if candidate_random_strategy_state is None:
                raise ValueError("candidate_random strategy state 未初始化。")
            persist_candidate_random_query_strategy_state(candidate_random_strategy_state)
            strategy_update = {
                "candidate_pool_size": len(candidate_random_strategy_state["candidate_rows"]),
                "random_pool_size": len(candidate_random_strategy_state["random_rows"]),
            }
        else:
            strategy_update = {}

        retained_report = data_result["report"]["retained"]
        round_summary = {
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
            "round_model_path": selected_model_path,
            "round_output_dir": str(train_output_dir),
            "selected_checkpoint_path": selected_model_path,
            "selected_checkpoint_step": selected_global_step,
            "checkpoint_ranking_path": checkpoint_ranking_path,
            "checkpoint_best_path": checkpoint_best_path,
            "stop_dataset": str(cfg.stop_dataset),
            "holdout_accuracy": holdout_accuracy,
            "improved": improved,
            "best_holdout_accuracy_so_far": best_holdout_accuracy,
            "no_improve_rounds": no_improve_rounds,
            "holdout_result_path": holdout_result_path,
            "holdout_raw_result_path": holdout_raw_result_path,
            "data_report_path": str(data_result["report_path"]),
            "retained_output_paths": data_result.get("retained_output_paths", {}),
        }
        _write_json(round_paths["round_summary_path"], round_summary)
        round_summaries.append(round_summary)
        seen_query_questions.update(current_round_questions)
        if cfg.advance_teacher_on_improvement_only and not improved:
            current_model = best_model_path or current_model
        else:
            current_model = selected_model_path
        if no_improve_rounds >= cfg.patience:
            stop_reason = "no_improvement"
        _write_json(
            history_path,
            _history_payload(
                loop_cfg=cfg,
                rounds=round_summaries,
                best_round_index=best_round_index,
                best_holdout_accuracy=best_holdout_accuracy,
                best_model_path=best_model_path,
                current_model=current_model,
                no_improve_rounds=no_improve_rounds,
                stop_reason=stop_reason,
                query_sampler_state=dump_query_sampler_state(query_sampler_state),
                anchor_sampler_state=dump_query_sampler_state(anchor_sampler_state),
                query_strategy_state_path=(
                    str(mixed_strategy_state["state_path"])
                    if mixed_strategy_state is not None
                    else str(candidate_random_strategy_state["state_path"])
                    if candidate_random_strategy_state is not None
                    else ""
                ),
            ),
        )
        
        if stop_reason == "no_improvement":
            break

    if stop_reason is None:
        if len(round_summaries) >= cfg.max_rounds:
            stop_reason = "max_rounds_reached"
        else:
            stop_reason = "completed"

    final_summary = {
        "stop_reason": stop_reason,
        "completed_rounds": len(round_summaries),
        "best_round_index": best_round_index,
        "best_holdout_accuracy": best_holdout_accuracy,
        "best_model_path": best_model_path or "",
        "history_path": str(history_path),
        "query_strategy": cfg.query_strategy,
        "query_strategy_state_path": (
            str(mixed_strategy_state["state_path"])
            if mixed_strategy_state is not None
            else str(candidate_random_strategy_state["state_path"])
            if candidate_random_strategy_state is not None
            else ""
        ),
    }
    _write_json(
        history_path,
        _history_payload(
            loop_cfg=cfg,
            rounds=round_summaries,
            best_round_index=best_round_index,
            best_holdout_accuracy=best_holdout_accuracy,
            best_model_path=best_model_path,
            current_model=current_model,
            no_improve_rounds=no_improve_rounds,
            stop_reason=stop_reason,
            query_sampler_state=dump_query_sampler_state(query_sampler_state),
            anchor_sampler_state=dump_query_sampler_state(anchor_sampler_state),
            query_strategy_state_path=(
                str(mixed_strategy_state["state_path"])
                if mixed_strategy_state is not None
                else str(candidate_random_strategy_state["state_path"])
                if candidate_random_strategy_state is not None
                else ""
            ),
        ),
    )
    _write_json(final_summary_path, final_summary)
    return {
        "history_path": str(history_path),
        "final_summary_path": str(final_summary_path),
        "summary": final_summary,
    }
