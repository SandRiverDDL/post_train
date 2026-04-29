from __future__ import annotations

from typing import Any

from post_train.config import load_eval_config, load_on_policy_data_config, load_sft_config
from post_train.on_policy.data import InsufficientRetainedSamplesError, load_query_candidates, prepare_on_policy_sft_dataset_from_queries
from post_train.on_policy.trainset import load_anchor_rows

from .loop_paths import build_round_data_config, build_round_name, build_round_paths
from .loop_round import (
    build_round_summary,
    build_round_trainset,
    is_improved,
    sample_queries_for_round,
    train_and_select_round_model,
    update_round_strategy,
)
from .loop_state import final_summary_path, history_path, history_payload, load_seen_query_questions, prepare_loop_run, write_json
from .strategy_candidate_random import build_candidate_random_query_strategy, load_candidate_random_query_strategy_state
from .strategy_common import (
    STRATEGY_CANDIDATE_RANDOM_MIX,
    STRATEGY_MIXED_BOOTSTRAP_CANDIDATE,
    build_query_sampler_state,
    dump_query_sampler_state,
    load_query_sampler_state,
    normalize_question,
)
from .strategy_mixed import build_mixed_query_strategy, load_mixed_query_strategy_state


def run_on_policy_loop(
    cfg,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_state = prepare_loop_run(cfg, resume=resume, overwrite=overwrite)
    base_data_cfg = load_on_policy_data_config(cfg.base_on_policy_data_config)
    base_train_cfg = load_sft_config(cfg.base_sft_config)
    eval_cfg = load_eval_config(cfg.eval_config)
    saved_history_path = run_state["history_path"]
    saved_final_summary_path = run_state["final_summary_path"]

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
    seen_query_questions = load_seen_query_questions(cfg, completed_rounds=start_round_index - 1)

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
            round_queries, sampler_report, round_query_report = sample_queries_for_round(
                cfg=cfg,
                query_rows=query_rows,
                query_pool_report=query_pool_report,
                query_sampler_state=query_sampler_state,
                mixed_strategy_state=mixed_strategy_state,
                candidate_random_strategy_state=candidate_random_strategy_state,
                seen_query_questions=seen_query_questions,
            )
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
            write_json(round_paths["round_summary_path"], round_summary)
            round_summaries.append(round_summary)
            break

        train_dataset_path, trainset_report = build_round_trainset(
            cfg=cfg,
            round_paths=round_paths,
            data_result=data_result,
            anchor_rows=anchor_rows,
            anchor_sampler_state=anchor_sampler_state,
        )
        train_output_dir, selection_info = train_and_select_round_model(
            cfg=cfg,
            base_train_cfg=base_train_cfg,
            eval_cfg=eval_cfg,
            round_index=round_index,
            round_paths=round_paths,
            current_model=current_model,
            train_dataset_path=train_dataset_path,
            trainset_report=trainset_report,
        )
        holdout_accuracy = float(selection_info["holdout_metrics"].get("normalized_accuracy", 0.0))
        improved = is_improved(
            holdout_accuracy,
            best_holdout_accuracy,
            min_delta=cfg.min_delta,
        )
        if improved:
            best_holdout_accuracy = holdout_accuracy
            best_round_index = round_index
            best_model_path = selection_info["selected_model_path"]
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        strategy_update = update_round_strategy(
            cfg=cfg,
            mixed_strategy_state=mixed_strategy_state,
            candidate_random_strategy_state=candidate_random_strategy_state,
            data_result=data_result,
            round_index=round_index,
        )

        round_summary = build_round_summary(
            cfg=cfg,
            round_index=round_index,
            round_name=round_name,
            current_model=current_model,
            data_result=data_result,
            train_dataset_path=train_dataset_path,
            trainset_report=trainset_report,
            sampler_report=sampler_report,
            strategy_update=strategy_update,
            train_output_dir=train_output_dir,
            selection_info=selection_info,
            holdout_accuracy=holdout_accuracy,
            improved=improved,
            best_holdout_accuracy=best_holdout_accuracy,
            no_improve_rounds=no_improve_rounds,
        )
        write_json(round_paths["round_summary_path"], round_summary)
        round_summaries.append(round_summary)
        current_round_questions = {
            question
            for question in (
                normalize_question(str(row.get("question", "")))
                for row in round_queries
            )
            if question
        }
        seen_query_questions.update(current_round_questions)
        if cfg.advance_teacher_on_improvement_only and not improved:
            current_model = best_model_path or current_model
        else:
            current_model = selection_info["selected_model_path"]
        if no_improve_rounds >= cfg.patience:
            stop_reason = "no_improvement"
        write_json(
            saved_history_path,
            history_payload(
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
        "stop_dataset": str(cfg.stop_dataset),
        "history_path": str(saved_history_path),
        "query_strategy": cfg.query_strategy,
        "query_strategy_state_path": (
            str(mixed_strategy_state["state_path"])
            if mixed_strategy_state is not None
            else str(candidate_random_strategy_state["state_path"])
            if candidate_random_strategy_state is not None
            else ""
        ),
    }
    write_json(
        saved_history_path,
        history_payload(
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
    write_json(saved_final_summary_path, final_summary)
    return {
        "history_path": str(saved_history_path),
        "final_summary_path": str(saved_final_summary_path),
        "summary": final_summary,
    }
