from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from post_train.config import (
    EvalConfig,
    OnPolicyDataConfig,
    OnPolicyLoopConfig,
    SFTTrainConfig,
    load_eval_config,
    load_on_policy_data_config,
    load_sft_config,
)
from post_train.eval import (
    resolve_model_args,
    result_from_harness_logs,
    result_from_vllm_raw_logs,
    run_harness_eval,
    run_vllm_raw_eval,
    write_eval_result,
    write_raw_eval_result,
)
from post_train.io import ensure_parent
from post_train.on_policy_data import (
    InsufficientRetainedSamplesError,
    load_query_candidates,
    prepare_on_policy_sft_dataset_from_queries,
)
from post_train.sft import train_sft


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


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
        "report_path": data_dir / "train.report.json",
        "holdout_output_path": output_dir / "holdout_eval" / "holdout.vllm_raw.json",
        "holdout_raw_output_path": output_dir / "holdout_eval" / "holdout.vllm_raw.raw.json",
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
) -> SFTTrainConfig:
    paths = build_round_paths(loop_cfg, round_index)
    return base_cfg.model_copy(
        update={
            "model_name": model_name,
            "train_dataset": Path(train_dataset),
            "output_dir": paths["output_dir"],
            "save_strategy": "no",
            "save_steps": None,
            "save_total_limit": None,
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
    runner: str | None = None,
    backend: str | None = None,
    batch_size: int | str | None = None,
    max_batch_size: int | None = None,
    max_new_tokens: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    resolved_runner = runner or eval_cfg.runner
    resolved_backend = backend or eval_cfg.backend
    resolved_batch_size = _effective_eval_setting(batch_size, eval_cfg.batch_size)
    resolved_max_batch_size = max_batch_size if max_batch_size is not None else eval_cfg.max_batch_size
    resolved_max_new_tokens = int(
        _effective_eval_setting(max_new_tokens, eval_cfg.max_new_tokens) or eval_cfg.max_new_tokens
    )
    model_args = resolve_model_args(
        model_name,
        eval_cfg.model_name,
        backend=resolved_backend,
        max_length=eval_cfg.max_seq_length,
        device=eval_cfg.device,
        attn_implementation=eval_cfg.attn_implementation,
        gpu_memory_utilization=eval_cfg.gpu_memory_utilization,
        max_lora_rank=eval_cfg.max_lora_rank,
    )
    task_name = Path(dataset_path).stem.replace("-", "_")
    if resolved_runner == "lm_eval":
        raw_result = run_harness_eval(
            backend=resolved_backend,
            model_args=model_args,
            dataset_path=dataset_path,
            task_name=task_name,
            batch_size=resolved_batch_size,
            max_batch_size=resolved_max_batch_size,
            limit=limit,
            max_gen_toks=resolved_max_new_tokens,
        )
        result = result_from_harness_logs(
            raw_result,
            task_name=task_name,
            dataset_path=dataset_path,
            model_name=model_name,
        )
    else:
        raw_result = run_vllm_raw_eval(
            model_args=model_args,
            dataset_path=dataset_path,
            task_name=task_name,
            batch_size=resolved_batch_size,
            limit=limit,
            max_gen_toks=resolved_max_new_tokens,
        )
        result = result_from_vllm_raw_logs(
            raw_result,
            task_name=task_name,
            dataset_path=dataset_path,
            model_name=model_name,
        )
    output_root = Path(output_dir)
    final_output_path = output_root / f"{task_name}.{resolved_runner}.json"
    raw_output_path = output_root / f"{task_name}.{resolved_runner}.raw.json"
    write_raw_eval_result(raw_output_path, raw_result)
    write_eval_result(final_output_path, result)
    return {
        "result": result,
        "result_path": str(final_output_path),
        "raw_result_path": str(raw_output_path),
    }


def _is_improved(current: float, best_so_far: float | None, *, min_delta: float) -> bool:
    if best_so_far is None:
        return True
    return current > (best_so_far + min_delta)


def _ensure_fresh_loop_dirs(cfg: OnPolicyLoopConfig) -> None:
    for path in (cfg.round_base_dir, cfg.data_base_dir):
        if path.exists():
            raise ValueError(f"自动循环输出目录已存在：{path}")


def _history_payload(
    *,
    loop_cfg: OnPolicyLoopConfig,
    rounds: list[dict[str, Any]],
    best_round_index: int | None,
    best_holdout_accuracy: float | None,
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "seed_model": loop_cfg.seed_model,
        "stop_dataset": str(loop_cfg.stop_dataset),
        "patience": loop_cfg.patience,
        "min_delta": loop_cfg.min_delta,
        "max_rounds": loop_cfg.max_rounds,
        "best_round_index": best_round_index,
        "best_holdout_accuracy": best_holdout_accuracy,
        "stop_reason": stop_reason or "",
        "rounds": rounds,
    }


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


def sample_round_queries(
    query_rows: list[dict[str, Any]],
    sampler_state: dict[str, Any],
    *,
    requested_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not query_rows:
        raise ValueError("query 池为空，无法为当前轮采样。")
    effective_query_count = min(requested_count, len(query_rows))
    if effective_query_count <= 0:
        raise ValueError("requested_count 必须大于 0。")

    start_epoch_index = int(sampler_state["epoch_index"])
    start_epoch_offset = int(sampler_state["epoch_offset"])
    sampled_rows: list[dict[str, Any]] = []
    crossed_epoch = False

    while len(sampled_rows) < effective_query_count:
        epoch_rows = sampler_state["epoch_rows"]
        epoch_offset = int(sampler_state["epoch_offset"])
        remaining = effective_query_count - len(sampled_rows)
        take = min(remaining, len(epoch_rows) - epoch_offset)
        sampled_rows.extend(epoch_rows[epoch_offset : epoch_offset + take])
        sampler_state["epoch_offset"] = epoch_offset + take
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
        "requested_query_count": requested_count,
        "effective_query_count": effective_query_count,
        "crossed_epoch": crossed_epoch,
        "epoch_index_start": start_epoch_index,
        "epoch_offset_start": start_epoch_offset,
        "epoch_index_end": int(sampler_state["epoch_index"]),
        "epoch_offset_end": int(sampler_state["epoch_offset"]),
    }


def run_on_policy_loop(cfg: OnPolicyLoopConfig) -> dict[str, Any]:
    _ensure_fresh_loop_dirs(cfg)
    base_data_cfg = load_on_policy_data_config(cfg.base_on_policy_data_config)
    base_train_cfg = load_sft_config(cfg.base_sft_config)
    eval_cfg = load_eval_config(cfg.eval_config)
    query_rows, query_pool_report = load_query_candidates(base_data_cfg)
    query_sampler_state = build_query_sampler_state(query_rows, seed=base_data_cfg.seed)
    history_path = cfg.round_base_dir / "history.json"
    final_summary_path = cfg.round_base_dir / "final_summary.json"

    current_model = cfg.seed_model
    best_holdout_accuracy: float | None = None
    best_round_index: int | None = None
    best_model_path: str | None = None
    no_improve_rounds = 0
    stop_reason: str | None = None
    round_summaries: list[dict[str, Any]] = []

    for round_index in range(1, cfg.max_rounds + 1):
        round_name = build_round_name(round_index)
        round_paths = build_round_paths(cfg, round_index)
        try:
            data_cfg = build_round_data_config(
                base_data_cfg,
                cfg,
                round_index=round_index,
                generation_model=current_model,
            )
            round_queries, sampler_report = sample_round_queries(
                query_rows,
                query_sampler_state,
                requested_count=base_data_cfg.query_count,
            )
            round_query_report = dict(query_pool_report)
            round_query_report["sampled_queries"] = len(round_queries)
            round_query_report["effective_query_count"] = sampler_report["effective_query_count"]
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

        train_cfg = build_round_train_config(
            base_train_cfg,
            cfg,
            round_index=round_index,
            model_name=current_model,
            train_dataset=data_result["retained_output_path"],
        )
        train_output_dir = train_sft(train_cfg)
        holdout_eval = evaluate_single_dataset(
            model_name=str(train_output_dir),
            dataset_path=cfg.stop_dataset,
            eval_cfg=eval_cfg,
            output_dir=round_paths["output_dir"] / "holdout_eval",
            runner=cfg.runner,
            backend=cfg.backend,
            batch_size=cfg.batch_size,
            max_batch_size=cfg.max_batch_size,
            max_new_tokens=cfg.max_new_tokens,
            limit=cfg.limit,
        )
        holdout_metrics = holdout_eval["result"]["metrics"]
        holdout_accuracy = float(holdout_metrics.get("normalized_accuracy", 0.0))
        improved = _is_improved(
            holdout_accuracy,
            best_holdout_accuracy,
            min_delta=cfg.min_delta,
        )
        if improved:
            best_holdout_accuracy = holdout_accuracy
            best_round_index = round_index
            best_model_path = str(train_output_dir)
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        retained_report = data_result["report"]["retained"]
        round_summary = {
            "round_index": round_index,
            "round_name": round_name,
            "generation_model": current_model,
            "retained_count": int(retained_report["kept"]),
            "retained_ratio": float(retained_report["retained_ratio"]),
            "query_sampling": sampler_report,
            "round_model_path": str(train_output_dir),
            "stop_dataset": str(cfg.stop_dataset),
            "holdout_accuracy": holdout_accuracy,
            "improved": improved,
            "best_holdout_accuracy_so_far": best_holdout_accuracy,
            "no_improve_rounds": no_improve_rounds,
            "holdout_result_path": str(holdout_eval["result_path"]),
            "holdout_raw_result_path": str(holdout_eval["raw_result_path"]),
            "data_report_path": str(data_result["report_path"]),
        }
        _write_json(round_paths["round_summary_path"], round_summary)
        round_summaries.append(round_summary)
        _write_json(
            history_path,
            _history_payload(
                loop_cfg=cfg,
                rounds=round_summaries,
                best_round_index=best_round_index,
                best_holdout_accuracy=best_holdout_accuracy,
                stop_reason=stop_reason,
            ),
        )

        current_model = str(train_output_dir)
        if no_improve_rounds >= cfg.patience:
            stop_reason = "no_improvement"
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
    }
    _write_json(
        history_path,
        _history_payload(
            loop_cfg=cfg,
            rounds=round_summaries,
            best_round_index=best_round_index,
            best_holdout_accuracy=best_holdout_accuracy,
            stop_reason=stop_reason,
        ),
    )
    _write_json(final_summary_path, final_summary)
    return {
        "history_path": str(history_path),
        "final_summary_path": str(final_summary_path),
        "summary": final_summary,
    }
