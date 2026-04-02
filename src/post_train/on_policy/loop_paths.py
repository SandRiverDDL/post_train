from __future__ import annotations

import math
from pathlib import Path

from post_train.config import OnPolicyDataConfig, OnPolicyLoopConfig, SFTTrainConfig


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
