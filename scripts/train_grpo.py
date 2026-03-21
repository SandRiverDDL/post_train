#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import load_grpo_config
from rl.grpo import (
    JsonlMetricsCallback,
    build_grpo_dataset,
    build_grpo_prompt,
    default_reward_weights,
    reward_functions,
    set_reward_config,
    set_reward_tokenizer,
)
from rl.io import ensure_parent, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Unsloth GRPO。")
    parser.add_argument("--config", default="configs/grpo.yaml", help="配置文件路径")
    return parser.parse_args()


def _looks_like_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def resolve_training_model_name(cfg) -> str:
    if _looks_like_adapter_dir(cfg.cold_start_model):
        return str(cfg.cold_start_model)
    return str(cfg.base_model_name)


def load_model_and_tokenizer(cfg):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

    PatchFastRL("GRPO", FastLanguageModel)
    from trl import GRPOConfig, GRPOTrainer

    model_name = resolve_training_model_name(cfg)
    load_kwargs = {
        "model_name": model_name,
        "max_seq_length": cfg.max_seq_length,
        "load_in_4bit": cfg.load_in_4bit,
        "fast_inference": cfg.fast_inference,
        "max_lora_rank": cfg.lora_rank,
    }
    if cfg.fast_inference:
        load_kwargs["gpu_memory_utilization"] = cfg.gpu_memory_utilization

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if not _looks_like_adapter_dir(cfg.cold_start_model):
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )

    if hasattr(model, "for_training"):
        model.for_training(use_gradient_checkpointing=True)

    return model, tokenizer, GRPOConfig, GRPOTrainer, is_bfloat16_supported


def preview_samples(dataset_path: Path) -> None:
    rows = read_jsonl(dataset_path)[:3]
    for row in rows:
        prompt = build_grpo_prompt(str(row["question"]))
        print("=" * 80)
        print(f"id: {row['id']}")
        print(str(row["question"])[:300])
        print("--- prompt ---")
        print(prompt)
        print("--- final_answer ---")
        print(row["final_answer"])


def init_wandb(cfg) -> Any | None:
    if cfg.report_to == "none":
        return None
    if cfg.report_to != "wandb":
        raise ValueError(f"不支持的 report_to: {cfg.report_to}")
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError(
            "当前配置启用了 wandb，但环境里没有安装 wandb。"
            "请先安装 wandb，或把 report_to 改为 none。"
        )
    os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)
    import wandb

    run_name = cfg.wandb_run_name or cfg.output_dir.name
    return wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        tags=cfg.wandb_tags,
        dir=str(cfg.output_dir),
        config=cfg.model_dump(mode="json"),
    )


def main() -> None:
    args = parse_args()
    cfg = load_grpo_config(args.config)
    preview_samples(cfg.train_dataset)
    wandb_run = init_wandb(cfg)

    model, tokenizer, GRPOConfig, GRPOTrainer, is_bfloat16_supported = load_model_and_tokenizer(cfg)
    train_dataset = build_grpo_dataset(cfg.train_dataset)
    set_reward_config(
        correct=cfg.reward_correct,
        wrong=cfg.reward_wrong,
        parse_fail=cfg.reward_parse_fail,
        strict_boxed_bonus=cfg.reward_strict_boxed_bonus,
        length_coef=cfg.reward_length_coef,
        use_relaxed_correctness=cfg.reward_use_relaxed_correctness,
    )
    set_reward_tokenizer(tokenizer)
    reward_funcs = reward_functions()

    training_args = GRPOConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        num_iterations=cfg.num_iterations,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_p=cfg.min_p,
        repetition_penalty=cfg.repetition_penalty,
        generation_kwargs=cfg.generation_kwargs,
        mask_truncated_completions=cfg.mask_truncated_completions,
        top_entropy_quantile=cfg.top_entropy_quantile,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to="none",
        seed=cfg.seed,
        log_completions=cfg.log_completions,
        num_completions_to_print=cfg.num_completions_to_print,
        reward_weights=default_reward_weights(),
    )

    callback = JsonlMetricsCallback(cfg.output_dir / "train_log.jsonl", wandb_run=wandb_run)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    config_snapshot = ensure_parent(cfg.output_dir / "config_used.yaml")
    with config_snapshot.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, allow_unicode=True, sort_keys=False)

    metrics_snapshot = ensure_parent(cfg.output_dir / "reward_contract.json")
    with metrics_snapshot.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "reward_functions": ["combined_reward"],
                "reward_weights": default_reward_weights(),
                "reward_formula": {
                    "correct": cfg.reward_correct,
                    "wrong": cfg.reward_wrong,
                    "parse_fail": cfg.reward_parse_fail,
                    "strict_boxed_bonus": cfg.reward_strict_boxed_bonus,
                    "length_coef": cfg.reward_length_coef,
                    "use_relaxed_correctness": cfg.reward_use_relaxed_correctness,
                },
                "loss_type": cfg.loss_type,
                "mask_truncated_completions": cfg.mask_truncated_completions,
                "top_entropy_quantile": cfg.top_entropy_quantile,
                "report_to": cfg.report_to,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
