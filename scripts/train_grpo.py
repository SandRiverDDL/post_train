#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import load_grpo_config
from rl.grpo import JsonlMetricsCallback, build_grpo_dataset, default_reward_weights, reward_functions
from rl.io import ensure_parent, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Qwen3 GRPO。")
    parser.add_argument("--config", default="configs/grpo.yaml", help="配置文件路径")
    return parser.parse_args()


def _looks_like_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def load_model_and_tokenizer(cfg):
    if cfg.use_unsloth:
        import unsloth  # noqa: F401
        from peft import PeftModel
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=True,
        )
        if _looks_like_adapter_dir(cfg.cold_start_model):
            model = PeftModel.from_pretrained(model, str(cfg.cold_start_model), is_trainable=True)
        return model, tokenizer

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        trust_remote_code=True,
    )
    if _looks_like_adapter_dir(cfg.cold_start_model):
        model = PeftModel.from_pretrained(model, str(cfg.cold_start_model), is_trainable=True)
    return model, tokenizer


def preview_samples(dataset_path: Path) -> None:
    rows = read_jsonl(dataset_path)[:3]
    for row in rows:
        print("=" * 80)
        print(f"id: {row['id']}")
        print(str(row["question"])[:300])
        print("--- prompt ---")
        print(row["prompt"])
        print("--- final_answer ---")
        print(row["final_answer"])


def main() -> None:
    args = parse_args()
    cfg = load_grpo_config(args.config)
    preview_samples(cfg.train_dataset)

    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_dataset = build_grpo_dataset(cfg.train_dataset)
    reward_funcs = reward_functions()

    training_args = GRPOConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        num_iterations=cfg.num_iterations,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        mask_truncated_completions=cfg.mask_truncated_completions,
        top_entropy_quantile=cfg.top_entropy_quantile,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        report_to="none",
        seed=cfg.seed,
        log_completions=cfg.log_completions,
        num_completions_to_print=cfg.num_completions_to_print,
        reward_weights=default_reward_weights(),
        use_vllm=cfg.use_vllm,
    )

    callback = JsonlMetricsCallback(cfg.output_dir / "train_log.jsonl")
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
                "reward_functions": ["correctness_reward", "parse_reward", "format_reward"],
                "reward_weights": default_reward_weights(),
                "loss_type": cfg.loss_type,
                "mask_truncated_completions": cfg.mask_truncated_completions,
                "top_entropy_quantile": cfg.top_entropy_quantile,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
