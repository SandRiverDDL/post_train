#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import load_grpo_config
from rl.grpo import (
    JsonlMetricsCallback,
    build_grpo_dataset,
    build_grpo_prompt,
    align_lm_head_dtype,
    default_reward_weights,
    ensure_trl_model_compat,
    ensure_trl_vllm_import_compat,
    reward_functions,
    stabilize_unquantized_kbit_linears,
)
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
            model = PeftModel.from_pretrained(
                model,
                str(cfg.cold_start_model),
                is_trainable=True,
                autocast_adapter_dtype=False,
            )
        return ensure_trl_model_compat(model), tokenizer

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model, upcast_linear_names = stabilize_unquantized_kbit_linears(model)
    if upcast_linear_names:
        preview = ", ".join(upcast_linear_names[:6])
        print(
            "[grpo] 检测到未量化的内部 Linear 模块，已将其提升到 float32 以规避 dtype 冲突："
            f"{preview}"
        )
    if align_lm_head_dtype(model, compute_dtype):
        print(f"[grpo] 已将 lm_head 对齐到 {compute_dtype}，避免 generation 阶段输出头 dtype 冲突。")
    if _looks_like_adapter_dir(cfg.cold_start_model):
        model = PeftModel.from_pretrained(
            model,
            str(cfg.cold_start_model),
            is_trainable=True,
        )
    return ensure_trl_model_compat(model), tokenizer


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


def resolve_precision_flags(cfg) -> tuple[bool, bool, bool]:
    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(cfg.bf16) if cfg.bf16 is not None else False
    use_fp16 = bool(cfg.fp16) if cfg.fp16 is not None else False
    if not has_cuda:
        use_bf16 = False
        use_fp16 = False

    return use_bf16, use_fp16, not has_cuda


def main() -> None:
    args = parse_args()
    cfg = load_grpo_config(args.config)
    preview_samples(cfg.train_dataset)
    ensure_trl_vllm_import_compat(use_vllm=cfg.use_vllm)

    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_dataset = build_grpo_dataset(cfg.train_dataset)
    reward_funcs = reward_functions()
    use_bf16, use_fp16, use_cpu = resolve_precision_flags(cfg)

    training_args = GRPOConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=use_cpu,
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
