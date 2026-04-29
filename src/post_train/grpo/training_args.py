from __future__ import annotations

from .model_loading import aligned_compute_dtype


def build_training_args(cfg):
    import torch
    from trl import GRPOConfig

    compute_dtype = aligned_compute_dtype(cfg)
    use_bf16 = bool(torch.cuda.is_available() and compute_dtype == torch.bfloat16)
    use_fp16 = bool(torch.cuda.is_available() and compute_dtype == torch.float16)
    return GRPOConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        report_to="none",
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        loss_type=cfg.loss_type,
        epsilon=cfg.epsilon,
        epsilon_high=cfg.epsilon_high,
        beta=cfg.beta,
        scale_rewards="none" if not cfg.scale_rewards else "group",
        importance_sampling_level=cfg.importance_sampling_level,
        mask_truncated_completions=cfg.mask_truncated_completions,
        use_vllm=cfg.use_vllm,
        vllm_mode=cfg.vllm_mode,
        vllm_server_base_url=cfg.vllm_server_base_url,
        vllm_server_host=cfg.vllm_server_host,
        vllm_server_port=cfg.vllm_server_port,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,
    )
