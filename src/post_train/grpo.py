from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

from post_train.io import ensure_parent


def _looks_like_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "adapter_config.json").exists()


def _resolve_compute_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _resolve_train_compute_dtype(cfg):
    return _resolve_compute_dtype(cfg.compute_dtype)


def build_quantization_kwargs(cfg) -> dict[str, Any]:
    if not cfg.load_in_4bit:
        return {}

    from transformers import BitsAndBytesConfig

    compute_dtype = _resolve_compute_dtype(cfg.bnb_4bit_compute_dtype)
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        "device_map": "auto",
    }


def _aligned_compute_dtype(cfg):
    return _resolve_train_compute_dtype(cfg)


def _ensure_trl_grpo_import_compatibility() -> None:
    try:
        from vllm import sampling_params as sampling_params_module
    except Exception:
        return
    if hasattr(sampling_params_module, "GuidedDecodingParams"):
        return

    class GuidedDecodingParams:
        def __init__(self, regex: str | None = None, **kwargs) -> None:
            self.regex = regex
            for key, value in kwargs.items():
                setattr(self, key, value)

    sampling_params_module.GuidedDecodingParams = GuidedDecodingParams


def preflight_check(cfg) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GRPO 训练需要可用 GPU；当前环境未检测到 CUDA 设备。")
    if cfg.use_vllm:
        vllm_version = metadata.version("vllm")
        trl_version = metadata.version("trl")
        if trl_version == "0.24.0" and vllm_version != "0.10.2":
            raise RuntimeError(
                f"当前 use_vllm=true，但检测到 trl=={trl_version} 且 vllm=={vllm_version}；TRL 0.24.0 当前只支持 vllm==0.10.2。"
            )


def build_training_args(cfg):
    import torch
    from trl import GRPOConfig

    compute_dtype = _aligned_compute_dtype(cfg)
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


def _candidate_model_nodes(model) -> list[tuple[str, Any]]:
    seen_ids: set[int] = set()
    queue: list[tuple[str, Any]] = [("model", model)]
    candidates: list[tuple[str, Any]] = []

    while queue:
        path, node = queue.pop(0)
        if node is None:
            continue
        node_id = id(node)
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        candidates.append((path, node))

        if hasattr(node, "base_model"):
            queue.append((f"{path}.base_model", getattr(node, "base_model")))
        if hasattr(node, "model"):
            queue.append((f"{path}.model", getattr(node, "model")))
        if hasattr(node, "get_base_model"):
            try:
                queue.append((f"{path}.get_base_model()", node.get_base_model()))
            except Exception:
                pass

    return candidates


def _iter_output_heads(model) -> list[tuple[str, Any]]:
    seen_ids: set[int] = set()
    heads: list[tuple[str, Any]] = []

    for path, node in _candidate_model_nodes(model):
        lm_head = getattr(node, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight") and id(lm_head) not in seen_ids:
            seen_ids.add(id(lm_head))
            heads.append((f"{path}.lm_head", lm_head))

        if hasattr(node, "get_output_embeddings"):
            try:
                output_embeddings = node.get_output_embeddings()
            except Exception:
                output_embeddings = None
            if output_embeddings is not None and hasattr(output_embeddings, "weight") and id(output_embeddings) not in seen_ids:
                seen_ids.add(id(output_embeddings))
                heads.append((f"{path}.get_output_embeddings()", output_embeddings))

    return heads


def align_output_head_dtypes(model, *, target_dtype) -> list[str]:
    touched_paths: list[str] = []
    for path, head in _iter_output_heads(model):
        if head.weight.dtype != target_dtype:
            head.weight.data = head.weight.data.to(target_dtype)
        touched_paths.append(path)
    return touched_paths


def collect_model_dtype_report(model) -> dict[str, str]:
    report: dict[str, str] = {}
    for path, head in _iter_output_heads(model):
        report[f"{path}.weight_dtype"] = str(head.weight.dtype)
    if hasattr(model, "get_input_embeddings"):
        try:
            input_embeddings = model.get_input_embeddings()
        except Exception:
            input_embeddings = None
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            report["model.get_input_embeddings().weight_dtype"] = str(input_embeddings.weight.dtype)
    return report


def assert_generation_dtype_ready(model, *, target_dtype) -> dict[str, str]:
    report = collect_model_dtype_report(model)
    expected_dtype = str(target_dtype)
    output_head_items = {key: value for key, value in report.items() if key.endswith(".weight_dtype") and "input_embeddings" not in key}
    if not output_head_items:
        raise RuntimeError("GRPO 生成前 dtype 检查失败：未找到任何可用于生成的 output head。")
    mismatched = {key: value for key, value in output_head_items.items() if value != expected_dtype}
    if mismatched:
        mismatch_text = ", ".join(f"{key}={value}" for key, value in sorted(mismatched.items()))
        raise RuntimeError(f"GRPO 生成前 dtype 检查失败：{mismatch_text}，expected={expected_dtype}。")
    return report


def load_model_and_tokenizer(cfg):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    compute_dtype = _resolve_train_compute_dtype(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }
    if cfg.load_in_4bit:
        model_kwargs.update(build_quantization_kwargs(cfg))
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_name, **model_kwargs)
    if cfg.load_in_4bit and getattr(base_model, "is_loaded_in_4bit", False):
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    elif torch.cuda.is_available():
        base_model = base_model.to("cuda")

    if _looks_like_adapter_dir(cfg.model_name_or_path):
        model = PeftModel.from_pretrained(
            base_model,
            cfg.model_name_or_path,
            is_trainable=True,
            autocast_adapter_dtype=False,
        )
    elif cfg.model_name_or_path == cfg.base_model_name:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
        if cfg.load_in_4bit and getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif torch.cuda.is_available():
            model = model.to("cuda")
    align_output_head_dtypes(model, target_dtype=compute_dtype)
    return model, tokenizer


class WandbSmoothingCallback:
    def __init__(self, cfg) -> None:
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            def __init__(self, outer) -> None:
                self.outer = outer

            def on_log(self, args, state, control, logs=None, **kwargs):
                self.outer.on_log(state.global_step, logs or {})
                return control

            def on_train_end(self, args, state, control, **kwargs):
                self.outer.finish()
                return control

        self.callback = _Impl(self)
        self.cfg = cfg
        self.enabled = cfg.report_to == "wandb"
        self._wandb = None
        self.whitelist = {
            "reward",
            "reward_std",
            "completions/mean_length",
            "clip_ratio/region_mean",
            "learning_rate",
        }
        self.debug_whitelist = {
            "loss",
            "frac_reward_zero_std",
            "completions/clipped_ratio",
            "entropy",
        }

    def init(self, cfg, training_args, *, train_dataset_size: int) -> None:
        if not self.enabled:
            return
        import wandb

        self._wandb = wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or Path(cfg.output_dir).name,
            config={
                "output_dir": str(cfg.output_dir),
                "train_dataset": str(cfg.train_dataset),
                "train_dataset_size": train_dataset_size,
                "max_train_samples": cfg.max_train_samples,
                "train_subset_seed": cfg.train_subset_seed,
                "train_subset_mode": cfg.train_subset_mode,
                "loss_type": cfg.loss_type,
                "num_generations": cfg.num_generations,
                "use_vllm": cfg.use_vllm,
                "vllm_mode": cfg.vllm_mode,
                "max_prompt_length": cfg.max_prompt_length,
                "max_completion_length": cfg.max_completion_length,
                "learning_rate": cfg.learning_rate,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "batch_size": cfg.batch_size,
                "mask_truncated_completions": cfg.mask_truncated_completions,
                "wandb_debug_metrics": cfg.wandb_debug_metrics,
            },
        )

    def _should_log(self, key: str) -> bool:
        if key in self.whitelist:
            return True
        if key.startswith("reward/") or key.startswith("rewards/"):
            return True
        return self.cfg.wandb_debug_metrics and key in self.debug_whitelist

    def on_log(self, step: int, logs: dict[str, Any]) -> None:
        if not self.enabled or self._wandb is None or step <= 0:
            return
        filtered_logs: dict[str, float] = {}
        for key, value in logs.items():
            if not self._should_log(key) or not isinstance(value, (int, float)):
                continue
            filtered_logs[key] = float(value)
        if filtered_logs:
            filtered_logs["step"] = step
            self._wandb.log(filtered_logs, step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


def train_grpo(cfg, reward_cfg) -> Path:
    from post_train.grpo_data import load_grpo_dataset, select_grpo_train_subset
    from post_train.grpo_rewards import build_reward_functions

    preflight_check(cfg)
    _ensure_trl_grpo_import_compatibility()

    from trl import GRPOTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_dataset = load_grpo_dataset(cfg.train_dataset)
    train_dataset = select_grpo_train_subset(
        train_dataset,
        max_train_samples=cfg.max_train_samples,
        seed=cfg.train_subset_seed,
        mode=cfg.train_subset_mode,
    )
    reward_funcs = build_reward_functions(reward_cfg)
    training_args = build_training_args(cfg)
    dtype_report = assert_generation_dtype_ready(model, target_dtype=_aligned_compute_dtype(cfg))
    for key, value in sorted(dtype_report.items()):
        print(f"grpo.{key}={value}")
    print(f"grpo.train_dataset_size={len(train_dataset)}")
    wandb_callback = WandbSmoothingCallback(cfg)
    wandb_callback.init(cfg, training_args, train_dataset_size=len(train_dataset))

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[wandb_callback.callback],
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    output_dir = ensure_parent(cfg.output_dir / "placeholder.txt").parent
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
