from __future__ import annotations

from collections import deque
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
    return _resolve_compute_dtype(cfg.bnb_4bit_compute_dtype)


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


def _align_trainable_param_dtypes(model, *, target_dtype) -> None:
    import torch

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if not torch.is_floating_point(param):
            continue
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)


def load_model_and_tokenizer(cfg):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    compute_dtype = _aligned_compute_dtype(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }
    model_kwargs.update(build_quantization_kwargs(cfg))
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_name, **model_kwargs)
    if getattr(base_model, "is_loaded_in_4bit", False):
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
        if getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif torch.cuda.is_available():
            model = model.to("cuda")
    _align_trainable_param_dtypes(model, target_dtype=compute_dtype)
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
        self.window = max(1, cfg.wandb_smoothing_window)
        self.buffers: dict[str, deque[float]] = {}
        self.enabled = cfg.report_to == "wandb"
        self._wandb = None
        self.whitelist = {
            "loss",
            "reward",
            "reward_std",
            "frac_reward_zero_std",
            "completions/mean_length",
            "completions/clipped_ratio",
            "entropy",
            "clip_ratio/region_mean",
            "learning_rate",
        }
        self.prefix_whitelist = ("rewards/", "reward/")

    def init(self, cfg, training_args) -> None:
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
                "loss_type": cfg.loss_type,
                "num_generations": cfg.num_generations,
                "use_vllm": cfg.use_vllm,
                "vllm_mode": cfg.vllm_mode,
                "max_prompt_length": cfg.max_prompt_length,
                "max_completion_length": cfg.max_completion_length,
                "learning_rate": cfg.learning_rate,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "batch_size": cfg.batch_size,
            },
        )

    def _should_log(self, key: str) -> bool:
        if key in self.whitelist:
            return True
        return any(key.startswith(prefix) for prefix in self.prefix_whitelist)

    def on_log(self, step: int, logs: dict[str, Any]) -> None:
        if not self.enabled or self._wandb is None or step <= 0:
            return
        smoothed_logs: dict[str, float] = {}
        for key, value in logs.items():
            if not self._should_log(key) or not isinstance(value, (int, float)):
                continue
            buffer = self.buffers.setdefault(key, deque(maxlen=self.window))
            buffer.append(float(value))
            smoothed_logs[f"smoothed/{key}"] = sum(buffer) / len(buffer)
        if smoothed_logs:
            smoothed_logs["step"] = step
            self._wandb.log(smoothed_logs, step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


def train_grpo(cfg, reward_cfg) -> Path:
    from post_train.grpo_data import load_grpo_dataset
    from post_train.grpo_rewards import build_reward_functions

    preflight_check(cfg)
    _ensure_trl_grpo_import_compatibility()

    from trl import GRPOTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_dataset = load_grpo_dataset(cfg.train_dataset)
    reward_funcs = build_reward_functions(reward_cfg)
    training_args = build_training_args(cfg)
    wandb_callback = WandbSmoothingCallback(cfg)
    wandb_callback.init(cfg, training_args)

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
