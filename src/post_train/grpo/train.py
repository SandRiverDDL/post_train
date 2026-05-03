from __future__ import annotations

from importlib import metadata
from pathlib import Path

from post_train.io import ensure_parent
from post_train.tracking import create_trainer_callback

from .callbacks import WandbSmoothingCallback
from .diagnostics import collect_adapter_diagnostics
from .model_loading import aligned_compute_dtype, assert_generation_dtype_ready, load_model_and_tokenizer
from .training_args import build_training_args


def ensure_trl_grpo_import_compatibility() -> None:
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


def train_grpo(cfg, reward_cfg) -> Path:
    from post_train.grpo.data import load_grpo_dataset, select_grpo_train_subset
    from post_train.grpo.rewards import QuestionStatsCallback, QuestionStatsRecorder, build_reward_functions

    preflight_check(cfg)
    ensure_trl_grpo_import_compatibility()

    from trl import GRPOTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    adapter_diagnostics = collect_adapter_diagnostics(model, cfg)
    for key, value in sorted(adapter_diagnostics.items()):
        print(f"grpo.adapter.{key}={value}")
    train_dataset = load_grpo_dataset(cfg.train_dataset)
    train_dataset = select_grpo_train_subset(
        train_dataset,
        max_train_samples=cfg.max_train_samples,
        seed=cfg.train_subset_seed,
        mode=cfg.train_subset_mode,
    )
    question_stats_recorder = QuestionStatsRecorder(cfg.question_stats_path) if cfg.question_stats_path else None
    reward_funcs = build_reward_functions(
        reward_cfg,
        max_completion_length=cfg.max_completion_length,
        stats_recorder=question_stats_recorder,
    )
    training_args = build_training_args(cfg)
    dtype_report = assert_generation_dtype_ready(model, target_dtype=aligned_compute_dtype(cfg))
    for key, value in sorted(dtype_report.items()):
        print(f"grpo.{key}={value}")
    print(f"grpo.train_dataset_size={len(train_dataset)}")
    wandb_callback = WandbSmoothingCallback(cfg)
    wandb_callback.init(cfg, training_args, train_dataset_size=len(train_dataset))
    callbacks = [wandb_callback.callback]
    mlflow_callback = create_trainer_callback()
    if mlflow_callback is not None:
        callbacks.append(mlflow_callback)
    if question_stats_recorder is not None:
        print(f"grpo.question_stats_path={cfg.question_stats_path}")
        callbacks.append(QuestionStatsCallback(question_stats_recorder).callback)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    output_dir = ensure_parent(cfg.output_dir / "placeholder.txt").parent
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
