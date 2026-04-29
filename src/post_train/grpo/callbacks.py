from __future__ import annotations

from pathlib import Path
from typing import Any


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
            "completions/clipped_ratio",
            "entropy",
            "clip_ratio/region_mean",
            "learning_rate",
        }
        self.debug_whitelist = {
            "loss",
            "frac_reward_zero_std",
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
