#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_grpo_reward_config, load_grpo_train_config
from post_train.experiments import register_training_run
from post_train.grpo import train_grpo
from post_train.grpo_runtime import GRPO_VLLM_SERVER_BASE_URL_ENV, apply_grpo_runtime_env_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 TRL GRPO 训练。")
    parser.add_argument("--config", required=True, help="GRPO 训练配置文件路径")
    parser.add_argument("--resume-from-checkpoint", help="从已有 checkpoint 继续训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_grpo_train_config(args.config)
    cfg = apply_grpo_runtime_env_overrides(cfg, env=os.environ)
    if args.resume_from_checkpoint:
        cfg = cfg.model_copy(update={"resume_from_checkpoint": args.resume_from_checkpoint})
    if os.environ.get(GRPO_VLLM_SERVER_BASE_URL_ENV):
        print(f"grpo.runtime_vllm_server_base_url={cfg.vllm_server_base_url}")
    reward_cfg = load_grpo_reward_config(cfg.reward_config)
    output_dir = train_grpo(cfg, reward_cfg)
    summary = register_training_run(
        route="grpo",
        config_path=args.config,
        output_dir=output_dir,
        train_dataset=cfg.train_dataset,
        base_model=cfg.model_name_or_path,
    )
    print(f"saved_model={output_dir}")
    print(f"wrote_run_summary={summary['summary_path']}")


if __name__ == "__main__":
    main()
