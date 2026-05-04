#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_sft_config
from post_train.experiments import infer_route_from_config_path, register_training_run
from post_train.sft import preview_training_samples, train_sft
from post_train.tracking import log_training_summary, start_run


def is_main_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 stage1 / stage2 SFT。")
    parser.add_argument("--config", required=True, help="SFT 配置文件路径")
    parser.add_argument("--set", action="append", default=[], help="覆盖配置字段，例如 --set distill_top_k=8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_sft_config(args.config)
    if args.set:
        from omegaconf import OmegaConf

        overrides = OmegaConf.to_container(OmegaConf.from_dotlist(args.set), resolve=True)
        cfg = type(cfg).model_validate({**cfg.model_dump(), **overrides})
    main_process = is_main_process()
    preview = preview_training_samples(cfg.train_dataset) if main_process else ""
    if main_process and preview:
        print(preview)
    route = infer_route_from_config_path(args.config)
    with start_run(
        run_name=cfg.output_dir.name,
        route=route,
        config_path=args.config,
        output_dir=cfg.output_dir,
        params=cfg,
    ):
        output_dir = train_sft(cfg)
        if not main_process:
            return
        summary = register_training_run(
            route=route,
            config_path=args.config,
            output_dir=output_dir,
            train_dataset=cfg.train_dataset,
            base_model=cfg.model_name,
        )
        log_training_summary(summary)
    print(f"saved_model={output_dir}")
    print(f"wrote_run_summary={summary['summary_path']}")


if __name__ == "__main__":
    main()
