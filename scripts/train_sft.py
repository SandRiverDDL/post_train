#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_sft_config
from post_train.sft import preview_training_samples, train_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 stage1 / stage2 SFT。")
    parser.add_argument("--config", required=True, help="SFT 配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_sft_config(args.config)
    preview = preview_training_samples(cfg.train_dataset)
    if preview:
        print(preview)
    output_dir = train_sft(cfg)
    print(f"saved_model={output_dir}")


if __name__ == "__main__":
    main()
