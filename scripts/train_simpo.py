#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_simpo_config
from post_train.simpo import train_simpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 TRL SimPO 训练。")
    parser.add_argument("--config", required=True, help="SimPO 配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simpo_config(args.config)
    output_dir = train_simpo(cfg)
    print(f"saved_model={output_dir}")


if __name__ == "__main__":
    main()
