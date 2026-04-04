#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.grpo_server import run_grpo_with_vllm_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在双卡上启动 DAPO-lite GRPO 训练。")
    parser.add_argument(
        '--config',
        default='configs/grpo/train_3090_dapo_server.yaml',
        help='GRPO 训练配置文件路径',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_grpo_with_vllm_server(args.config))


if __name__ == '__main__':
    main()
