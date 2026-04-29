#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.grpo.server import run_grpo_with_vllm_server_on_gpus


def parse_gpu_list(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--vllm-gpus 必须是逗号分隔的 GPU index，例如 5,6。") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在双卡上启动 DAPO-lite GRPO 训练。")
    parser.add_argument(
        '--config',
        default='configs/grpo/train_3090_dapo_server.yaml',
        help='GRPO 训练配置文件路径',
    )
    parser.add_argument('--trainer-gpu', type=int, help='强制指定 trainer 使用的物理 GPU index')
    parser.add_argument('--vllm-gpu', type=int, help='强制指定 vLLM server 使用的物理 GPU index')
    parser.add_argument('--vllm-gpus', type=parse_gpu_list, help='强制指定 vLLM server 使用的物理 GPU index 列表，例如 5,6')
    parser.add_argument('--vllm-port', type=int, help='强制指定 vLLM server 监听端口，例如 8001')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        run_grpo_with_vllm_server_on_gpus(
            args.config,
            trainer_gpu=args.trainer_gpu,
            vllm_gpu=args.vllm_gpu,
            vllm_gpus=args.vllm_gpus,
            vllm_port=args.vllm_port,
        )
    )


if __name__ == '__main__':
    main()
