#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.config import load_grpo_config
from rl.grpo import ensure_trl_vllm_import_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 GRPO 使用的 vLLM server。")
    parser.add_argument("--config", default="configs/grpo.yaml", help="GRPO 配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_grpo_config(args.config)
    if not cfg.use_vllm:
        raise ValueError("当前配置未启用 use_vllm=true，不需要启动 vLLM server。")

    compat_applied = ensure_trl_vllm_import_compat(use_vllm=True, vllm_mode=cfg.vllm_mode)
    if compat_applied:
        print("[grpo] 已为当前 trl/vLLM 组合应用 server 模式导入兼容。")

    from trl.scripts.vllm_serve import ScriptArguments, main as vllm_serve_main

    max_model_len = cfg.max_prompt_length + cfg.max_completion_length
    script_args = ScriptArguments(
        model=str(cfg.base_model_name),
        host=cfg.vllm_server_host,
        port=cfg.vllm_server_port,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        data_parallel_size=cfg.vllm_data_parallel_size,
        trust_remote_code=True,
    )
    print(
        "[grpo] 启动 vLLM server："
        f" model={cfg.base_model_name}"
        f" host={cfg.vllm_server_host}"
        f" port={cfg.vllm_server_port}"
        f" tp={cfg.vllm_tensor_parallel_size}"
        f" dp={cfg.vllm_data_parallel_size}"
        f" max_model_len={max_model_len}"
    )
    vllm_serve_main(script_args)


if __name__ == "__main__":
    main()
