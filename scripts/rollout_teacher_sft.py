#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.rollout.teacher_sft import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_SOURCE,
    build_teacher_sft_prompts,
    merge_teacher_sft_shards,
    run_teacher_sft_shard,
)


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值：{value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="教师模型 rollout SFT raw 数据，不生成 OPD topK。")
    parser.add_argument("--mode", choices=("prompts", "shard", "merge", "launch"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--prompt-source", default=str(DEFAULT_PROMPT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--responses-per-prompt", type=int, default=1)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat-template-enable-thinking", type=parse_bool, default=False)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--assistant-prefill", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpus", default=None, help="launch 模式使用，逗号分隔，例如 0,2,7。")
    parser.add_argument("--session-prefix", default=None, help="launch 模式 tmux session 前缀。")
    return parser.parse_args()


def _shard_kwargs(args: argparse.Namespace) -> dict:
    return {
        "output_dir": args.output_dir,
        "num_shards": args.num_shards,
        "model": args.model,
        "tokenizer_name": args.tokenizer_name,
        "use_chat_template": args.use_chat_template,
        "chat_template_enable_thinking": args.chat_template_enable_thinking,
        "system_prompt": args.system_prompt,
        "assistant_prefill": args.assistant_prefill,
        "responses_per_prompt": args.responses_per_prompt,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }


def _launch_shards(args: argparse.Namespace) -> dict:
    if not args.gpus:
        raise ValueError("launch 模式必须传 --gpus。")
    prompts_path = Path(args.output_dir) / "prompts.jsonl"
    if not prompts_path.exists():
        build_teacher_sft_prompts(
            prompt_source=args.prompt_source,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) != args.num_shards:
        raise ValueError(f"--gpus 数量必须等于 --num-shards：gpus={gpus}, num_shards={args.num_shards}")
    base = Path(args.output_dir)
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_prefix = args.session_prefix or base.name.replace("/", "_")
    jobs = []
    for shard_index, gpu in enumerate(gpus):
        session = f"{session_prefix}_s{shard_index}"
        log_path = log_dir / f"shard{shard_index}.log"
        cmd_parts = [
            "CUDA_VISIBLE_DEVICES=" + shlex.quote(gpu),
            shlex.quote(str(ROOT / ".venv/bin/python")),
            shlex.quote(str(ROOT / "scripts/rollout_teacher_sft.py")),
            "--mode shard",
            "--model",
            shlex.quote(args.model),
            "--output-dir",
            shlex.quote(args.output_dir),
            "--num-shards",
            str(args.num_shards),
            "--shard-index",
            str(shard_index),
            "--sample-size",
            str(args.sample_size),
            "--seed",
            str(args.seed),
            "--responses-per-prompt",
            str(args.responses_per_prompt),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--chat-template-enable-thinking",
            str(args.chat_template_enable_thinking).lower(),
        ]
        if args.tokenizer_name:
            cmd_parts += ["--tokenizer-name", shlex.quote(args.tokenizer_name)]
        if not args.use_chat_template:
            cmd_parts += ["--no-use-chat-template"]
        if args.system_prompt:
            cmd_parts += ["--system-prompt", shlex.quote(args.system_prompt)]
        if args.assistant_prefill:
            cmd_parts += ["--assistant-prefill", shlex.quote(args.assistant_prefill)]
        if args.enforce_eager:
            cmd_parts += ["--enforce-eager"]
        command = " ".join(cmd_parts) + f" >> {shlex.quote(str(log_path))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", command], check=True)
        jobs.append({"shard_index": shard_index, "gpu": gpu, "session": session, "log": str(log_path), "command": command})
    jobs_path = base / "jobs.json"
    jobs_path.write_text(json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"jobs": jobs, "outputs": {"jobs": str(jobs_path)}}


def main() -> None:
    args = parse_args()
    if args.mode == "prompts":
        result = build_teacher_sft_prompts(
            prompt_source=args.prompt_source,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    elif args.mode == "shard":
        if args.shard_index is None:
            raise ValueError("shard 模式必须传 --shard-index。")
        result = run_teacher_sft_shard(shard_index=args.shard_index, **_shard_kwargs(args))
    elif args.mode == "merge":
        result = merge_teacher_sft_shards(output_dir=args.output_dir, num_shards=args.num_shards, model=args.model)
    else:
        result = _launch_shards(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
