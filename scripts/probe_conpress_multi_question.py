#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.rollout.conpress_probe import run_conpress_probe
from post_train.rollout.math_sft import DEFAULT_MATH_DATASET
from post_train.tracking import log_data_artifacts, log_data_result, start_run


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值：{value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 ConPress 多题 prompt 的切分、格式与正确率稳定性。")
    parser.add_argument("--model", required=True, help="HF 模型 ID 或本地模型路径。")
    parser.add_argument("--output-dir", default="data/probes/conpress_n3_probe", help="输出目录。")
    parser.add_argument("--dataset", default=DEFAULT_MATH_DATASET, help="默认 EleutherAI/hendrycks_math。")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=99, help="总题目数，会向下取整到 questions-per-prompt 的倍数。")
    parser.add_argument("--questions-per-prompt", type=int, default=3)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--levels", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--query-pool", default=None, help="直接使用已有 query_pool.jsonl，保证同题对照；指定后不重新采样。")
    parser.add_argument("--num-shards", type=int, default=1, help="pack 分片总数，用于多进程数据并行。")
    parser.add_argument("--shard-index", type=int, default=None, help="当前 pack 分片编号，从 0 开始。")
    parser.add_argument("--tokenizer-name", default=None, help="用于 chat template 的 tokenizer；默认等于 model。")
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat-template-enable-thinking", type=parse_bool, default=None)
    parser.add_argument("--exclude-visual", action="store_true", help="过滤含 asy/diagram/figure 等视觉题面标记的题。")
    parser.add_argument(
        "--pack-strategy",
        choices=["contiguous", "shuffle", "balanced_level"],
        default=None,
        help="多题打包策略；默认 query-pool 用 contiguous，重新采样用 shuffle。",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["default", "xml_oneshot", "xml_tags"],
        default="default",
        help="prompt 格式；xml_tags 只给标签约束，xml_oneshot 会额外放示例。",
    )
    parser.add_argument("--assistant-prefill", default=None, help="追加到 assistant generation prompt 后的预填文本，例如 '<think>\\nQuestion 1:'。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with start_run(
        run_name=f"conpress-probe-n{args.questions_per_prompt}",
        route="conpress_probe",
        output_dir=args.output_dir,
        params=vars(args),
    ):
        report = run_conpress_probe(
            model=args.model,
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            split=args.split,
            sample_size=args.sample_size,
            questions_per_prompt=args.questions_per_prompt,
            samples_per_prompt=args.samples_per_prompt,
            levels=args.levels,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            cache_dir=args.cache_dir,
            query_pool_path=args.query_pool,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            use_chat_template=args.use_chat_template,
            chat_template_enable_thinking=args.chat_template_enable_thinking,
            tokenizer_name=args.tokenizer_name,
            exclude_visual=args.exclude_visual,
            pack_strategy=args.pack_strategy,
            prompt_style=args.prompt_style,
            assistant_prefill=args.assistant_prefill,
        )
        log_data_result({"outputs": report["outputs"], "report": report["metrics"]}, artifact_name="conpress_probe.json")
        log_data_artifacts([Path(report["outputs"]["report"])], artifact_path="data")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
