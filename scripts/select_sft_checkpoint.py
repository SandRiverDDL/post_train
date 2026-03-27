#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_eval_config
from post_train.sft_selection import run_checkpoint_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评测 SFT checkpoints 并自动选择 dev 最优模型。")
    parser.add_argument("--eval-config", required=True, help="评测配置文件路径")
    parser.add_argument("--train-output-dir", required=True, help="SFT 训练输出目录")
    parser.add_argument("--dataset", required=True, help="用于选择 checkpoint 的 dev 数据集路径")
    parser.add_argument("--backend", default=None, choices=("vllm",), help="覆盖评测后端")
    parser.add_argument("--batch-size", default=None, help="覆盖 batch size")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="覆盖生成长度上限")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_cfg = load_eval_config(args.eval_config)
    result = run_checkpoint_selection(
        train_output_dir=args.train_output_dir,
        dataset_path=args.dataset,
        eval_cfg=eval_cfg,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    print(f"wrote_ranking={result['ranking_path']}")
    print(f"wrote_best={result['best_path']}")


if __name__ == "__main__":
    main()
