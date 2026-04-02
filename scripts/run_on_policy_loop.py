#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_on_policy_loop_config
from post_train.experiments import register_on_policy_loop_run
from post_train.on_policy_loop import run_on_policy_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动运行多轮 on-policy SFT，并按准确率早停。")
    parser.add_argument("--config", required=True, help="on-policy loop 配置文件路径")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--resume", action="store_true", help="继续一个未完成的 loop run。")
    mode_group.add_argument("--overwrite", action="store_true", help="删除旧的 loop 目录并从头开始。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_on_policy_loop_config(args.config)
    result = run_on_policy_loop(cfg, resume=args.resume, overwrite=args.overwrite)
    summary = register_on_policy_loop_run(
        config_path=args.config,
        loop_cfg=cfg,
        final_summary=result["summary"],
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"wrote_history={result['history_path']}")
    print(f"wrote_final_summary={result['final_summary_path']}")
    print(f"wrote_run_summary={summary['summary_path']}")


if __name__ == "__main__":
    main()
