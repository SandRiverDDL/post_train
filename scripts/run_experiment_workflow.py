#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_workflow_config
from post_train.workflow import run_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单轮实验 workflow。")
    parser.add_argument("--config", required=True, help="workflow 配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_workflow_config(args.config)
    result = run_workflow(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
