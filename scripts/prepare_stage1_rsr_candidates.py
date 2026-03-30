#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.config import load_stage1_rsr_candidates_config
from post_train.rsr_data import prepare_stage1_rsr_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 stage1 RSR 候选池数据。")
    parser.add_argument("--config", required=True, help="stage1 RSR 候选池配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage1_rsr_candidates_config(args.config)
    result = prepare_stage1_rsr_candidates(cfg)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"rsr_dropped_truncated={result['report']['filters']['dropped_truncated']}")
    print(f"rsr_dropped_truncated_ratio={result['report']['filters']['dropped_truncated_ratio']:.4f}")
    print(f"wrote_stage1_rsr_candidates={result['output_path']}")
    print(f"wrote_stage1_rsr_candidates_report={result['report_path']}")
    print(f"wrote_stage1_rsr_candidates_unmatched={result['unmatched_preview_path']}")


if __name__ == "__main__":
    main()
