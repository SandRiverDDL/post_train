#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/dataY/fsw/cache/huggingface/hub/models--hbx--JustRL-DeepSeek-1.5B/snapshots/0637e4096c789c67f9eecbe8355e0bdeddede1c2}"
OUT="${OUT:-data/rollout/math_sft/justrl_deepseek_math_all_len3650}"

.venv/bin/python scripts/rollout_math_sft.py \
  --mode merge \
  --model "$MODEL" \
  --output-dir "$OUT" \
  --raw-shards \
    "$OUT/shards/shard0-of-2/raw_samples.jsonl" \
    "$OUT/shards/shard1-of-2/raw_samples.jsonl"
