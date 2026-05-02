#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/dataY/fsw/cache/huggingface/hub/models--hbx--JustRL-DeepSeek-1.5B/snapshots/0637e4096c789c67f9eecbe8355e0bdeddede1c2}"
OUT="${OUT:-data/rollout/math_sft/justrl_deepseek_math_all_len3650}"
RETAINED_COUNT="${RETAINED_COUNT:-1000}"
MIX_LONG_SAMPLE_SIZE="${MIX_LONG_SAMPLE_SIZE:-1000}"

.venv/bin/python scripts/select_math_sft.py \
  --model "$MODEL" \
  --raw-samples "$OUT/raw/raw_samples.merged.jsonl" \
  --output-dir "$OUT" \
  --retained-count "$RETAINED_COUNT" \
  --mix-long-sample-size "$MIX_LONG_SAMPLE_SIZE"
