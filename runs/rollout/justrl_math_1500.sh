#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

.venv/bin/python scripts/rollout_math_sft.py \
  --model /mnt/dataY/fsw/cache/huggingface/hub/models--hbx--JustRL-DeepSeek-1.5B/snapshots/0637e4096c789c67f9eecbe8355e0bdeddede1c2 \
  --output-dir data/rollout/math_sft/justrl_deepseek_1500 \
  --sample-size 1500 \
  --responses-per-prompt 1 \
  --retained-count 1000 \
  --mix-long-sample-size 1000 \
  --max-new-tokens 2048 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
