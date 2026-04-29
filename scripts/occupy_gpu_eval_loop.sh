#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-2}"
MODEL_PATH="${2:-outputs/stage1_mix_long_sft/checkpoint-300}"
OUTPUT_DIR="${3:-outputs/gpu_occupy_eval_loop/gpu${GPU_ID}}"
LOG_PATH="${OUTPUT_DIR}.log"

mkdir -p "${OUTPUT_DIR}"

while true; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" .venv/bin/python scripts/eval_model.py \
    --config configs/eval/default.yaml \
    --model "${MODEL_PATH}" \
    --tasks gsm8k \
    --output "${OUTPUT_DIR}" \
    --batch-size 16 \
    --max-new-tokens 512 \
    >> "${LOG_PATH}" 2>&1
  sleep 10
done
