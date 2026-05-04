#!/usr/bin/env bash
set -euo pipefail

# 历史归档：等待 ConPress ASFT 并行选 ckpt 后评测 MATH500/GSM8K。
# 当前推荐直接用 scripts/select_sft_checkpoint_parallel.py 和 scripts/eval_model.py。
ROOT="/mnt/dataset/fengshuwen/post_train"
BEST_JSON="$ROOT/outputs/stage1_conpress_qwen3_1p7b_asft_topk/dev_eval_parallel/best_checkpoint.json"
LOG_DIR="$ROOT/logs/eval"
CONFIG="$ROOT/configs/eval/qwen3_1p7b_dev.yaml"

mkdir -p "$LOG_DIR"
cd "$ROOT"

while [[ ! -s "$BEST_JSON" ]]; do
  echo "waiting_for_best_checkpoint=$BEST_JSON"
  sleep 30
done

BEST_CKPT="$(PYTHONPATH=src .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$BEST_JSON")"
echo "best_checkpoint=$BEST_CKPT"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/eval_model.py \
  --config "$CONFIG" \
  --model "$BEST_CKPT" \
  --dataset data/eval/math500_test.jsonl \
  --backend vllm \
  --batch-size 16 \
  --max-new-tokens 2048 \
  > "$LOG_DIR/conpress_asft_best_math500_20260504.log" 2>&1 &

MATH_PID=$!

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/eval_model.py \
  --config "$CONFIG" \
  --model "$BEST_CKPT" \
  --dataset data/eval/gsm8k_test.jsonl \
  --backend vllm \
  --batch-size 16 \
  --max-new-tokens 2048 \
  > "$LOG_DIR/conpress_asft_best_gsm8k_20260504.log" 2>&1 &

GSM_PID=$!

wait "$MATH_PID"
wait "$GSM_PID"

echo "benchmark_finished=true"
