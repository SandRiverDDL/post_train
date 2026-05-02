#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/dataY/fsw/cache/huggingface/hub/models--hbx--JustRL-DeepSeek-1.5B/snapshots/0637e4096c789c67f9eecbe8355e0bdeddede1c2}"
OUT="${OUT:-data/rollout/math_sft/justrl_deepseek_math_all_len3650}"
SAMPLE_SIZE="${SAMPLE_SIZE:-all}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3650}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
NUM_SHARDS="${NUM_SHARDS:-2}"
GPU0="${GPU0:-3}"
GPU1="${GPU1:-4}"
LOG_DIR="${LOG_DIR:-$OUT/logs}"
RETAINED_COUNT="${RETAINED_COUNT:-1000}"
MIX_LONG_SAMPLE_SIZE="${MIX_LONG_SAMPLE_SIZE:-1000}"
PID0=""
PID1=""

if [[ "$NUM_SHARDS" != "2" ]]; then
  echo "当前脚本只编排 2 个并行 shard；如需更多 shard，请直接调用 scripts/rollout_math_sft.py。" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

cleanup() {
  if [[ -n "$PID0" ]]; then kill "$PID0" 2>/dev/null || true; fi
  if [[ -n "$PID1" ]]; then kill "$PID1" 2>/dev/null || true; fi
}

trap cleanup INT TERM

echo "shard 0: GPU=$GPU0 log=$LOG_DIR/shard0.log"
CUDA_VISIBLE_DEVICES="$GPU0" .venv/bin/python scripts/rollout_math_sft.py \
  --mode shard \
  --model "$MODEL" \
  --output-dir "$OUT" \
  --sample-size "$SAMPLE_SIZE" \
  --responses-per-prompt "$RESPONSES_PER_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --num-shards "$NUM_SHARDS" \
  --shard-index 0 \
  > "$LOG_DIR/shard0.log" 2>&1 &

PID0=$!

echo "shard 1: GPU=$GPU1 log=$LOG_DIR/shard1.log"
CUDA_VISIBLE_DEVICES="$GPU1" .venv/bin/python scripts/rollout_math_sft.py \
  --mode shard \
  --model "$MODEL" \
  --output-dir "$OUT" \
  --sample-size "$SAMPLE_SIZE" \
  --responses-per-prompt "$RESPONSES_PER_PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --num-shards "$NUM_SHARDS" \
  --shard-index 1 \
  > "$LOG_DIR/shard1.log" 2>&1 &

PID1=$!

echo "tail -f $LOG_DIR/shard0.log"
echo "tail -f $LOG_DIR/shard1.log"
echo "INTERVAL=10 OUT=$OUT runs/rollout/watch_justrl_math_shards.sh"

STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
trap - INT TERM
if [[ "$STATUS" -ne 0 ]]; then
  exit "$STATUS"
fi

echo "merging raw shards"
.venv/bin/python scripts/rollout_math_sft.py \
  --mode merge \
  --model "$MODEL" \
  --output-dir "$OUT" \
  --raw-shards \
    "$OUT/shards/shard0-of-2/raw_samples.jsonl" \
    "$OUT/shards/shard1-of-2/raw_samples.jsonl"

echo "selecting SFT datasets"
.venv/bin/python scripts/select_math_sft.py \
  --model "$MODEL" \
  --raw-samples "$OUT/raw/raw_samples.merged.jsonl" \
  --output-dir "$OUT" \
  --retained-count "$RETAINED_COUNT" \
  --mix-long-sample-size "$MIX_LONG_SAMPLE_SIZE"

echo "wrote rollout and SFT outputs under $OUT"
