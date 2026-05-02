#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/lightning_opd/candidate.yaml}"
STAGES="${STAGES:-prompts,shard,merge}"
eval "$(
  PYTHONPATH=src .venv/bin/python - "$CONFIG" <<'PY'
import os
import shlex
import sys
from pathlib import Path

from post_train.config import load_lightning_opd_data_config

cfg = load_lightning_opd_data_config(sys.argv[1])

items = {
    "OUT": cfg.output_dir,
    "RAW_ROLLOUT_DIR": cfg.output_dir,
    "PROMPT_SOURCE": cfg.prompt_source,
    "SAMPLE_SIZE": cfg.sample_size,
    "SEED": cfg.seed,
    "TOP_K": cfg.top_k,
    "GPUS": ",".join(str(item) for item in ([cfg.gpu0] if cfg.num_shards == 1 else [cfg.gpu0, cfg.gpu1])),
    "STUDENT_MODEL": cfg.student_model,
    "STUDENT_BASE_MODEL": cfg.student_base_model or "",
    "TEACHER_MODEL": cfg.teacher_model,
    "TOKENIZER_NAME": cfg.tokenizer_name or "",
    "MAX_NEW_TOKENS": cfg.max_new_tokens,
    "MAX_MODEL_LEN": cfg.max_model_len,
    "TEMPERATURE": cfg.temperature,
    "TOP_P": cfg.top_p,
    "GPU_MEMORY_UTILIZATION": cfg.gpu_memory_utilization,
    "TEACHER_BATCH_SIZE": cfg.teacher_batch_size,
    "TEACHER_LOAD_IN_4BIT": "1" if cfg.teacher_load_in_4bit else "0",
}
items["LOG_DIR"] = Path(items["OUT"]) / "logs"
items["CACHE_DIR"] = Path(items["OUT"]) / "cache"
items["TMPDIR"] = Path(items["OUT"]) / "cache" / "tmp"
items["TRITON_CACHE_DIR"] = Path(items["OUT"]) / "cache" / "triton"
items["TORCHINDUCTOR_CACHE_DIR"] = Path(items["OUT"]) / "cache" / "torchinductor"
items["CUDA_CACHE_PATH"] = Path(items["OUT"]) / "cache" / "cuda"
items["XDG_CACHE_HOME"] = Path(items["OUT"]) / "cache" / "xdg"
items["NUM_SHARDS"] = len(str(os.environ.get("GPUS", items["GPUS"])).split(","))

for key, value in items.items():
    selected = os.environ.get(key, str(value))
    print(f"{key}={shlex.quote(selected)}")
PY
)"
PIDS=()

has_stage() {
  local needle="$1"
  case ",$STAGES," in
    *,"$needle",*) return 0 ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ "${#GPU_ARRAY[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "GPUS 数量必须等于 NUM_SHARDS：GPUS=$GPUS NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
export TMPDIR
export TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR
export CUDA_CACHE_PATH
export XDG_CACHE_HOME

cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}

trap cleanup INT TERM

echo "config=$CONFIG"
echo "stages=$STAGES"
echo "gpus=$GPUS num_shards=$NUM_SHARDS tmpdir=$TMPDIR"

if has_stage prompts; then
  .venv/bin/python scripts/prepare_lightning_opd_data.py \
    --config "$CONFIG" \
    --mode prompts \
    --output-dir "$OUT" \
    --prompt-source "$PROMPT_SOURCE" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED"
fi

if has_stage shard || has_stage teacher; then
  mode="shard"
  if has_stage teacher && ! has_stage shard; then
    mode="teacher"
  fi
  for shard_index in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$shard_index]}"
    log_path="$LOG_DIR/${mode}${shard_index}.log"
    echo "$mode shard $shard_index: GPU=$gpu log=$log_path"
    bool_arg="--teacher-load-in-4bit"
    if [[ "$TEACHER_LOAD_IN_4BIT" != "1" ]]; then
      bool_arg="--no-teacher-load-in-4bit"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/prepare_lightning_opd_data.py \
      --config "$CONFIG" \
      --mode "$mode" \
      --output-dir "$OUT" \
      --raw-rollout-dir "$RAW_ROLLOUT_DIR" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard_index" \
      --student-model "$STUDENT_MODEL" \
      --student-base-model "$STUDENT_BASE_MODEL" \
      --teacher-model "$TEACHER_MODEL" \
      --tokenizer-name "$TOKENIZER_NAME" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --teacher-batch-size "$TEACHER_BATCH_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      "$bool_arg" \
      > "$log_path" 2>&1 &
    PIDS+=("$!")
    echo "tail -f $log_path"
  done
fi

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || STATUS=1
done
trap - INT TERM
if [[ "$STATUS" -ne 0 ]]; then
  exit "$STATUS"
fi

if has_stage merge; then
  .venv/bin/python scripts/prepare_lightning_opd_data.py \
    --config "$CONFIG" \
    --mode merge \
    --output-dir "$OUT" \
    --num-shards "$NUM_SHARDS"
fi

echo "wrote Lightning-OPD data under $OUT"
