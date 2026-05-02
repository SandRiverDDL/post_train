#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-data/rollout/math_sft/justrl_deepseek_math_all_len3650}"
LOG_DIR="${LOG_DIR:-$OUT/logs}"
INTERVAL="${INTERVAL:-10}"

latest_progress() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    echo "日志尚不存在：$log_path"
    return
  fi
  local line
  line="$(tr '\r' '\n' < "$log_path" | grep -E 'Processed prompts|Adding requests|ERROR|Traceback|wrote' | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    tail -n 1 "$log_path" 2>/dev/null || true
  else
    echo "$line"
  fi
}

while true; do
  date '+%F %T'
  echo "shard0: $(latest_progress "$LOG_DIR/shard0.log")"
  echo "shard1: $(latest_progress "$LOG_DIR/shard1.log")"
  echo
  sleep "$INTERVAL"
done
