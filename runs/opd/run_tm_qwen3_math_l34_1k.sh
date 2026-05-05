#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_DIR="${VERL_DIR:-${ROOT_DIR}/../verl}"

TOKENIZER_NAME="${TOKENIZER_NAME:-/mnt/dataY/fsw/cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
DATA_ROOT="${DATA_ROOT:-${VERL_DIR}/data/opd_math/hendrycks_l4_prompt256_500_seed42}"
LEVELS="${LEVELS:-4}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
SEED="${SEED:-42}"
MAX_PROMPT="${MAX_PROMPT:-256}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-$((MAX_PROMPT + MAX_RESPONSE_LENGTH + 1))}"

STUDENT_BASE_MODEL="${STUDENT_BASE_MODEL:-/mnt/dataY/fsw/cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
STUDENT_LORA_ADAPTER="${STUDENT_LORA_ADAPTER:-${ROOT_DIR}/outputs/stage1_qwen3_1p7b_bf16_lora_qwen3_4b_instruct2507_math_correct_len4096_sft_ddp3_b2_acc6_bs36/checkpoint-150}"
TEACHER_MODEL="${TEACHER_MODEL:-/mnt/dataY/fsw/cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"

PROJECT_NAME="${PROJECT_NAME:-tm_opd_qwen3_math}"
EXP_NAME="${EXP_NAME:-qwen3_1p7b_sft150_l4_prompt256_500_k1}"
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-250}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:--1}"
LOGGER="${LOGGER:-[\"console\"]}"

STUDENT_WORLD_SIZE="${STUDENT_WORLD_SIZE:-2}"
TEACHER_WORLD_SIZE="${TEACHER_WORLD_SIZE:-1}"
STUDENT_GPU_MEMORY_UTILIZATION="${STUDENT_GPU_MEMORY_UTILIZATION:-0.65}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}"
STUDENT_MICRO_BATCH_SIZE_PER_GPU="${STUDENT_MICRO_BATCH_SIZE_PER_GPU:-1}"
STUDENT_MAX_TOKEN_LEN_PER_GPU="${STUDENT_MAX_TOKEN_LEN_PER_GPU:-$((STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH)))}"

cd "$ROOT_DIR"
PYTHONPATH=src .venv/bin/python scripts/prepare_verl_opd_data.py \
  --mode math-levels \
  --levels $LEVELS \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --max-prompt-tokens "$MAX_PROMPT" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --output-dir "$DATA_ROOT"

cd "$VERL_DIR"

export STUDENT_BASE_MODEL
export STUDENT_LORA_ADAPTER
export TEACHER_MODEL
export DATA_ROOT
export PROJECT_NAME
export EXP_NAME
export TRAIN_PROMPT_BSZ
export TOTAL_TRAINING_STEPS
export MAX_PROMPT
export MAX_RESPONSE_LENGTH
export MAX_NUM_TOKENS
export STUDENT_WORLD_SIZE
export TEACHER_WORLD_SIZE
export STUDENT_GPU_MEMORY_UTILIZATION
export TEACHER_GPU_MEMORY_UTILIZATION
export STUDENT_MICRO_BATCH_SIZE_PER_GPU
export STUDENT_MAX_TOKEN_LEN_PER_GPU
export SAVE_FREQ
export TEST_FREQ
export LOGGER
export DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE:-k1}"
export USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-True}"
export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

bash examples/on_policy_distillation_trainer/run_qwen25_math_opd.sh \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.8}" \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K:-20}" \
  distillation.teacher_models.teacher_model.inference.temperature=1.0 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.default_local_dir="${CHECKPOINT_DIR:-outputs/opd/${EXP_NAME}}"
