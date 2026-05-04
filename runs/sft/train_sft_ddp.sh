#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-${1:-}}"
if [[ -z "$CONFIG" ]]; then
    echo "usage: CONFIG=configs/stage1/xxx.yaml GPUS=0,1,2 runs/sft/train_sft_ddp.sh [--set key=value ...]" >&2
    exit 2
fi
if [[ "${1:-}" == "$CONFIG" ]]; then
    shift
fi

GPUS="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-.venv/bin/torchrun}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "python not found or not executable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "torchrun not found or not executable: $TORCHRUN_BIN" >&2
    exit 2
fi

echo "SFT DDP: config=$CONFIG gpus=$GPUS nproc_per_node=$NPROC_PER_NODE python=$PYTHON_BIN"

CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train_sft.py \
    --config "$CONFIG" \
    "$@"
