set shell := ["bash", "-cu"]

config := "configs/sft.yaml"
train_eval_dataset := "data/train_sft.jsonl"
dev_dataset := "data/eval/gsm8k_dev200.jsonl"
test_dataset := "data/eval/math500_test.jsonl"

prepare:
    .venv/bin/python scripts/prepare_data.py

prepare-tiny-overfit input="data/train_sft.jsonl" output="data/train_tiny_overfit.jsonl" num_samples="16" seed="42":
    .venv/bin/python scripts/prepare_tiny_overfit.py \
      --input {{input}} \
      --output {{output}} \
      --num-samples {{num_samples}} \
      --seed {{seed}}

prepare-tiny-overfit-protocol:
    .venv/bin/python scripts/prepare_tiny_overfit.py \
      --input data/train_sft.jsonl \
      --output data/train_tiny_overfit_protocol.jsonl \
      --num-samples 16 \
      --seed 42 \
      --mode protocol

prepare-tiny-overfit-shortcot:
    .venv/bin/python scripts/prepare_tiny_overfit.py \
      --input data/train_sft.jsonl \
      --output data/train_tiny_overfit_shortcot.jsonl \
      --num-samples 16 \
      --seed 42 \
      --mode shortcot \
      --max-completion-tokens 256

prepare-tiny-overfit-longclean:
    .venv/bin/python scripts/prepare_tiny_overfit.py \
      --input data/train_sft.jsonl \
      --output data/train_tiny_overfit_longclean.jsonl \
      --num-samples 16 \
      --seed 42 \
      --mode longclean

check-data input="data/train_sft.jsonl" show_failures="10":
    .venv/bin/python scripts/check_sft_data.py --input {{input}} --show-failures {{show_failures}}

train:
    .venv/bin/python scripts/train_sft.py --config {{config}}

train-tiny-overfit:
    .venv/bin/python scripts/train_sft.py \
      --config {{config}} \
      --train-dataset data/train_tiny_overfit.jsonl \
      --output-dir outputs/tiny-overfit-qwen3-1.7b \
      --epochs 10 \
      --batch-size 2 \
      --gradient-accumulation-steps 1

train-tiny-overfit-protocol:
    .venv/bin/python scripts/train_sft.py \
      --config {{config}} \
      --train-dataset data/train_tiny_overfit_protocol.jsonl \
      --output-dir outputs/tiny-overfit-protocol-qwen3-1.7b \
      --epochs 10 \
      --batch-size 2 \
      --gradient-accumulation-steps 1

train-tiny-overfit-shortcot:
    .venv/bin/python scripts/train_sft.py \
      --config {{config}} \
      --train-dataset data/train_tiny_overfit_shortcot.jsonl \
      --output-dir outputs/tiny-overfit-shortcot-qwen3-1.7b \
      --epochs 10 \
      --batch-size 2 \
      --gradient-accumulation-steps 1

train-tiny-overfit-longclean:
    .venv/bin/python scripts/train_sft.py \
      --config {{config}} \
      --train-dataset data/train_tiny_overfit_longclean.jsonl \
      --output-dir outputs/tiny-overfit-longclean-qwen3-1.7b \
      --epochs 10 \
      --batch-size 2 \
      --gradient-accumulation-steps 1

eval-tiny-overfit:
    .venv/bin/python scripts/eval_dataset.py \
      --config {{config}} \
      --dataset data/train_tiny_overfit.jsonl \
      --model outputs/tiny-overfit-qwen3-1.7b \
      --backend vllm \
      --batch-size 3 \
      --max-new-tokens 512 \
      --limit 16

eval-tiny-overfit-protocol:
    .venv/bin/python scripts/eval_dataset.py \
      --config {{config}} \
      --dataset data/train_tiny_overfit_protocol.jsonl \
      --model outputs/tiny-overfit-protocol-qwen3-1.7b \
      --backend vllm \
      --batch-size 2 \
      --max-new-tokens 128 \
      --limit 16 \
      --output outputs/eval_tiny_overfit_protocol.json

eval-tiny-overfit-shortcot:
    .venv/bin/python scripts/eval_dataset.py \
      --config {{config}} \
      --dataset data/train_tiny_overfit_shortcot.jsonl \
      --model outputs/tiny-overfit-shortcot-qwen3-1.7b \
      --backend vllm \
      --batch-size 2 \
      --max-new-tokens 256 \
      --limit 16 \
      --output outputs/eval_tiny_overfit_shortcot.json

eval-tiny-overfit-longclean:
    .venv/bin/python scripts/eval_dataset.py \
      --config {{config}} \
      --dataset data/train_tiny_overfit_longclean.jsonl \
      --model outputs/tiny-overfit-longclean-qwen3-1.7b \
      --backend vllm \
      --batch-size 2 \
      --max-new-tokens 756 \
      --limit 16 \
      --output outputs/eval_tiny_overfit_longclean.json

log-tiny-overfit-longclean:
    .venv/bin/python scripts/log_result.py \
      --input outputs/eval_tiny_overfit_longclean.json \
      --run-name tiny-overfit-longclean \
      --dataset train_tiny_overfit_longclean \
      --mode longclean

log-tiny-overfit-shortcot:
    .venv/bin/python scripts/log_result.py \
      --input outputs/eval_tiny_overfit_shortcot.json \
      --run-name tiny-overfit-shortcot \
      --dataset train_tiny_overfit_shortcot \
      --mode shortcot

log-tiny-overfit-protocol:
    .venv/bin/python scripts/log_result.py \
      --input outputs/eval_tiny_overfit_protocol.json \
      --run-name tiny-overfit-protocol \
      --dataset train_tiny_overfit_protocol \
      --mode protocol

eval-dev model="" backend="" batch_size="4" max_new_tokens="96" limit="" output="":
    .venv/bin/python scripts/eval_dataset.py \
      --config {{config}} \
      --dataset {{dev_dataset}} \
      {{ if backend != "" { "--backend " + backend } else { "" } }} \
      {{ if model != "" { "--model " + model } else { "" } }} \
      --batch-size {{batch_size}} \
      --max-new-tokens {{max_new_tokens}} \
      {{ if limit != "" { "--limit " + limit } else { "" } }} \
      {{ if output != "" { "--output " + output } else { "" } }}

eval-train:
    .venv/bin/python scripts/eval_dataset.py \
      --config {{config}} \
      --dataset data/train_sft.jsonl

eval-test model="" backend="" batch_size="4" max_new_tokens="96" limit="" output="":
    .venv/bin/python scripts/eval_dataset.py \
      --config {{config}} \
      --dataset {{test_dataset}} \
      {{ if backend != "" { "--backend " + backend } else { "" } }} \
      {{ if model != "" { "--model " + model } else { "" } }} \
      --batch-size {{batch_size}} \
      --max-new-tokens {{max_new_tokens}} \
      {{ if limit != "" { "--limit " + limit } else { "" } }} \
      {{ if output != "" { "--output " + output } else { "" } }}
