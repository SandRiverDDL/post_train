# Runbook

## 默认流程

### 1. 生成数据

```bash
just prepare
```

常用覆盖：

```bash
.venv/bin/python scripts/prepare_data.py --min-length 256 --max-length 1024
```

### 2. 检查训练数据

```bash
just check-data
```

期望至少满足：

- `boxed_rate=1.0`
- `parse_success_rate=1.0`
- `consistent_rate=1.0`
- `empty_final_answer=0`

### 3. 启动 SFT

```bash
just train
```

### 4. 跑开发集评测

```bash
just eval-dev
```

常用覆盖：

```bash
just eval-dev limit=5
just eval-dev model=outputs/sft-qwen3-1.7b
just eval-dev batch_size=2 max_new_tokens=64
```

### 5. 跑训练集工程验收

```bash
just eval-train
```

用途：

- 检查 `format_success / parse_success / normalized_accuracy`
- 判断 LoRA 后模型是否已经学会 `Final answer: \boxed{...}` 协议
- 作为进入 GRPO 前的工程门槛，不用于宣称泛化能力

常用覆盖：

```bash
just eval-tiny-overfit
```

### 6. 跑测试集评测

```bash
just eval-test
```

常用覆盖：

```bash
just eval-test model=outputs/sft-qwen3-1.7b
just eval-test limit=20
```

## 参数约定

- `CONFIG`：评测配置文件路径，默认 `configs/eval.yaml`
- `model`：覆盖评测模型；不传则使用配置中的 base model
- `backend`：`vllm` 或 `hf`
- `batch_size`：评测 batch size
- `max_new_tokens`：评测最大生成 token 数
- `limit`：仅跑前 N 条，适合 smoke test

如果不想用 `just`，也可以直接调用 Python CLI，例如：

```bash
.venv/bin/python scripts/eval_dataset.py \
  --dataset data/eval/gsm8k_dev200.jsonl \
  --backend vllm \
  --batch-size 4 \
  --max-new-tokens 96 \
  --limit 5
```

训练集工程验收示例：

```bash
.venv/bin/python scripts/eval_dataset.py \
  --dataset data/train_sft.jsonl \
  --model outputs/sft-qwen3-1.7b \
  --backend vllm \
  --batch-size 2 \
  --max-new-tokens 256 \
  --limit 20
```

## vLLM 说明

- 默认 backend 是 `vllm`
- 当前默认 `eval_gpu_memory_utilization=0.85`
- 如果 engine 初始化仍失败，可继续降低到 `0.8`
- 如果评测 base model 走本地 cache，默认模型路径来自 `configs/eval.yaml`
- `hf` 后端仅建议作为排障备用；默认工作流统一使用 `vllm`
