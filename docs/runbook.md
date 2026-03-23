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

## 0.6B Prompt V2

用于快速迭代的新链路入口：

```bash
.venv/bin/python scripts/train_sft.py --config configs/sft_qwen3_0.6b_v2.yaml
```

```bash
.venv/bin/python scripts/eval_dataset.py \
  --config configs/eval_qwen3_0.6b_v2.yaml \
  --model outputs/sft-qwen3-0.6b-v2 \
  --dataset data/eval/gsm8k_dev200.jsonl \
  --backend vllm \
  --batch-size 4 \
  --max-new-tokens 512 \
  --output outputs/eval_sft_qwen3_0.6b_v2_gsm8k_dev200_official.json
```

```bash
.venv/bin/python scripts/train_grpo.py --config configs/grpo_base_qwen3_0.6b_v2.yaml
```

当前经验上不应默认 final 最优。每轮 `GRPO` 后至少应补评一个中早期 checkpoint，例如：

```bash
.venv/bin/python scripts/eval_dataset.py \
  --config configs/eval_qwen3_0.6b_v2.yaml \
  --model outputs/grpo-qwen3-0.6b-ablate-v2/checkpoint-25 \
  --dataset data/eval/gsm8k_dev200.jsonl \
  --backend vllm \
  --batch-size 4 \
  --max-new-tokens 512 \
  --output outputs/eval_grpo_qwen3_0.6b_v2_checkpoint25_gsm8k_dev200_official.json
```

当前 Phase 2 的默认解读顺序应为：

1. 先比较 `checkpoint-25/50` 与 final。
2. 若 early checkpoint 更好，则优先缩短单次训练窗口，而不是继续延长 step。
3. 不要仅根据 final 一个点判断本轮 GRPO 是否有效。

## GRPO Step 语义

当前主线使用的是 `Unsloth GRPO`，这里的 `step` 不要按普通 trainer 直觉理解。

约定：

- `global_step` 表示一次参数更新，不是生成了多少条 completion
- `max_steps = -1` 时，实际步数由 trainer 按 dataloader 长度自动推导
- `num_generations` 会影响有效 train batch 的约束；如果 `batch_size * gradient_accumulation_steps * world_size` 不是 `num_generations` 的整数倍，`Unsloth` 可能改写有效 `train_batch_size`
- 一旦发生这种改写，自动推导出的 `max_steps` 可能远大于直觉上的 `样本数 / batch_size`
- 是否发生改写，以运行产物里的 `trainer_state.json` 为准，重点看：
  - `train_batch_size`
  - `max_steps`
  - `num_train_epochs`

经验建议：

- 配置 `batch_size` 时，优先让它成为 `num_generations` 的整数倍
- 不要仅凭 yaml 里的 `batch_size` 和 `max_steps=-1` 手算训练总步数
- 真正开始训练后，先看一次 `checkpoint-10/trainer_state.json` 再决定 checkpoint 密度和总训练窗口

## Big-Math 扩充

先过滤外部候选：

```bash
.venv/bin/python scripts/filter_big_math_dataset.py \
  --output data/grpo/big_math_quintile2_filtered.jsonl
```

再通过 manifest 合并并去重：

```bash
.venv/bin/python scripts/merge_train_corpora.py \
  --manifest path/to/data_merge.yaml
```

最后按预算选择最终训练集：

```bash
.venv/bin/python scripts/select_merged_corpus.py \
  --manifest path/to/data_merge.yaml
```

约定：

- `filter_big_math_dataset.py` 当前固定面向 `open-r1/Big-Math-RL-Verified-Processed`
- `quintile_2` 通过 `--config-name` 选择
- `merge_train_corpora.py` 只生成 merge 后的去重 pool
- `select_merged_corpus.py` 从 pool 中按 `selection.target_size` 和每源 `min_count/max_count` 选择最终训练集
- 建议把 `data/eval/gsm8k_dev200.jsonl` 等冻结 eval 工件放进 manifest 的 `dedup_against`

## vLLM 说明

- 默认 backend 是 `vllm`
- 当前默认 `eval_gpu_memory_utilization=0.85`
- 如果 engine 初始化仍失败，可继续降低到 `0.8`
- 如果评测 base model 走本地 cache，默认模型路径来自 `configs/eval.yaml`
- `hf` 后端仅建议作为排障备用；默认工作流统一使用 `vllm`
