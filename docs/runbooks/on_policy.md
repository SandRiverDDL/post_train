# On-Policy Runbook

这份文档覆盖当前主线：单轮数据准备、单轮训练、单轮评测，以及自动循环。

## 单轮数据准备

```bash
.venv/bin/python scripts/prepare_on_policy_sft_data.py --config configs/on_policy/data.yaml
```

默认口径：

- seed 模型：`outputs/stage1_sft_5000/checkpoint-200`
- query registry：`data/stage1_train_5000.jsonl`
- 语义：已见 5000 题上的 self-refine / instability mining
- 单轮 prepare 默认 full-harvest 整个 registry
- 每题采样 `4` 个 responses
- 保留全量 `raw_samples`
- 导出 `any_correct_shortest` 与 `mixed_only_shortest` 两套 retained

默认产物：

- `data/on_policy/query_pool.round1.jsonl`
- `data/on_policy/raw_samples.round1.jsonl`
- `data/on_policy/train.round1.jsonl`
- `data/on_policy/train.round1.mixed_only_shortest.jsonl`
- `data/on_policy/train.round1.report.json`

## 单轮训练

训练 `any_correct_shortest`：

```bash
.venv/bin/python scripts/train_sft.py --config configs/on_policy/sft.yaml
```

训练 `mixed_only_shortest`：

```bash
.venv/bin/python scripts/train_sft.py --config configs/on_policy/sft_mixed.yaml
```

默认行为：

- 从 `outputs/stage1_sft_5000/checkpoint-200` 继续训练
- 在 retained 数据上训练 `2 epoch`
- 默认 `learning_rate=1e-5`

## 单轮评测

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval/default.yaml \
  --runner vllm_raw \
  --model outputs/on_policy_sft/round1 \
  --dataset data/eval/global_dev_math500_150.jsonl \
  --batch-size 6 \
  --output outputs/eval
```

## 自动循环

```bash
.venv/bin/python scripts/run_on_policy_loop.py --config configs/on_policy/loop.yaml
```

默认行为：

- 从 `seed_model` 开始进入 round1
- 每轮抽样数量由 `round_query_count` 单独控制，默认 `256`
- 每轮自动串起 `prepare -> train -> holdout eval`
- query 按 epoch 洗牌消费，先无放回扫完整个池，再重洗继续
- 连续 `3` 轮没有更高的 holdout accuracy 就停止
- 默认最多跑 `10` 轮

默认产物：

- `outputs/on_policy_loop/history.json`
- `outputs/on_policy_loop/final_summary.json`
- `outputs/on_policy_loop/round*/round_summary.json`
- `data/on_policy_loop/round*/`

## 手动循环建议

1. 准备 round-k retained 数据
2. 训练 round-k SFT
3. 用该轮最终模型跑 `gsm8k` / `math500`
4. 把该轮最终模型作为下一轮 `generation_model`
