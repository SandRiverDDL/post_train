# On-Policy Runbook

这份文档覆盖当前主线：单轮数据准备、单轮训练、单轮评测，以及自动循环。

实现说明：

- `scripts/run_on_policy_loop.py` 仍是唯一自动循环入口
- 内部实现已经拆到 `src/post_train/on_policy/` 子包
- 顶层 `on_policy_loop.py` 与 `on_policy_query_strategy.py` 现在只作为兼容导出层保留

## 单轮数据准备

```bash
.venv/bin/python scripts/prepare_on_policy_sft_data.py --config configs/on_policy/data.yaml
```

默认口径：

- seed 模型：`outputs/stage1_sft_5000/checkpoint-200`
- query registry：`data/stage1/train_5000.jsonl`
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

按论文口径复现 `opSFT`：

```bash
.venv/bin/python scripts/train_sft.py --config configs/on_policy/opsft.yaml
```

默认行为：

- 从 `outputs/stage1_sft_5000/checkpoint-200` 继续训练
- 在 retained 数据上训练 `2 epoch`
- 默认 `learning_rate=1e-5`
- `configs/on_policy/opsft.yaml` 会切到 `loss_mode=opsft`、`1 epoch`、`learning_rate=5e-7`

## 单轮评测

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval/default.yaml \
  --model outputs/on_policy_sft/round1 \
  --dataset data/eval/math500_dev200.jsonl \
  --batch-size 6 \
  --output outputs/eval
```

默认结果目录示例：

- `outputs/eval/on_policy_sft/round1/math500_dev200/result.json`
- `outputs/eval/on_policy_sft/round1/math500_dev200/raw.json`

## 自动循环

```bash
.venv/bin/python scripts/run_on_policy_loop.py --config configs/on_policy/loop.yaml
```

按论文口径运行 `opSFT` loop：

```bash
.venv/bin/python scripts/run_on_policy_loop.py --config configs/on_policy/loop_opsft.yaml
```

继续未完成 run：

```bash
.venv/bin/python scripts/run_on_policy_loop.py --config configs/on_policy/loop.yaml --resume
```

清空旧 run 后重开：

```bash
.venv/bin/python scripts/run_on_policy_loop.py --config configs/on_policy/loop.yaml --overwrite
```

默认行为：

- 从 `seed_model` 开始进入 round1
- `loop.yaml` 是当前默认 on-policy 主线：`512 prompt / round`，`query_strategy=mixed_bootstrap_candidate`
- `loop_opsft.yaml` 是当前 `opSFT` 小池子迁移线：`640 prompt / round`、`8 rollouts / prompt`、`candidate:random = 50:50`
- 当前主配置默认 `query_strategy=mixed_bootstrap_candidate`
- `configs/on_policy/data_opsft.yaml` 会把 rollout `temperature` 固定到 `1.0`
- `configs/on_policy/data_opsft.yaml` 会把 rollout `temperature` 固定到 `1.0`，并把 `responses_per_query` 提到 `8`
- `configs/on_policy/loop_opsft.yaml` 还会开启 `advance_teacher_on_improvement_only=true`，且只有 holdout 提升至少 `1%` 才推进 teacher
- 当前 `opSFT` 线使用 `data/eval/math500_dev200.jsonl` 作为开发评测集；这份 dev 由 `HuggingFaceH4/MATH-500` 按 `level` 分层抽样生成
- 每轮自动串起 `prepare -> train -> holdout eval`
- bootstrap 池来自 `data/on_policy/raw_samples.round1.jsonl` 中的 `all_correct`
- candidate 池默认是 `data/on_policy/candidate_pool.mixed_1000.jsonl`
- 训练集默认不是直接用 primary retained，而是 `mixed_plus_anchor`
- anchor 默认来自 `data/stage1/train_5000.jsonl` 的原始轨迹，目标占比 `25%`
- 每轮训练默认 `1 epoch`，并在轮内自动保存约 `3~4` 个 checkpoints
- 每轮正式推进到下一轮的 teacher 是该轮 dev 上选出的 best checkpoint
- 连续 `3` 轮没有更高的 holdout accuracy 就停止
- 默认最多跑 `10` 轮
- `--resume` 只用于恢复未完成 run，不用于继续一个已自然结束的实验
- resume 只做到“按轮恢复”，不支持训练内部 checkpoint 级恢复

实验性 mixed 策略：

- 30% 从固定 bootstrap `all_correct` 池抽，70% 从外部 candidate 池抽
- candidate 题目累计两次被抽中且两次都 `all_correct` 后会冻结
- 策略状态单独落到 `data/on_policy_loop/query_strategy/`，不和稳定训练/评测流程耦合
- 若要回退旧行为，可手动切回 `query_strategy=uniform_epoch`

默认产物：

- `outputs/on_policy_loop/history.json`
- `outputs/on_policy_loop/final_summary.json`
- `outputs/on_policy_loop/round*/round_summary.json`
- `outputs/on_policy_loop/round*/holdout_eval/<task>/result.json`
- `outputs/on_policy_loop/round*/holdout_eval/<task>/raw.json`
- `data/on_policy_loop/round*/`

## 手动循环建议

1. 准备 round-k retained 数据
2. 训练 round-k SFT
3. 用该轮最终模型跑 `gsm8k` / `math500`
4. 把该轮最终模型作为下一轮 `generation_model`
