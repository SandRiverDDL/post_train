# GRPO Runbook

这份文档覆盖 `GRPO` 数据准备与训练的最小运行路径。

## 数据准备

```bash
.venv/bin/python scripts/prepare_grpo_data.py --config configs/grpo/data.yaml
```

默认行为：

- 从 `rd211/Big-Math-RL-Verified-Filtered` 的 `train` split 读取题目
- 仅保留 `0.25 < llama8b_solve_rate < 0.5` 的样本
- 与 `data/on_policy_loop/query_strategy/candidate_pool.jsonl` 混合
- 先做 exact dedup，再按 `anchor_share=0.3` 采样混合

默认产物：

- `data/grpo/train.jsonl`
- `data/grpo/train.report.json`

## 训练

保守单卡配置：

```bash
.venv/bin/python scripts/train_grpo.py --config configs/grpo/train_cheap.yaml
```

24G 4090 配置：

```bash
.venv/bin/python scripts/train_grpo.py --config configs/grpo/train_4090.yaml
```

默认行为：

- reward 从 `configs/grpo/reward.yaml` 读取
- 正确答案奖励 `1.0`
- parse 失败惩罚 `-0.5`
- 当前默认训练路线是纯 LoRA：
  - `load_in_4bit=false`
  - `compute_dtype=bfloat16`
- 当前稳定默认超参数：
  - `max_train_samples=500`
  - `train_subset_mode=fixed_random`
  - `train_subset_seed=42`
  - `mask_truncated_completions=true`
- 其余训练超参数直接以对应 YAML 为准
- 训练端会先从完整 `train.jsonl` 中取固定随机 500 条，再交给 GRPOTrainer
- 默认 loss 为 `dr_grpo`
- 默认 `wandb` 只记录主面板 5 个指标：
  - `reward/correctness`
  - `reward/parse_penalty`
  - `completions/mean_length`
  - `clip_ratio/region_mean`
  - `learning_rate`
- 若需要排障，再打开 `wandb_debug_metrics=true`

注意：

- 当前实现会在训练前做 preflight
- 没有 GPU 会直接失败
- `use_vllm=true` 时若本机不是 `vllm==0.10.2` 会直接失败
