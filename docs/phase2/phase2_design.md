# Phase 2：GRPO 训练

## 目标

训练一个可稳定 rollout、可稳定解析 boxed 协议、并能继续扩展的数学 GRPO 基线。

## 成功标准

- `scripts/train_grpo.py` 能稳定跑过前几个训练 step
- reward 解析稳定，不再因为 boxed 协议失败而大面积掉样本
- 完成 `base vs SFT vs GRPO` 对照
- `GSM8K dev200` 相比 SFT 有正向提升

## Pipeline

数据集 -> rollout -> reward -> GRPO update -> eval

## 默认决策

- trainer：`TRL GRPOTrainer`
- loss：`dapo`
- 可切换：`dr_grpo`
- cold start：Phase 1 SFT LoRA
- reward：`correctness + parse + format`
- data：`NuminaMath 2k short`
- HES：只预留接口，v1 不实现
- 训练形式：QLoRA 风格，不做全量微调

## 当前阻塞

当前最大的工程阻塞不是 reward，而是 GRPO 训练在真实 GPU 环境中的 dtype / quantization 兼容问题。

已观察到的真实报错：

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

更详细的背景和判断见：

- [spec_grpo.md](/home/chy/code/active/rl/docs/phase2/spec_grpo.md)
- [STATE.md](/home/chy/code/active/rl/STATE.md)

## 组件

- rollout engine
- reward function
- trainer
- evaluation

## 详细规格

- rollout：[spec_rollout.md](/home/chy/code/active/rl/docs/phase2/spec_rollout.md)
- reward：[spec_reward.md](/home/chy/code/active/rl/docs/phase2/spec_reward.md)
- trainer：[spec_grpo.md](/home/chy/code/active/rl/docs/phase2/spec_grpo.md)
- dataset：[spec_rl_dataset.md](/home/chy/code/active/rl/docs/phase2/spec_rl_dataset.md)
