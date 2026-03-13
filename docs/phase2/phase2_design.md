# Phase 2：GRPO 训练

## 目标

训练一个可稳定 rollout、可稳定解析 boxed 协议、并能继续扩展的数学 GRPO 基线，当前主干优先 `Unsloth GRPO` 的单卡可运行性。

## 成功标准

- `scripts/train_grpo.py` 能在单卡环境稳定跑过前几个训练 step
- reward 解析稳定，不再因为 boxed 协议失败而大面积掉样本
- 完成 `base vs SFT vs GRPO` 对照
- `GSM8K dev200` 相比 SFT 有正向提升

## Pipeline

数据集 -> rollout -> reward -> GRPO update -> eval

## 默认决策

- trainer：`Unsloth PatchFastRL("GRPO")`
- loss：`dr_grpo`
- 可切换：`dapo`
- cold start：Phase 1 SFT LoRA
- reward：`combined_reward(correct/wrong/parse_fail + format bonus + length penalty)`
- data：`GSM8K train 2k short`
- HES：只预留接口，v1 不实现
- 训练形式：QLoRA 风格，不做全量微调

## 当前阻塞

当前最大的工程阻塞仍然是单卡环境下 GRPO 训练实现本身的可运行性与显存稳定性；reward 已收敛为单一组合口径。

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
