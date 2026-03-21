# Phase 2：GRPO 训练

> 当前设计文档：若与代码或状态摘要冲突，以 [`STATE.md`](/home/chy/code/active/rl/STATE.md)、[`TASK.md`](/home/chy/code/active/rl/TASK.md) 和当前实现为准。

## 目标

训练一个可稳定 rollout、可稳定解析 boxed 协议、并能继续扩展的数学 GRPO 基线，当前主干优先 `Unsloth GRPO` 的单卡可运行性。

## 成功标准

- `scripts/train_grpo.py` 能在单卡环境稳定完成一轮训练
- reward 解析稳定，不再因为 boxed 协议失败而大面积掉样本
- 完成 `base vs SFT vs GRPO` 对照
- `GSM8K dev200` 相比 SFT 至少不出现明显目标错位

## Pipeline

数据集 -> rollout -> reward -> GRPO update -> eval

## 默认决策

- trainer：`Unsloth PatchFastRL("GRPO")`
- loss：`dr_grpo`
- 可切换：`dapo`
- cold start：Phase 1 SFT LoRA
- reward：`combined_reward(relaxed correct/wrong/parse fail + weak strict bonus + length penalty)`
- data：当前主线为 `GSM8K train 1k middiff`
- eval config：默认使用 `configs/eval.yaml`
- HES：只预留接口，v1 不实现
- 训练形式：QLoRA 风格，不做全量微调

## 当前阻塞

当前最大的工程阻塞已经从“训练脚本能否启动”切换为“训练目标是否正确”。

当前真实问题：

- 最近一轮 `1K middiff` GRPO 训练后，`GSM8K dev200` 上
  - `strict-match` 上升到 `0.67`
  - `flexible-extract` 下降到 `0.57`
- 这说明训练并未崩溃，但优化目标和外部评测目标存在错位

当前最可疑因素：

- strict boxed reward 过窄
- `format/length` 这类 shaping reward 可能拉偏策略
- 只看最终 epoch checkpoint，未做中间点选择

当前默认修正方向：

- reward 系数外置到 yaml
- 配置切到 `_base_ + override` 结构
- 支持按 step 保存 checkpoint，并顺序扫 `dev200`

更详细背景见：

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
