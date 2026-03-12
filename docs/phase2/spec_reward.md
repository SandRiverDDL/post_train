# Phase 2 Reward 规格

## 目标

先实现一个结果优先、可稳定解析的最小 reward contract。

## 默认 reward 组合

- `correctness_reward`
- `parse_reward`
- `format_reward`

## 默认权重

- `correct = 1.0`
- `parse = 0.02`
- `format = 0.02`

## 语义

### correctness

- 预测答案与标准答案一致：`1.0`
- 否则：`0.0`

### parse

- 能按 boxed 协议提取答案：正辅助项
- 否则：负辅助项

### format

- 满足 strict `Final answer: \boxed{...}`：正辅助项
- 否则：负辅助项

## 设计原则

- 正确性是主信号
- parse / format 只做轻量 shaping
- 不在 v1 引入过程奖励、风格奖励、长度奖励

## 解析逻辑

继续复用当前项目已有答案解析逻辑：

- [answers.py](/home/chy/code/active/rl/src/rl/answers.py)

## 当前实现

- reward functions：[grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)

## 后续扩展位

- reward mix strategy
- truncation policy
- 更复杂的 verifier shaping

## 非目标

- v1 不做复杂过程监督
- v1 不做多阶段 reward curriculum
