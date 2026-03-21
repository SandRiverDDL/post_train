# Phase 2 Reward 规格

## 目标

先实现一个结果优先、可稳定解析的最小 reward contract。

## 默认 reward 组合

当前主线使用**单一组合 reward**：

- `combined_reward`

所有 reward 系数都直接由训练 yaml 提供。

## 默认权重

- `combined_reward = 1.0`

## 语义

### combined reward

- relaxed 或 strict 正确性，由 `reward_use_relaxed_correctness` 控制
- 正确答案分数：`reward_correct`
- 错误答案分数：`reward_wrong`
- 解析失败分数：`reward_parse_fail`
- strict boxed 奖励：`reward_strict_boxed_bonus`
- 长度惩罚：`reward_length_coef * cleaned_completion_tokens`

语义约束：

- `wrong` 与 `parse fail` 互斥
- strict boxed bonus 可与 `correct` 或 `wrong` 叠加
- `parse fail` 不再叠加 strict bonus

## 设计原则

- 正确性仍是主信号
- `format` 只做轻量 shaping
- `length` 只做轻微线性惩罚，不替代 `max_completion_length`
- 不在 v1 引入过程奖励或风格奖励

## 解析逻辑

继续复用当前项目已有答案解析逻辑：

- [answers.py](/home/chy/code/active/rl/src/rl/answers.py)

## 当前实现

- reward functions：[grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)

## 当前日志诊断项

训练日志额外记录以下诊断指标：

- `rewards/correct_rate`
- `rewards/relaxed_correct_rate`
- `rewards/strict_correct_rate`
- `rewards/wrong_rate`
- `rewards/parse_fail_rate`
- `rewards/format_rate`
- `rewards/strict_boxed_rate`
- `rewards/mean_length_penalty`

## 非目标

- v1 不做复杂过程监督
- v1 不做多阶段 reward curriculum
