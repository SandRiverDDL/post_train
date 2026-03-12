# Phase 2 Rollout 规格

## 目标

定义统一的 prompt contract、生成长度和采样参数，使 rollout 可稳定运行。

## Prompt Contract

默认 prompt：

```text
Question:
{question}

要求：
请给出必要推理。
最后一行必须严格写成：
Final answer: \boxed{...}

Solution:
```

## 原则

- prompt 必须显式声明 boxed 协议
- reward contract 与 prompt contract 必须一致
- 长期应与 SFT 共享协议文案，不维护两套独立标准

## 默认长度

- `max_prompt_length = 512`
- `max_completion_length = 384`
- `max_seq_length = 1024`

## 调整规则

- 不默认上来就把 completion 拉到 `768` 或 `1024`
- 只有当日志显示明显截断时，再提升 completion 上限

## 默认采样参数

- `temperature = 1.0`
- `top_p = 1.0`
- `top_k = None`
- `min_p = None`
- `repetition_penalty = 1.0`

数值兜底：

- `remove_invalid_values = true`
- `renormalize_logits = true`

## 日志

至少记录：

- reward 均值
- format / parse / correct rate
- clipped ratio
- mean completion length

## 当前实现

- prompt 构造与日志 callback：[grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)
- 训练入口：[train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py)

## 非目标

- v1 不做复杂 guided decoding
- v1 不做 HES rollout masking
