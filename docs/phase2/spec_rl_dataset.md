# Phase 2 数据规格

## 目标

为 GRPO 提供一份短、干净、可验证、适合 rollout 的冻结训练工件。

## 默认数据源

- `nlile/NuminaMath-1.5-RL-Verifiable`

当前不在 v1 整体更换主数据集。

## 冻结工件

- `data/grpo/train_grpo_2k_short.jsonl`

GRPO 不直接复用 SFT 工件路径。

## 样本选择

默认规则：

- 总量：`2000`
- 按 `problem_type` 分层采样
- `response token length = 64~256`
- `final_answer` 可稳定抽取
- 可映射到 boxed 协议

## 字段约定

每条记录至少包含：

```json
{
  "id": "...",
  "question": "...",
  "prompt": "...",
  "final_answer": "...",
  "source": "numinamath",
  "problem_type": "Algebra"
}
```

可选调试字段：

- `response_tokens`
- `reference_solution`

## 原则

- 优先短 response，降低 rollout 难度
- 保留分层采样，避免 2k 全偏到单一题型
- 不把原始长解答全量直接拿来做 GRPO 训练输入

## 当前实现

- 数据准备入口：[prepare_grpo_data.py](/home/chy/code/active/rl/scripts/prepare_grpo_data.py)

## 非目标

- v1 不追求覆盖最大难度
- v1 不引入第二套独立 RL 数据格式
