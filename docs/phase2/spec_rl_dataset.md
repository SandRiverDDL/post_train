# Phase 2 数据规格

## 目标

为 GRPO 提供一份短、干净、可验证、适合 rollout 的冻结训练工件。

## 默认数据源

- `openai/gsm8k` `train(main)`

当前 GRPO 主线不再默认使用 `NuminaMath`，因为对 `1.7B/4B` 冷启动 policy 来说，reward 过于稀疏。

## 冻结工件

- `data/grpo/train_grpo_gsm8k_2k_short.jsonl`
- `data/grpo/train_grpo_gsm8k_tiny_short.jsonl`（单卡调试工件）

GRPO 不直接复用 SFT 工件路径。

## 样本选择

默认规则：

- 总量：`2000`
- 随机采样
- `response token length <= 128`
- `final_answer` 可稳定抽取
- 可映射到 boxed 协议

tiny 调试规则：

- 总量：`64`
- 随机采样
- `response token length <= 128`
- 仅用于快速验证 reward 与 rollout 是否工作

## 字段约定

每条记录至少包含：

```json
{
  "id": "...",
  "question": "...",
  "prompt": "...",
  "final_answer": "...",
  "source": "gsm8k",
  "problem_type": "Arithmetic"
}
```

可选调试字段：

- `response_tokens`
- `reference_solution`

## 原则

- 优先短 response，降低 rollout 难度
- 优先选择 reward 更密集、答案更短的题源
- 主线优先保证 `1.7B` 单卡可训练，而不是追求更难数学题覆盖

## 当前实现

- 数据准备入口：[prepare_grpo_data.py](/home/chy/code/active/rl/scripts/prepare_grpo_data.py)
- 离线打分入口：[score_grpo_candidates.py](/home/chy/code/active/rl/scripts/score_grpo_candidates.py)
- 训练子集选择入口：[select_grpo_scored_subset.py](/home/chy/code/active/rl/scripts/select_grpo_scored_subset.py)

## 中等难度筛选

当 `GSM8K short` 候选集较大时，GRPO 主线优先使用经 `SFT` policy 离线打分后的中等难度子集，而不是直接随机抽样。

当前默认打分规则：

- 后端：`vLLM`
- 每题采样：`4` 次
- 采样参数：`temperature=0.8`，`top_p=0.95`
- 判分协议：strict boxed

当前默认筛选规则：

- `0.25 <= correct_rate <= 0.50`
- `parse_rate >= 0.50`

打分结果应先全量持久化为 scored 工件，再由独立脚本生成 `1K` 或其他规模的训练子集。

## 非目标

- v1 不追求覆盖最大难度
- v1 不引入第二套独立 RL 数据格式
