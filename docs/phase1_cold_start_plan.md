# Phase 1 Cold Start Plan

> 历史归档文档：本文件记录 Phase 1 冷启动设计，不再作为当前 Phase 2 的主执行文档。当前事实来源请优先查看 [`STATE.md`](/home/chy/code/active/rl/STATE.md) 和 [`TASK.md`](/home/chy/code/active/rl/TASK.md)。

## 目标

当前 Phase 1 的首要目标不是进一步追求 benchmark 分数，而是先得到一个能稳定输出：

```text
Final answer: \boxed{...}
```

的 SFT 冷启动模型，使其能够安全进入后续 GRPO。

当前失败的根因不是数据清洗，而是：

1. 当前 SFT target 过长、过杂
2. boxed 协议只出现在长解答尾部，监督信号太弱
3. 当前训练大概率是整段文本统一做 loss，没有专门强化回答部分

因此，本计划将 Phase 1 的 SFT 拆成“协议优先”的两阶段冷启动。

## 设计原则

1. 冷启动优先学协议，不优先学长推理
2. 优先复用现有 `NuminaMath-1.5-RL-Verifiable`，先派生更适合冷启动的子集
3. 不把“原始长 solution + 尾部 boxed”作为唯一 SFT 数据
4. 训练时只对回答部分计算 loss
5. 只有当 boxed 协议稳定后，才进入 GRPO

## 数据方案

### 数据源

第一步不更换主数据集，继续使用：

- `nlile/NuminaMath-1.5-RL-Verifiable`

但不再直接把原始长 `solution` 整段作为唯一 target。

### 新增两个冷启动训练工件

#### 1. `data/train_sft_protocol.jsonl`

用途：

- 协议学习
- 快速教会模型稳定输出 boxed 最终答案

目标格式：

```text
Final answer: \boxed{...}
```

也就是说，response 只保留最终答案协议，不保留长推理。

#### 2. `data/train_sft_shortcot.jsonl`

用途：

- 在不破坏 boxed 协议的前提下，补一点短推理能力

目标格式：

```text
<极短推理，可选 1~4 行>

Final answer: \boxed{...}
```

约束：

- target 长度应明显短于当前 `train_sft.jsonl`
- 建议 target 控制在 `128~192` tokens 内
- 不保留原始超长证明型解答

### 采样建议

冷启动阶段建议优先使用：

1. 题目较短
2. 最终答案可稳定抽取
3. 原始解答结构较清晰
4. 不依赖超长证明才能成立的样本

这一步的目标不是覆盖所有数学子领域，而是先把 reward contract 学稳。

## 训练方案

### Stage A：Protocol SFT

数据：

- `data/train_sft_protocol.jsonl`

prompt：

```text
Question:
{question}

Solution:
```

response：

```text
Final answer: \boxed{...}
```

目标：

1. 稳定学会 boxed 协议
2. 提高 `format_success`
3. 提高 `parse_success`

说明：

- 这一阶段不要求强推理能力
- 先把“能被 reward 稳定解析”作为第一目标

### Stage B：Short-CoT SFT

数据：

- `data/train_sft_shortcot.jsonl`

prompt：

```text
Question:
{question}

Solution:
```

response：

```text
<短推理>

Final answer: \boxed{...}
```

目标：

1. 保持 boxed 协议不退化
2. 补充短推理能力
3. 为后续 GRPO 提供更自然的初始化

说明：

- 这一阶段仍然不直接回到原始长解答全量训练
- 如果 Stage B 明显破坏格式稳定性，应优先保留 Stage A 产物

## Loss 方案

当前训练需要改为：

- `prompt` 与 `response` 分离
- 只对 `response` 计算 loss

这里的含义不是“当前只在学 question”，而是：

- 不应让 `Question:`、问题正文、`Solution:` 前缀和回答正文拥有同等 loss 权重
- 当前最关键的监督对象是回答部分，尤其是 boxed 协议

实现要求：

1. 训练脚本支持 `prompt/response` 结构
2. 使用 completion-only masking 或等价机制
3. 明确验证 loss 只落在 response token 上

## 验收标准

### 进入 GRPO 前的硬门槛

在训练集工程验收上，至少满足：

- `format_success >= 0.95`
- `parse_success >= 0.95`

如果达不到，不进入 GRPO。

### 辅助观察指标

- `normalized_accuracy`
- `GSM8K dev200` 上相对 base 的提升
- 样本人工抽查

说明：

- 冷启动阶段，`normalized_accuracy` 不是第一优先级
- 只要协议不稳定，就不应进入 GRPO

## 实施顺序

1. 新增冷启动数据准备逻辑，产出：
   - `data/train_sft_protocol.jsonl`
   - `data/train_sft_shortcot.jsonl`
2. 修改训练脚本，使其支持 response-only loss
3. 先训练 Stage A protocol 模型
4. 在 `train` 上做 strict format 验收
5. 若协议稳定，再训练 Stage B short-CoT 模型
6. 再次进行 strict format 验收
7. 补跑 `GSM8K dev200`
8. 只有协议稳定后，才进入 GRPO

## 非目标

本计划当前不做：

1. 直接整体更换主数据集
2. 直接回到原始长 solution 全量 SFT
3. 先调 LoRA 超参数来掩盖训练目标问题
4. 在 boxed 协议不稳定时继续推进 GRPO

## 与现有文档的关系

- 问题诊断见 [REVIEW.md](/home/chy/code/active/rl/REVIEW.md)
- Phase 1 总目标见 [docs/phase1_mvp.md](/home/chy/code/active/rl/docs/phase1_mvp.md)
- 当前执行项见 [TASK.md](/home/chy/code/active/rl/TASK.md)

本文件只负责回答一个问题：

**如何把当前 Phase 1 SFT 改造成可用于 GRPO 冷启动的协议稳定版本。**
