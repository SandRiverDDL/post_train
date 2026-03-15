---
name: llm-bug-hunter
description: 当 LLM 训练或 evaluation 出现异常时使用。适用于 SFT、RL、GRPO 等后训练 pipeline 的 BUG 定位与修复。
---

# LLM Bug Hunter：后训练 Pipeline 调试

## 目标

定位并修复影响 **LLM 后训练正确性** 的 BUG。

适用场景：

- SFT 训练异常
- RL / GRPO 训练异常
- rollout 逻辑错误
- reward 计算错误
- evaluation 结果异常
- parser / metric 错误


优先级：

1. pipeline 正确性
2. 算法逻辑正确性
3. evaluation 正确性


避免：

- 代码风格审查
- 不必要重构
- 架构 redesign


---

# LLM Pipeline 模型

典型 pipeline：


dataset
↓
tokenizer
↓
prompt construction
↓
rollout / generation
↓
reward / verifier
↓
trainer
↓
evaluation


调试时必须逐层验证。


---

# Step 1 — 理解训练阶段

首先确认当前 Phase：

- SFT
- RL / GRPO
- evaluation
- inference

如果项目尚未理解：

使用 `repo-orientation` skill。


---

# Step 2 — 检查 Dataset

验证：

- dataset split 是否正确
- prompt / response 字段
- answer 格式
- tokenizer 截断情况


常见问题：

- response 被截断
- answer 不在 response
- dataset split 错误
- prompt template 不一致


---

# Step 3 — 检查 Tokenization

验证：

- tokenizer 是否与模型匹配
- special tokens 是否正确
- sequence length 是否截断


常见问题：


tokenizer mismatch
sequence truncation
missing special tokens



---

# Step 4 — 检查 Rollout

RL / GRPO 特别容易出错。

验证：

- rollout 数量
- temperature / sampling
- generation length


常见问题：

- rollout 数量错误
- max_length 太小
- generation 提前停止


---

# Step 5 — 检查 Reward

验证：

- reward parser
- reward scaling
- reward 是否正确传播


常见问题：


reward parser ≠ evaluation parser
reward 恒定
reward NaN


这会导致训练信号错误。


---

# Step 6 — 检查 Trainer

验证：

- loss 是否正确
- gradient 是否更新
- optimizer step 是否生效


检查：

- 参数是否变化
- learning rate 是否正确


---

# Step 7 — 检查 Evaluation

evaluation 常见 BUG：

- parser 错误
- metric 计算错误
- dataset 使用错误


必须确保：


reward logic == evaluation logic



否则：

训练目标与 evaluation 不一致。


---

# Step 8 — Root Cause 分析

建立完整因果链：


异常指标
↓
直接原因
↓
根本原因


确保根因能解释：

- 指标变化
- 训练行为


---

# Step 9 — 最小修复

修复原则：

- 最小修改
- 不改变架构
- 修复根因


只有在 Root Cause 已确认时才提供 patch。


---

# 输出格式

Findings

Issue 1

Problem:
...

Evidence:
...

Direct Cause:
...

Root Cause:
...

Fix Proposal:
...

Risk:
...


Issue 2

...