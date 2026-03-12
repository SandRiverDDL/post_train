---
name: bug-hunter
description: 当需要定位或修复 BUG 时使用。适用于调试 ML/LLM pipeline、训练错误、evaluation 错误或实现与设计不一致的问题。
---

# Bug Hunter：BUG 定位与修复

## 目标

定位并修复影响系统正确性的 BUG。

优先级：

1. pipeline 正确性
2. 算法正确性
3. evaluation 正确性


避免：

- 代码风格审查
- 不必要的重构
- 架构重设计


---

# Step 1 — 理解系统

如果尚未理解项目，先使用 `repo-orientation` skill。

确保理解：

- 当前 Phase
- 系统 pipeline
- 当前任务


---

# Step 2 — 检查 Pipeline

检查 pipeline 是否正确连接：


dataset
→ tokenizer
→ rollout
→ reward
→ trainer
→ evaluation


重点检查：

- 数据格式
- tensor / token shape
- batch size
- sequence length
- 模块连接


常见问题：

- tokenizer 使用错误
- sequence 被截断
- parser 使用错误
- dataset split 错误


---

# Step 3 — 检查算法逻辑

验证关键算法实现。

示例：

RL 训练：

- rollout 数量正确
- reward 计算正确
- loss 公式正确

训练流程：

- gradient 正常更新
- optimizer step 生效
- 模型参数发生变化


---

# Step 4 — 检查 Evaluation

Evaluation 是常见 BUG 来源。

检查：

- evaluation dataset
- answer parser
- metric 计算
- 与 reward 逻辑一致


示例问题：


reward parser ≠ evaluation parser


这会导致训练信号错误。


---

# Step 5 — Root Cause 分析

分析：


问题现象
→ 直接原因
→ 根本原因


不要只描述表面现象。


---

# Step 6 — 最小修复

修复原则：

- 最小修改
- 不改变架构
- 修复根本原因


---

# 输出格式

Findings

Issue 1

Problem:
...

Root Cause:
...

Fix:
...

Issue 2
...


必要时提供 patch。