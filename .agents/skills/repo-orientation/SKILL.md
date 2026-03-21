---
name: repo-orientation
description: 在新会话开始时使用。用于快速理解仓库结构、当前开发阶段、当前任务以及系统 pipeline。适用于任何需要快速熟悉项目的场景。
---

# Repo Orientation：快速理解项目

## 目标

在新会话中快速理解：

- 项目目标
- 当前开发阶段
- 当前任务
- 系统 pipeline


---

# Step 1 — 阅读项目文档

按以下顺序阅读：

1. `AGENTS.md`
2. `ROADMAP.md`
3. `STATE.md`
4. `TASK.md`

提取信息：

- 项目目标
- 当前 Phase
- 当前任务
- 已知问题
- 文档与代码冲突点

注意：

- 优先顺序读取，避免无意义重复读取；只有在发现关键歧义或与代码冲突时，才回看必要片段。
- 不要反复读取相同文档
- 若代码与文档冲突，以代码为准


---

# Step 2 — 构建系统模型

根据文档推断系统 pipeline。

示例：


dataset
→ tokenizer
→ rollout
→ reward
→ trainer
→ evaluation


识别：

- 每个模块职责
- 输入数据
- 输出数据
- 模块依赖


---

# Step 3 — 定位关键模块

找到实现当前 Phase 的关键代码文件。

常见关键组件：

- dataset loader
- tokenizer
- rollout / sampler
- reward
- trainer
- evaluation


---

# Step 4 — 输出项目理解

总结以下信息：


Project Summary

当前 Phase:
...

系统 Pipeline:
...

当前任务:
...

关键模块:
...
