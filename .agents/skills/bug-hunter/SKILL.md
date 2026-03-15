---
name: bug-hunter
description: 当系统行为异常、程序报错或结果不符合预期时使用。用于定位并修复影响系统正确性的 BUG。
---

# Bug Hunter：系统 BUG 定位与最小修复

## 目标

定位并修复影响系统**正确性**的问题。

优先级：

1. 系统行为正确性
2. 算法或逻辑正确性
3. 数据处理正确性
4. evaluation / metric 正确性


避免：

- 代码风格审查
- 不必要重构
- 架构 redesign
- 大规模代码修改


本 skill 的核心目标是：

**Root Cause Analysis + Minimal Fix**

而不是全面代码审查。


---

# 调试原则

1. **Evidence First**

不要猜测问题。

所有结论必须基于：

- 代码
- 日志
- stack trace
- 数据
- 运行结果


2. **区分直接原因与根本原因**

示例：

直接原因：

evaluation accuracy 为 0


根本原因：

answer parser 与 evaluation parser 不一致



3. **最小修改原则**

只修改解决问题所必需的代码。

避免：

- 重构模块
- 改变系统结构
- 修改无关代码


4. **Patch 触发条件**

只有在满足以下条件时才直接提供 patch：

- Root Cause 已确认
- 修复范围局部
- 修改风险低
- 不涉及架构调整

否则仅提供 Fix Proposal。


---

# Step 1 — 理解问题

收集以下信息：

- 期望行为
- 实际行为
- 错误信息 / 异常日志
- 最近代码改动
- 相关模块


如果项目结构尚未理解：

使用 `repo-orientation` skill。


---

# Step 2 — 定位问题范围

确定问题发生的位置：

常见层级：


输入数据
↓
数据处理
↓
核心逻辑
↓
模型 / 算法
↓
输出处理
↓
evaluation / metric


确定异常发生在哪一层。


---

# Step 3 — 证据收集

寻找能够解释问题的证据：

可能来源：

- stack trace
- logging
- 中间变量
- tensor / data shape
- 配置参数
- 数据格式


重点验证：

- 输入数据是否正确
- 模块接口是否匹配
- 参数是否正确传递
- 中间状态是否合理


---

# Step 4 — Root Cause 分析

建立完整因果链：


问题现象
→ 直接原因
→ 根本原因


确保：

- 根本原因能够解释全部异常
- 不只是表面问题


---

# Step 5 — 最小修复方案

提出修复方案：

要求：

- 修改范围最小
- 不影响无关模块
- 不改变系统架构


若可安全修复，可提供 patch。


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