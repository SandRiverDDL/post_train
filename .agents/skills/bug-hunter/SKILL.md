---
name: bug-hunter
description: 当系统行为异常、程序报错或结果不符合预期时使用。用于定位并修复影响系统正确性的 BUG。
---

# Bug Hunter：系统 BUG 定位

## 目标

定位并修复影响 **系统正确性** 的问题。

优先级：

1. 行为正确性
2. 算法逻辑正确性
3. 数据处理正确性
4. evaluation / metric 正确性


避免：

- 代码风格审查
- 不必要重构
- 架构 redesign


本 skill 的核心任务是：

**Root Cause Analysis + Minimal Fix**


---

# 调试原则

**Evidence First**

所有结论必须基于：

- 代码
- 日志
- stack trace
- 数据
- 实际运行结果


**区分直接原因与根本原因**


问题现象
→ 直接原因
→ 根本原因



**最小修改原则**

只修改解决问题所必需的代码。


---

# 调试 Checklist

理解问题：

- 期望行为
- 实际行为
- 错误日志
- 最近代码改动


定位问题层级：


输入
→ 数据处理
→ 核心逻辑
→ 模型 / 算法
→ 输出
→ evaluation



验证关键点：

- 输入数据格式
- 参数是否正确传递
- 中间变量是否合理
- 接口是否匹配


---

# Root Cause 分析

建立因果链：


问题现象
→ 直接原因
→ 根本原因


确保根因能够解释全部异常。


---

# 修复原则

优先提供 **Fix Proposal**。

只有在满足以下条件时才直接 patch：

- Root Cause 已确认
- 修改范围局部
- 修改风险低


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