---
name: llm-bug-hunter
description: 当 LLM 训练或 evaluation 出现异常时使用。适用于 SFT、RL、GRPO 等后训练 pipeline 的 BUG 定位。
---

# LLM Bug Hunter：后训练调试

## 目标

定位并修复 **LLM 后训练 pipeline** 的正确性问题。


适用场景：

- SFT 训练异常
- RL / GRPO 训练异常
- rollout 行为异常
- reward 计算错误
- evaluation 指标异常


优先级：

1. pipeline 正确性
2. 训练信号正确性
3. evaluation 正确性


---

# LLM Pipeline

典型 pipeline：


dataset
→ tokenizer
→ prompt construction
→ rollout / generation
→ reward / verifier
→ trainer
→ evaluation


调试必须逐层验证。


---

# 调试 Checklist

Dataset

- dataset split 是否正确
- prompt / response 字段
- answer 格式


Tokenizer

- tokenizer 是否匹配模型
- special tokens
- sequence truncation


Rollout

- rollout 数量
- generation length
- sampling 参数


Reward

- reward parser
- reward scaling
- reward 是否恒定


Trainer

- loss 是否合理
- gradient 是否更新
- optimizer step 是否生效


Evaluation

- dataset 是否正确
- metric 是否正确
- answer parser


---

# 关键一致性检查

必须验证：


reward logic == evaluation logic


否则训练目标与评测目标不一致。


---

# Root Cause 分析

建立因果链：


异常指标
→ 直接原因
→ 根本原因



---

# 输出格式

Findings

Issue 1

Problem:
...

Evidence:
...

Root Cause:
...

Fix Proposal:
...

Risk:
...