# OPD 参考笔记

## 目录定位

本目录只放 OPD 相关论文/实现的参考笔记，不直接替代 `docs/STATE.md` 的当前状态，也不把一次性实验命令写进 refs。

当前结构：

- `thunlp-rethinking-opd.md`：THUNLP OPD 论文与 repo 口径，重点是 topK token reward、冷启动和 prompt selection。
- `thinking-machines-opd.md`：Thinking Machines / verl 标准 sampled-token OPD。
- `topk-reverse-kl.md`：Revisiting OPD 的 teacher topK truncated reverse-KL。
- `stableopd.md`：长度膨胀与稳定化补丁。
- `SUMMARY.md`：各方法优劣、互补关系和推荐优先级。
- `MIGRATION.md`：迁移到本项目时应该怎么落地。
- `SOURCES.md`：来源链接。

## 当前总判断

OPD 不是单一方法名，而是一类“student rollout + teacher dense token signal”的路线。当前最不建议继续扩大的是 **Lightning OPD top1 sampled-token**；更值得做的是：

- 先用 Thinking Machines / verl k1 做 smoke，验证在线链路。
- 主力参考 THUNLP OPD 的 topK token reward 和失败恢复策略。
- 若要做更“分布蒸馏”的版本，再参考 Revisiting OPD 的 topK truncated reverse-KL。
- 所有 OPD 实验都必须把长度、截断、boxed/parse 和重复率作为一等指标。

当前本项目的 k1 baseline 小测试已收敛为 MATH train Level 4、prompt token `<=256`、`500` prompts、`max_response_length=2048`、student 两卡 + teacher 一卡；细节见 `MIGRATION.md`。
