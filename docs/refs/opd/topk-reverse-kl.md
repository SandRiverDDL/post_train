# TopK Truncated Reverse-KL OPD

## 来源

- 论文：`Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes`
- arXiv：`2603.25562`
- 本地详细笔记：`docs/refs/revisiting-opd/REVIEW.md`

## 方法

这条路线把 OPD 从 sampled-token reward 改成 teacher topK 局部分布蒸馏。

```text
S_t = TopK_teacher(c_t)
q_hat = normalize pi_T over S_t
p_hat = normalize pi_student over S_t
loss = KL(p_hat || q_hat)
```

关键点：

- 必须在 topK support 内重新归一化。
- 通常用 teacher topK，而不是 student topK。
- 更像直接 loss，不是把 token reward 塞进 advantage。
- 需要 special-token mask、truncation mask 和 top-p rollout。

## 和 THUNLP topK reward 的区别

THUNLP topK reward 是用 topK 集合构造 3D token-level reward/advantage；topK truncated reverse-KL 是直接让 student 在 teacher topK support 上拟合 teacher 分布。

前者按源码会保留 `[B,T,K]` 的 advantage，并在 PPO loss 中对 topK token 的 logprob 逐项更新；后者通常是一个直接 supervised/distillation loss，不走 PPO ratio、old logprob 和 clipping。

因此二者都可能使用 topK 和 KL 形状的 logprob 差异，但优化路径不同：

```text
THUNLP topK reward:
  A_{t,k} = (log p_T - log p_S) * weight
  loss = PPO/PG(log p_S_current(v_{t,k}), old_logp, A_{t,k})

topK truncated reverse-KL loss:
  loss = sum_{v in support} p_student_hat(v) *
         (log p_student_hat(v) - log p_teacher_hat(v))
```

## 原论文实验要点

论文报告 sampled-token OPD 有三类失败模式：token-level supervision imbalance、teacher 在 student drift prefix 上不可靠、tokenizer/special-token mismatch。修复组合是 teacher topK support、top-p rollout 和 special-token masking。

本项目如果迁移，建议先 K=16，再视显存和稳定性试 K=32。
