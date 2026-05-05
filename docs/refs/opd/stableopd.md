# StableOPD / 长度稳定化

## 来源

- 论文：`Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models`
- arXiv：`2604.08527`

## 方法关注点

StableOPD 关注 OPD 中的长度膨胀：训练推进后，student rollout 会突然变长，截断样本占比上升，重复饱和，并导致验证性能下降。

论文提出的稳定化方向包括：

- reference-based divergence constraint
- rollout mixture distillation
- 对 truncation collapse 的显式监控

## 对本项目的意义

这和本项目 Lightning OPD 观察高度一致：后期 ckpt 变长、no-box、重复推理，8192 也不能明显救回。

因此无论采用 THUNLP topK reward 还是 topK reverse-KL，都应该保留：

- 截断样本整条 mask
- 输出长度 p90/p95 监控
- 重复 boxed / `Final Answer` / `wait` 统计
- 必要时加入 SFT/reference anchor 或 rollout mixture

