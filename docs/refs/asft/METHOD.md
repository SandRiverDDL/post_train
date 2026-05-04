# ASFT 方法笔记

## 背景：SFT 与 DFT

普通 SFT 对 demonstration token 做最大似然：

```text
L_SFT = - log p_theta(y_t | x, y_<t)
```

DFT 在 token 级别引入动态权重：

```text
weight_t = stopgrad(p_theta(y_t | x, y_<t))
L_DFT = weight_t * L_SFT
```

直觉：

- 普通 SFT 对低概率 token 的梯度可能很大，容易把模型推向 memorization。
- DFT 让模型优先学习当前分布下更可信、更接近自身能力边界的 token。
- 但 DFT 会越来越偏向当前模型已经高概率的轨迹，导致数据覆盖变窄。

## ASFT 的修改

ASFT 在 DFT 上加 reference/base model 的 KL 约束：

```text
L_ASFT = L_DFT + lambda * KL(pi_base || pi_theta)
```

实现上通常只在 response label 有效位置计算：

```text
ce_t = CE(logits_theta[t], y_t)
target_prob_t = softmax(logits_theta[t])[y_t].detach()
dft_t = target_prob_t * ce_t

kl_t = sum_v softmax(logits_base[t])[v] *
       (log_softmax(logits_base[t])[v] - log_softmax(logits_theta[t])[v])

loss = mean_valid(dft_t + lambda * kl_t)
```

关键点：

- `target_prob_t` 必须 stop-gradient。
- `pi_base` 固定，不参与训练。
- KL 方向建议用 `KL(base || current)`，不是 `KL(current || base)`。
- `lambda` 控制 anchor 强度；过小等于 DFT，过大接近被 base 束缚。

## 为什么不是 SFT + KL

论文将 `SFT w/ KL` 作为 baseline。它只加 anchor，但没有 DFT 的动态重加权，因此不等价于 ASFT。

方法差异：

| 方法 | token CE | DFT 权重 | base KL |
|---|---:|---:|---:|
| SFT | yes | no | no |
| SFT w/ KL | yes | no | yes |
| DFT | yes | yes | no |
| ASFT | yes | yes | yes |

论文结果里，`SFT w/ KL` 经常不是最优，说明收益不是单纯来自“别偏离 base”，而是 DFT 的重加权和 KL anchor 的组合。

## 实现成本

全量微调时需要额外 reference forward，显存和算力成本会上升。LoRA 场景可以用 `disable_adapter()` 得到 base logits，避免加载第二份模型。

本仓库当前实现路线：

- 已支持 `loss_mode: dft`，使用当前 forward 的 gold-token probability 做 stop-gradient 权重。
- 已支持 `loss_mode: asft_topk`，使用 LoRA `disable_adapter()` 获取 base/reference topK，并计算 topK truncated `KL(base || current)`。
- 仍不实现 ASFT full KL，避免在 3090 上引入额外 reference forward 和 full-vocab KL 中间张量风险。

当前 `asft_topk` 口径：

- 默认 `asft_top_k=32`、`asft_kl_weight=0.03`。
- 先 no-grad base forward 取 topK，释放 base full logits 后再 current forward。
- KL 在 base topK support 内重新归一化，只近似 full-vocab forward KL。
- 非 LoRA / 非 `trl_peft` 路径暂不支持。

## 当前实测

Qwen3-1.7B BF16 LoRA、Mix-Long clean<2048、三卡 DDP、同一评测口径下：

| 方法 | best ckpt | MATH500 | GSM8K | 平均输出 tokens |
|---|---:|---:|---:|---|
| SFT | 100 | 0.644 | 0.7885 | 555.4 / 618.9 |
| DFT | 25 | 0.656 | 0.7923 | 329.5 / 208.7 |
| ASFT-topK | 50 | 0.680 | 0.8120 | 565.3 / 610.6 |

当前结论：ASFT-topK 是这组三者里最好的候选，MATH500 和 GSM8K 都有提升；但它没有继承 DFT 的短输出优势，且仍会出现尾部 `\boxed{}` 或 `Final Answer` 重复。因此 ASFT-topK 当前适合作为强一点的 stage1 对照，不应被描述成已经解决格式稳定性问题。

## 和 OPD / topK KD 的区别

ASFT 不需要 teacher logits，也不需要 on-policy rollout。它只依赖当前训练数据和 base/reference model。

区别：

| 方法 | 额外模型 | 额外数据 | 学习对象 |
|---|---|---|---|
| ASFT | base/reference | 无 | demonstration token + base 分布约束 |
| Lightning OPD | teacher | student rollout + teacher logprob | teacher 对 student trajectory 的偏好 |
| topK KD | teacher | teacher topK logits | teacher 分布 |

因此 ASFT 更适合作为 SFT 目标替换；OPD/KD 更像 teacher 分布吸收。
