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

- 先支持 `loss_mode: dft`，不引入 reference model。
- DFT 使用当前 forward 的 gold-token probability 做 stop-gradient 权重。
- 先不实现 ASFT full KL，避免在 3090 上引入额外 reference forward 和 full-vocab KL 显存风险。

后续如果实现 ASFT：

- 增加 `asft_kl_weight`。
- LoRA 下优先用 adapter-disabled logits 作为 `pi_base`。
- 非 LoRA 下需要显式加载 frozen reference model，或先不支持。

## 和 OPD / topK KD 的区别

ASFT 不需要 teacher logits，也不需要 on-policy rollout。它只依赖当前训练数据和 base/reference model。

区别：

| 方法 | 额外模型 | 额外数据 | 学习对象 |
|---|---|---|---|
| ASFT | base/reference | 无 | demonstration token + base 分布约束 |
| Lightning OPD | teacher | student rollout + teacher logprob | teacher 对 student trajectory 的偏好 |
| topK KD | teacher | teacher topK logits | teacher 分布 |

因此 ASFT 更适合作为 SFT 目标替换；OPD/KD 更像 teacher 分布吸收。
