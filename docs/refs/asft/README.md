# ASFT：Anchored Supervised Fine-Tuning

## 定位

ASFT 是 ICLR 2026 的 `Anchored Supervised Fine-Tuning`。用户口头说的 `AST`，公开论文与官方仓库对应名称应为 `ASFT`。

一句话结论：ASFT 是在 DFT 上加一个 base/reference model 的 KL anchor，用来保留 DFT 的 token 概率重加权优势，同时抑制训练过程中向少数高概率轨迹漂移。

## 核心思想

DFT 将 SFT token loss 乘上当前模型对目标 token 的概率：

```text
L_DFT = - stopgrad(p_theta(y_t | x, y_<t)) * log p_theta(y_t | x, y_<t)
```

这会降低低概率 token 的爆炸梯度，强化模型已经较能解释的 token。论文用 RWR 框架解释 DFT：它相当于选择了更紧的辅助分布，因此比普通 SFT 更接近 RL 目标下界。

问题是 DFT 会自我强化：训练越久，权重越集中到当前模型高概率的样本/轨迹上，导致 distributional drift、有效样本数下降和任务不稳定。

ASFT 的补丁很小：

```text
L_ASFT = L_DFT + lambda * E_s[ KL(pi_base(.|s) || pi_theta(.|s)) ]
```

这里 `pi_base` 是固定 reference，一般是训练起点模型。论文强调使用 forward KL：`KL(base || current)`，目的是 mode-covering，防止模型过度坍缩到少数高概率模式。

## 对本项目的意义

ASFT 不是生成短 CoT 的方法，也不解决 teacher rollout 噪声；它是一个训练目标。它更适合放在已经清洗好的 SFT 数据上，作为 SFT/DFT 的更稳替代。

在当前仓库里，它最值得作为三个小实验之一：

1. `SFT`：普通 CE baseline。
2. `DFT`：目标 token 概率重加权。
3. `ASFT`：DFT + base KL anchor。

建议先用同一批短 clean solution 数据测试，不要直接喂 JustRL/Nemotron raw 长 CoT。ASFT 的 anchor 只能抑制训练漂移，不能修复错误答案、截断、串题和多 boxed。

## 推荐阅读顺序

- `METHOD.md`：公式、loss 口径和实现要点。
- `EXPERIMENTS.md`：论文实验、超参和对本仓库的可复现实验建议。
- `SOURCES.md`：来源链接与出处。
