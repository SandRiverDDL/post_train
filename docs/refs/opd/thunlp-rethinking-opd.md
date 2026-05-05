# THUNLP OPD：Rethinking On-Policy Distillation

## 来源

- 论文：`Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe`
- arXiv：`2604.13016`
- Repo：`https://github.com/thunlp/OPD`

## 核心问题

THUNLP 关注的不是“OPD 是否能跑”，而是“为什么有些 OPD 成功、有些失败”。公开 README 给出的两条必要条件是：

1. student 和 teacher 要有 compatible thinking patterns。
2. teacher 要提供 student 训练数据之外的新能力；分数更高但能力分布不互补，也可能没有有效信号。

它还强调，成功 OPD 的 token-level 机制不是全 vocab 对齐，而是在 student 访问到的状态上，对齐少数高概率 token。这个共享 token 集很小，但能覆盖大部分概率质量。

## THUNLP-style topK reward 到底是什么

它不是 `topK truncated reverse-KL`，也不是普通 teacher CE。按开源代码看，它是 **用 topK 局部 token 集合构造 3D token-level reward/advantage**。

公开 repo 的 OPD 配置是：

```text
ADV_ESTIMATOR=token_reward_direct
LOG_PROB_TOP_K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
N_RESPONSES=4
```

对应源码位置：

- `verl/workers/actor/dp_actor.py::compute_distillation_reward`
- `verl/trainer/ppo/core_algos.py::compute_token_reward_direct_advantage`
- `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla`

如果 `LOG_PROB_TOP_K=0`，就退化成 sampled-token OPD：只看 student 实际采样出的 token。

如果 `LOG_PROB_TOP_K=16`，每个位置不只看 sampled token，而是构造一个局部 token 集：

```text
prefix c_t = prompt + response_<t
V_t = selected_topK(c_t)
```

`TOP_K_STRATEGY` 决定 `S_t` 来自哪里：

| strategy | token 集含义 |
|---|---|
| `only_stu` | 取 student topK token，再查询 teacher 对这些 token 的 logprob |
| `only_tch` | 取 teacher topK token |
| `intersection` | 只保留 student topK 与 teacher topK 交集 |
| `union` | 合并 student topK 与 teacher topK |
| `union-intersection` | 对称差，即只保留只在一边 topK 出现的 token |

`REWARD_WEIGHT_MODE` 决定这些 token 怎么加权：

| mode | 直觉 |
|---|---|
| `student_p` | 按 student 概率加权，默认值，更关注 student 当前认为可能的 token |
| `teacher_p` | 按 teacher 概率加权，更强地跟 teacher 偏好走 |
| `none` | 不按概率加权 |

默认 `only_stu + student_p` 的源码等价形式是：

```text
V_t = TopK_student(c_t)
w_{t,k} = softmax_K(log p_S(v_{t,k} | c_t))
A_{t,k} = -(log p_S(v_{t,k} | c_t) - log p_T(v_{t,k} | c_t)) * w_{t,k}
        =  (log p_T(v_{t,k} | c_t) - log p_S(v_{t,k} | c_t)) * w_{t,k}
```

这里 `A` 不是先 sum 成 `[B,T]` 的标量 reward，而是保留为 `[B,T,K]`：

```text
rm_scores.shape = [batch, response_len, K]
advantages = rm_scores * response_mask[..., None]
```

训练时如果 `advantages.dim() == 3`，actor 会重新前向得到这些 topK token 的当前 `log_prob`，再把 3D `old_log_prob / log_prob / advantages` 送进 PPO loss。`compute_policy_loss_vanilla` 对 K 维逐项算 ratio 和 clipping，最后对 K 维求和。

所以 THUNLP-style topK reward 的关键不是“给 sampled token 一个标量 reward”，而是：**把 topK token 都变成 policy-gradient 风格的局部动作候选来更新**。

各 strategy 的源码语义：

| strategy | reward support | 公式骨架 |
|---|---|---|
| `only_stu` | student topK | `-(S_logp - T_on_S) * weight` |
| `only_tch` | teacher topK | `-(S_on_T - T_logp) * weight` |
| `intersection` | student topK 与 teacher topK 交集 | invalid token mask 后同 `only_stu` |
| `union` | student topK + teacher topK 去重 | union support 上同一公式 |
| `union-intersection` | 对称差 | union support 上用未归一化概率权重 |

## 和 topK truncated reverse-KL 的区别

| 项 | THUNLP-style topK reward | topK truncated reverse-KL |
|---|---|---|
| 训练形态 | 3D token reward / advantage + PPO/PG loss | 直接分布 loss |
| token 集 | student/teacher/intersection/union 可选 | 通常 teacher topK |
| 默认权重 | `student_p` | support 内 softmax 归一化 |
| 更新对象 | policy gradient 风格更新 topK token logprob | 让 student topK 分布贴 teacher topK 分布 |
| 优点 | 更贴近 verl/RL 框架，容易做在线 OPD | 分布语义更干净，减少 sampled-token 噪声 |
| 风险 | reward 设计、符号和归一化敏感 | 需要实现 topK gather 和局部 KL，可能要改 loss |

## 原论文/Repo 实验口径

公开 README 中的默认训练设置：

- verl `v0.7.0`
- SFT 使用 LlamaFactory `v0.9.5`
- 8 x NVIDIA A800 80GB
- `N_RESPONSES=4`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESP_LENGTH=7168`
- `MAX_VAL_RESP_LENGTH=31744`
- `LOG_PROB_TOP_K=16`
- `TOP_K_STRATEGY=only_stu`
- `REWARD_WEIGHT_MODE=student_p`
- non-thinking Qwen3 需要显式 `enable_thinking=False`

README 中还提到：

- 用 DAPO-Math-17K 作为一个默认训练数据入口。
- 用 teacher rollout 生成 SFT 数据。
- 用 GRPO 作为 RL 对照时，`ADV_ESTIMATOR=grpo` 且 `LOG_PROB_TOP_K=0`。
- 可用 DeepMath 去重避免和 DAPO-Math-17K 数据泄漏。

## 对本项目的意义

THUNLP-style topK reward 比当前 Lightning top1 更值得试，因为它直接减少 sampled-token 单点信号的偶然性。

但它仍然不是“保证稳定”的分布蒸馏：

- 如果 teacher/student thinking pattern 不兼容，仍会失败。
- 如果 teacher 没给 student 新能力，topK reward 可能只是噪声。
- 如果输出持续变长，仍要引入长度稳定补丁或截断 mask。

本项目若实现，第一版应优先复刻：

```text
K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
N_RESPONSES=2 或 4
max_response_length=2048
```
