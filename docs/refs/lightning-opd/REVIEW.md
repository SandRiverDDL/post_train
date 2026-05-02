# Lightning OPD Review

## 一句话结论

Lightning OPD 将 OPD 的 teacher logprob 离线预计算，训练时只对 student 做前向，从而避免 live teacher server。它适合先做成独立 offline OPD trainer 或 verl 的独立 mode；不建议直接混进当前 online EXOPD 分支。

## 概念映射

| 角色/张量 | 论文含义 | 本仓库可能对应 | 备注 |
|---|---|---|---|
| student / `pi_theta` | 当前训练策略 | actor / LoRA student | 训练时更新 |
| SFT reference / `pi_ref` | SFT 后参考策略，用于离线采样 rollouts | merged SFT checkpoint | Lightning OPD 的 rollout 来源 |
| teacher / `pi_T` | 为 SFT 与 OPD 提供监督的同一个 teacher | JustRL 或其他 teacher | teacher consistency 是核心前提 |
| rollout response | 从 `pi_ref` 离线生成的固定响应 | parquet 中预存 `responses` | 训练时不再在线 rollout |
| `teacher_log_probs` | teacher 对固定 response token 的逐 token logprob | parquet 新字段 | 训练时直接读取 |
| `student_log_probs` | 当前 student 对同一 token 的 logprob | actor forward | 训练时在线计算 |
| verifier/reward | 评测或过滤信号 | boxed reward 可用于 eval | Lightning OPD 核心不是 verifier reward |
| base/reference correction | G-OPD/EXOPD 的 base 修正 | `base_log_prob` | 第一版不建议混用 |

## 训练步骤

1. 离线阶段：用 SFT reference (`pi_ref`) 对 prompts 生成 responses。
2. 离线阶段：用 teacher 对这些 fixed responses 计算逐 token `teacher_log_probs`。
3. 保存训练数据：prompt、response、response mask、teacher logprobs，以及必要的 tokenizer/template 元信息。
4. 训练阶段：读取固定 response，不再调用 online rollout。
5. 训练阶段：只用当前 student 对固定 response 计算 `student_log_probs`。
6. 用 teacher/student logprob 差构造 token-level OPD objective，更新 student。

近似伪代码：

```python
batch = read_offline_batch()
student_log_probs = actor.log_prob(batch.prompt, batch.response)
reverse_kl = student_log_probs - batch.teacher_log_probs
advantages = -reverse_kl
loss = policy_loss(student_log_probs, advantages, batch.response_mask)
```

## 关键公式

Lightning OPD 的核心 token 监督可以写成：

```text
A_t(theta) = log pi_T(a_t | s_t) - log pi_theta(a_t | s_t)
```

与当前 online EXOPD 最大区别：

```text
online EXOPD:
  student 在线 rollout
  teacher/ref 在线算 logprob
  base 在线算 logprob
  lambda extrapolation

Lightning OPD:
  SFT reference 离线 rollout
  teacher logprob 离线预计算
  训练时只算 student logprob
```

因此第一版不应默认启用 EXOPD 的 `base_log_prob` / `lambda_vals` 路径。

## 效果与证据

论文摘要报告：

- 标准 OPD 需要训练期间 live teacher inference server，基础设施成本高。
- 朴素 offline OPD 不稳定，关键原因是 teacher consistency。
- Lightning OPD 强制 SFT 阶段与 OPD 阶段使用同一个 teacher，并离线预计算 teacher logprob。
- Qwen3-8B-Base 起点上，AIME 2024 达到 69.9%，约 30 GPU hours；相对标准 OPD 约 4.0x 加速。

这些结论来自论文摘要与 arXiv 页面；实现前仍应读完整 PDF 的算法和实验设置。

## 不足和风险

- 当前 merged SFT checkpoint 是否由 JustRL teacher 数据训练得到尚不确定；若不是，严格 teacher consistency 不成立。
- 最大工程风险是 token 对齐：`teacher_log_probs` 必须和 response token、response mask 严格对齐。
- 如果 teacher 和 student tokenizer/chat template 不一致，离线 logprob 保存和训练读取会复杂很多。
- Lightning OPD 固定 rollout，不再是真正 online rollout；直接复用当前 `generate_sequences -> compute_ref_log_prob -> update` 流程会增加不必要复杂度。
- 与 EXOPD/G-OPD 的 base correction 混合需要重新定义语义；第一版不建议混用。

## 实现建议

优先做独立 offline OPD MVP，而不是直接深改 online EXOPD：

1. 新增 `scripts/prepare_lightning_opd_data.py`：
   - 用 SFT reference 生成 responses。
   - 用 teacher 计算 per-token `teacher_log_probs`。
   - 保存 parquet。
2. 新增 offline trainer 或 verl 独立 mode：
   - 读取固定 response。
   - 跳过 rollout generation。
   - 跳过 teacher/ref worker。
   - 只算 student logprob 并更新 LoRA。
3. 第一版只支持 single-teacher，不支持 EXOPD base correction。

若接入 verl，预计改动：

- `ray_trainer.py`：新增 `algorithm.lightning_opd.enabled`，跳过 rollout/ref forward。
- `dp_actor.py`：新增 offline OPD loss 或复用 `compute_opd_reverse_kl` 的标准 OPD部分。
- 数据读取/准备：支持 `responses`、`response_mask`、`teacher_log_probs`。
- YAML：新增 lightning OPD experiment config。
- tests：覆盖 schema、shape 对齐、trainer 跳过 ref worker、loss 公式。

改动量估计：

- MVP：约 300-600 行。
- 稳健可维护版本：约 700-1200 行。
- 完整 recipe 加实验记录：约 1200-2000 行。

## Smoke Test 建议

1. 生成 10 条 offline rollouts。
2. 验证 `responses`、`response_mask`、`teacher_log_probs` shape 一致。
3. 对同一 batch 比较 online teacher logprob 与离线保存值。
4. 跑 1 step offline OPD，确认不加载 teacher/ref worker。
5. 跑 100/1000 样本小训练，观察 loss、长度分布和 boxed exact match。
