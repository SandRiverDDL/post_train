# Revisiting OPD Review

## 一句话结论

这篇论文最值得吸收的是：标准 sampled-token OPD 在长推理里太脆，应该从“只看学生采样到的那个 token”改成“在 teacher top-K 局部支持集上做截断 reverse-KL”。这直接支持本仓库后续实现 `teacher_top_k=16/32` 的 sparse OPD，而不是继续只存单 token logprob。

## 概念映射

| 角色/张量 | 论文含义 | 本仓库可能对应 | 备注 |
|---|---|---|---|
| student / `pi_theta` | 当前训练策略 | SFT checkpoint + LoRA student | 在线 rollout，训练时更新 |
| teacher / `q` | 提供局部分布监督的强模型 | Nemo / JustRL / OpenMath teacher | 最好和 student tokenizer/template 尽量接近 |
| rollout response | student 生成的 on-policy 轨迹 | vLLM rollout `response` | 不是 teacher 固定轨迹 |
| sampled-token OPD | 只比较 sampled token 的 teacher/student logprob | 旧 OPD / 单 logprob 监督 | 论文认为长链路下不稳定 |
| teacher top-K support | teacher 在每个 prefix 下概率最高的 K 个 token | `teacher_topk_token_ids` | 论文默认 K=32；本仓库可先试 16 |
| truncated reverse-KL | 在 top-K 支持集上归一化后做 KL | OPD loss 新主项 | 必须做 support 内 renormalization |
| top-p rollout | 控制 rollout 不离 teacher 支持太远 | vLLM `top_p=0.9` | 论文认为 top-K 需要和 top-p 配合 |
| special-token masking | 屏蔽 tokenizer/template mismatch 的 token | `<think>`、EOS、chat special tokens mask | 对跨模型 teacher 尤其重要 |

## 论文诊断

论文把 sampled-token OPD 的问题拆成三类：

1. 单 token 信号不平衡：多数 sampled token 的 teacher-student logprob gap 可能是负的，训练容易被少量局部正信号、filler token 或短续写牵着走。
2. teacher 在 student drift prefix 上不可靠：student 长链路生成后，prefix 可能已经偏离 teacher 常见分布，此时 teacher 对下一 token 的高概率不再等价于整条轨迹质量高。
3. tokenizer / special-token mismatch 会制造假负样本：不同 tokenizer 或 chat template 会让语义等价的输出在 token 级 logprob 上被错误惩罚。

理论侧重点不是“sequence-level OPD 一定更好”，而是相反：sequence-level reverse-KL 更接近原目标但方差随长度爆炸；token-level OPD 有偏但方差更可控。因此工程上应保留 token-level 局部更新，但把单 token 监督扩展到局部 top-K 分布监督。

## 核心算法

对每个 prompt，用当前 student 生成一组 responses。对 response 中每个 token 位置，设 prefix 为 `c_t`。

1. teacher 前向，取 `S(c_t)=TopK_teacher(c_t)`。
2. student 前向，只 gather 这些 teacher top-K token 的 logits。
3. 在 `S(c_t)` 内分别对 teacher logits 和 student logits 做 softmax，得到归一化分布。
4. 最小化截断 reverse-KL：

```text
KL(student_hat || teacher_hat)
= sum_{v in S(c_t)} student_hat(v) * [log student_hat(v) - log teacher_hat(v)]
```

5. 对 response mask、truncation mask 和 special-token mask 做联合过滤。

重要细节：

- 不是普通 forward-KL teacher-to-student；论文公式是 reverse-KL 的局部截断近似。
- renormalization 不能省，否则 support 内概率质量不可比，论文消融里会崩。
- 只做 teacher top-K 不够，还要控制 rollout sampling；论文用 `top_p=0.9`。
- K 不是越大越好；论文默认 32，消融里 16/32/48 都可跑，但太小会不稳。

## 复现路线

优先不要完整复现论文的 7B + 8H100 设置，而是做本仓库可控的小复现。

### V0：离线 top-K 数据检查

目标：验证 schema 和 loss 对齐，不训练大模型。

1. 从当前 candidate prompt 池抽 32-128 条。
2. 用 student checkpoint 生成 rollout，`top_p=0.9`、`temperature=1.0`。
3. 用 teacher 前向保存：
   - `input_ids`
   - `response_mask`
   - `teacher_topk_token_ids`
   - `teacher_topk_logits` 或 `teacher_topk_logprobs`
   - `special_token_mask`
4. 同一 batch 用 student gather top-K logits，检查 shape 与 mask。
5. 计算一版 `KL(student_hat || teacher_hat)`，确认 loss finite。

### V1：小训练

目标：比较 sampled-token / top-K OPD 是否稳定。

默认参数建议：

```yaml
teacher_top_k: 16
rollout_top_p: 0.9
rollout_temperature: 1.0
max_response_length: 2048
loss_type: topk_reverse_kl
kl_temperature: 1.0
learning_rate: 2e-6
warmup_steps: 0
max_steps: 100-300
```

对照组：

- SFT baseline checkpoint
- sampled-token OPD 或当前 Lightning OPD 单 logprob 版本
- teacher top-K reverse-KL，K=16
- 可选 K=32

### V2：接入当前 Lightning OPD 数据管线

当前 Lightning OPD 已有 teacher top-K 数据雏形，可以吸收这篇论文的 loss：

- 如果训练数据已经离线保存 teacher top-K，则先做 offline sparse OPD。
- 如果想严格 on-policy，应在训练中周期性重新 rollout，而不是固定老 rollout。
- 第一版可以接受“offline top-K OPD”作为工程近似，但报告里不能叫完整 on-policy 复现。

## 和 Lightning OPD 的关系

两篇论文关注点不同：

| 方向 | Lightning OPD | Revisiting OPD |
|---|---|---|
| 核心问题 | teacher logprob 在线计算太贵 | sampled-token OPD 信号太脆 |
| 数据 | 固定 offline rollout | student on-policy rollout |
| teacher 信息 | fixed response 上的 teacher logprob | 每个 prefix 的 teacher top-K 分布 |
| 主要收益 | 省 teacher server / 加速 | 稳定训练 / 减少单 token 噪声 |
| 本仓库吸收方式 | 离线预计算 teacher 信息 | top-K sparse reverse-KL loss |

组合方式：

- 短期：Lightning OPD 的离线数据管线 + Revisiting OPD 的 top-K reverse-KL loss。
- 中期：定期刷新 student rollout，让数据更接近 on-policy。
- 长期：如果迁移 verl，再做真正 online top-K OPD。

## 对本仓库的建议

1. 不要再只存 `teacher_token_logprobs`；至少存 `teacher_topk_token_ids + teacher_topk_logprobs`。
2. 当前计划用 `K=16` 是合理低成本版本；如果显存和存储允许，再跑 `K=32`。
3. loss 口径应命名为 `topk_reverse_kl`，不要叫普通 KD。
4. 必须显式记录：
   - rollout sampling 参数
   - top-K 大小
   - 是否 support renormalization
   - special-token mask 规则
   - teacher/student tokenizer 是否一致
5. 评测时重点看：
   - `boxed_rate`
   - `parse_success_rate`
   - 平均输出 token
   - 截断率
   - MATH500 / GSM8K

## 风险

- teacher 与 student tokenizer 不一致时，top-K token id 不能直接跨 tokenizer 使用；必须保证训练时 gather 的 token id 属于 student vocab。若 teacher tokenizer 不同，只能退回共享文本对齐或换同 tokenizer teacher。
- 如果 rollout top-p 太宽，student 会进入 teacher 不熟悉的 prefix，top-K 监督仍可能误导。
- 如果 K 太小，local support 信息不足；如果 K 太大，存储和 gather 开销上升。
- 论文的主实验是 7B、16K response、H100 环境；本仓库 1.5B/3090 只能先吸收机制，不能按绝对分数复现。
- 本仓库 ConPress 实验里，Qwen3-4B Non-Thinking teacher 在剩余 837 个 hard-tail 题上的单题正常 prompt rollout 平均 `5777.7` tokens，约 `49%` 打满 `8192`，correct 只有 `14.22%`；这说明同 tokenizer teacher 也可能因为长度和风格分布错配而给 sampled-token OPD 提供负信号。

## Smoke Test 建议

1. 构造 2 条短样本，手动验证 top-K shape：
   - `teacher_topk_token_ids.shape == [response_tokens, K]`
   - `teacher_topk_logprobs.shape == [response_tokens, K]`
2. 对同一 batch 计算 student gather logits，确认无 OOV、无 shape mismatch。
3. 检查 support 内 softmax 后每行和为 1。
4. 加 special-token mask 后，loss token 数仍大于 0。
5. 跑 1 step trainer，确认 loss finite、grad norm finite。
6. 跑 100 step 小训练，对比 SFT baseline，观察 boxed rate 不应明显掉。
