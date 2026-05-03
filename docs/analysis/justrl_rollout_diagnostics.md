# JustRL Rollout 诊断

## 当前结论

- JustRL / Nemotron / Qwen chat reasoning 模型做 rollout 时必须套 tokenizer chat template。
- 裸 prompt 产物不能用于判断 `<think>` 稳定性、`</think>` 闭合率、summary 边界或 boxed 风格。
- JustRL-Nemotron chat-template 轨迹仍明显偏长，raw CoT 不适合直接 SFT。
- `</think>` 后 summary 更接近可用 clean-solution，但仍需 boxed/correct/长度/重复答案过滤。
- 如果目标是压缩 CoT，应优先看 ConPress 路线；如果目标是 clean SFT，应优先提取 summary-only。

## 有效数据口径

当前有效 JustRL-Nemotron chat-template probe：

- query pool：`data/rollout/math_sft/justrl_nemotron_chat_probe50_len8192/query_pool.jsonl`
- raw：`data/rollout/math_sft/justrl_nemotron_chat_probe50_len8192/raw/raw_samples.merged.jsonl`
- `use_chat_template=true`
- `system_prompt=None`
- `assistant_prefill=None`

裸 prompt probe：

- `data/rollout/math_sft/justrl_nemotron_probe50_len8192/raw/raw_samples.merged.jsonl`
- 只作为错误采样记录，不再进入有效统计。

## JustRL-Nemotron Chat 50 题统计

| 指标 | 数值 |
|---|---:|
| response 数 | 50 |
| raw correct | 31/50 |
| parse_ok | 42/50 |
| boxed | 42/50 |
| `<think>` | 50/50 |
| `</think>` | 42/50 |
| output words mean | 2805.5 |
| output words min / p50 / p75 | 924 / 2264.5 / 3949.5 |
| output words p90 / p95 / max | 4634.8 / 5071.5 / 6099 |
| summary words mean | 230.5 |
| summary words min / p50 / p75 | 17 / 241.5 / 265.2 |
| summary words p90 / p95 / max | 320.3 / 327.9 / 482 |

离线复核：

- 归一化 `\dfrac -> \frac`、last/any boxed、无序逗号列表后，等价正确约 `40-41/50`。
- `</think>` 后 summary 内有正确 boxed 的样本约 `38/50`。
- 加 `output_tokens < 4000` 后约剩 `34/50` 可用候选。
- 8 条未闭合样本全部 `parse_ok=false` 且无 boxed，集中在 Level 5、组合、几何、图形/读图、复杂代数。

## 主要问题

- 输出长：mean `2805.5` words，p95 `5071.5`。
- 冗余强：大量 `Wait`、`Let me check`、verify、alternative approach。
- `double_boxed` 普遍：`<think>` 内 boxed 后，summary 又 boxed。
- 简单题也会过度解释，容易把 student 拉成长输出风格。
- 长题和图形/几何题容易不闭合或无 boxed。
- 当前正式 parser 已改为读取最后一个 boxed，但这仍不能证明轨迹完整；必须同时检查闭合、summary、多 boxed 与截断。

## SFT 使用建议

不建议：

- 不要直接使用 raw `<think>` 轨迹做 SFT。
- 不要使用裸 prompt 产物做任何格式稳定性结论。

summary-only 候选规则：

- 必须来自 chat-template rollout。
- 必须同时有 `<think>` 与 `</think>`。
- `</think>` 后 summary 非空。
- summary 内必须有 `\boxed{...}`。
- last/any boxed 与标准答案等价。
- 首版建议 `output_tokens < 4000`。
- 过滤多个互相矛盾 boxed、明显读题漂移、summary 继续长篇验证、未闭合、无 boxed。

raw CoT 候选规则：

- 仅作为对照或 OPD/teacher signal，不作为默认 imitation target。
- 若必须用，需去重尾部重复答案、截断 verification、确保最后答案等价。

## 与 ConPress 的关系

- 单题 JustRL-Nemotron chat 输出太长；ConPress N=3 能把 pack/single ratio 压到 `0.459`，但稳定性不足。
- 当前 ConPress 最稳配置为 `exclude_visual + balanced_level + default natural prompt`，详见 `docs/analysis/conpress_probe_diagnostics.md`。
- ConPress 产物是“压缩 CoT”，不是 summary-only clean solution；两条路线应分开评估。

## 历史 DeepSeek 口径

旧 JustRL-DeepSeek 结果保留为历史参考：

- `justrl_deepseek_1500/raw_samples.jsonl`：1500 条，平均约 `1013.6` tokens，`</think>` 闭合率 `36.1%`，无 boxed `59.4%`。
- `justrl_deepseek_math_all_len3650/raw/raw_samples.merged.jsonl`：7498 条，平均约 `1442.1` tokens，`</think>` 闭合率 `64.5%`，无 boxed `31.9%`。
- `train.rollout_rejection_lt2000_matched321.jsonl`：321 条，boxed `100%`，但 double_boxed 约 `90.3%`。

这些历史数据不改变当前结论：raw reasoning 轨迹必须清洗，不能直接 SFT。
