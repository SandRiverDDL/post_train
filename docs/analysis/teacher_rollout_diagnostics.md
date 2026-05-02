# Teacher Rollout 诊断：JustRL 与 OpenMath-Nemotron

## 背景

本诊断用于记录 Lightning-OPD / SFT 蒸馏前对 teacher rollout 的观察，避免只依赖会话上下文。

相关本地产物：

- JustRL 全量采样：`data/rollout/math_sft/justrl_deepseek_math_all_len3650/raw/raw_samples.merged.jsonl`
- Nemotron raw probe：`data/rollout/math_sft/openmath_nemotron_len_probe_100/shards/shard0-of-1/`
- Nemotron chat probe：`data/rollout/math_sft/openmath_nemotron_chat_probe_100/raw_samples.jsonl`
- Nemotron chat prefill probe：`data/rollout/math_sft/openmath_nemotron_chat_prefill_done_probe_100/raw_samples.jsonl`
- 当前 OPD 训练集：`data/lightning_opd/nemotron_candidate/train.jsonl`
- 当前 OPD gap 诊断：`data/lightning_opd/nemotron_candidate/student_teacher_gap_report.json`

## 关键统计

JustRL 全量采样使用 `max_new_tokens=3650`：

- 样本数：`7498`
- tokenizer 输出长度均值约 `2756.6`
- `p50=3085`
- `p75/p90/p95/p99=3650`
- 命中 `3650` 截断上限：`2981/7498 = 39.76%`
- boxed 相关：`boxed=5104`，`parse_ok=5070`，`correct=4035`
- 无 boxed：`2394`

Nemotron raw 100 条 probe：

- tokenizer 输出长度均值约 `2289.99`
- `p50=2190`
- `p75/p90/p95/p99=3650`
- 命中 `3650` 截断上限：`29/100`

Nemotron chat template 100 条 probe：

- tokenizer 输出长度均值约 `2778.23`
- `p50=2923`
- `p75/p90/p95/p99=3650`
- 命中 `3650` 截断上限：`38/100`
- `<think>`：`100/100`
- no boxed：`35/100`
- 正确：`54/100`

Nemotron chat prefill `"<think> Okay, I think I have finished thinking. </think>"` 100 条 probe：

- generated-only tokenizer 输出长度均值约 `2817.9`
- `p50=3036`
- `p75/p90/p95/p99=3650`
- 命中 `3650` 截断上限：`34/100`
- 正确：`54/100`
- 去掉预填后，生成内容仍有 `<think>`：`2/100`
- 去掉预填后，生成内容仍有 `</think>`：`63/100`

当前 OPD 训练集 `data/lightning_opd/nemotron_candidate/train.jsonl`：

- 样本数：`1000`
- response token 均值：`676.03`
- `p50=615`
- `p75=807`
- `p90=997`
- `p95=1193`
- `p99=3076`
- `max=3076`

当前 OPD student/teacher gap 诊断：

- 详见 `docs/analysis/lightning_opd_gap_diagnostics.md`
- 样本数：`1000`
- response token 数：`676030`
- `teacher_logprob > student_logprob`：`29.91%`
- `teacher_logprob < student_logprob`：`60.67%`
- `delta_mean = teacher_logprob - student_logprob = -0.1750`
- top16 overlap 平均交集：`9.84 / 16`
- top16 overlap ratio mean：`0.6148`

## 样本模式判断

JustRL 的主要问题：

- 长推理偏置明显，很多简单题也会输出大量 verification 和自我纠错。
- `<think>...</think>` 后常再写正式答案，导致 `double_boxed` 和重复答案很多。
- 约四成全量样本命中生成上限，说明很多轨迹并非自然结束。
- 对 SFT 直接蒸馏风险较高，容易把学生训练成更啰嗦、更容易截断的风格。

Nemotron 的主要问题：

- raw 模式格式不稳定，boxed 口径不够可靠。
- chat template 能稳定触发 `<think>`，但整体更长，截断比例没有下降。
- 预填“已经思考完”不能有效压缩推理，模型仍会继续展开，且大量样本继续生成 `</think>`。
- 能力不差，但作为直接 SFT label 仍需要过滤或压缩。

## 当前结论

- 如果目标是直接 SFT imitation，JustRL 和 Nemotron 的 raw rollout 都不适合无清洗使用。
- 如果目标是 Lightning-OPD，Nemotron 更适合作 teacher：OPD 只需要 teacher 在 student trajectory 上的 logprob，不需要直接模仿 Nemotron 的长输出风格。
- 当前更稳的路线是：由 `stage1_mix_long_sft/checkpoint-300` 生成 student trajectory，再用 Nemotron 对这些轨迹前向打 topK/logprob，最后从同一个 SFT ckpt 继续 OPD 训练。
- 之前 OPD 显著下降的主要原因不是 Nemotron 一定差，而是训练起点错用了 base model；应改为从 `outputs/stage1_mix_long_sft/checkpoint-300` adapter 继续训练。
- 当前 gap 诊断显示：Nemotron 与 SFT student 的 top16 support 有中等偏高重叠，但 teacher 对 student 实际 sampled token 的 logprob 多数更低；因此 topK support 蒸馏有尝试价值，而继续放大 topK=1 sampled-token OPD 风险较高。

## 后续建议

- SFT 蒸馏不要直接使用 teacher raw 长 COT，应先做长度过滤、boxed 完整性过滤、重复答案清理。
- OPD 小实验优先使用 `max_steps=300`、constant LR、低 warmup、从 SFT ckpt 继续训。
- 若继续放大 OPD，建议额外比较 `distill_top_k=1`、`distill_top_k=16 + topk_kd_weight>0` 与 `topK reverse-KL`，观察 MATH500、boxed rate、输出长度和 gap 指标。
- 每次 teacher/rollout 选择都应保留 `report.json` 和本分析文档更新，避免结论只存在会话里。
