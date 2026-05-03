# Teacher Rollout 诊断：JustRL 与 OpenMath-Nemotron

## 背景

本诊断用于记录 Lightning-OPD / SFT 蒸馏前对 teacher rollout 的观察，避免只依赖会话上下文。

相关本地产物：

- JustRL 全量采样：`data/rollout/math_sft/justrl_deepseek_math_all_len3650/raw/raw_samples.merged.jsonl`
- Nemotron raw probe：`data/rollout/math_sft/openmath_nemotron_len_probe_100/shards/shard0-of-1/`
- Nemotron chat probe：`data/rollout/math_sft/openmath_nemotron_chat_probe_100/raw_samples.jsonl`
- Nemotron chat prefill probe：`data/rollout/math_sft/openmath_nemotron_chat_prefill_done_probe_100/raw_samples.jsonl`
- Nemotron ConPress probe：`data/probes/conpress_n3_smoke9_nemotron_chat/`
- JustRL-Nemotron 裸 prompt probe：`data/rollout/math_sft/justrl_nemotron_probe50_len8192/raw/raw_samples.merged.jsonl`
- JustRL-Nemotron chat-template probe：`data/rollout/math_sft/justrl_nemotron_chat_probe50_len8192/raw/raw_samples.merged.jsonl`
- Qwen3-4B-2507 Thinking 5k 数据：`data/rsr/qwen3_4b_2507/short-sys_5k_gen1.json`
- Qwen3-4B Non-Thinking Step500 教师 SFT rollout probe：`data/rollout/teacher_sft/qwen3_4b_non_thinking_step500_chat_nt_probe30/`
- 当前 OPD 训练集：`data/lightning_opd/nemotron_candidate/train.jsonl`
- 当前 OPD gap 诊断：`data/lightning_opd/nemotron_candidate/student_teacher_gap_report.json`

## Prompt Template 规则

- JustRL、Nemotron、Qwen 系 chat/instruct/reasoning 模型做 rollout 时，必须先确认是否使用 tokenizer chat template。
- 对本仓库当前 MATH rollout，裸 prompt 不能用于判断 `<think>` 格式稳定性，因为它绕过了模型训练时的 `<|im_start|>user` / `<|im_start|>assistant` 对话格式。
- 裸 prompt probe 只能作为弱参考，用于观察大致长度和能力；不能作为 summary 边界、`<think>` 闭合率、boxed 风格的正式结论。
- 若目标是稳定 thinking 阶段，应使用 `tokenizer.apply_chat_template(..., add_generation_prompt=True)`；如需强制边界，再显式 assistant prefill `<think>\n`，并把 prefill 写入 report。
- Qwen3 Non-Thinking 模型必须显式传 `enable_thinking=False`。该设置会在 rendered prompt 的 assistant prefix 中插入空 `<think>\n\n</think>\n\n`；这不是要求 response 生成 `<think>`，因此 response 里的 `<think>` 计数应通常为 `0`。
- 教师 SFT rollout 不应放入 `data/lightning_opd/`，也不应生成 `teacher_topk`；Lightning-OPD 只在后续读取 raw rollout 做 teacher forward / topK logprob。

## 教师 SFT rollout 入口

- 入口：`scripts/rollout_teacher_sft.py`
- 核心逻辑：`src/post_train/rollout/teacher_sft.py`
- 后台分片：`--mode launch --gpus 0,2,7 --num-shards 3`
- 单 shard 输出：`shards/shard{i}-of-{n}/raw_rollouts.jsonl` 与 `report.json`
- merge 输出：`raw/raw_rollouts.merged.jsonl` 与根目录 `report.json`
- `launch` 会用 `tmux new-session -d` 后台启动各 shard，并写 `logs/shard{i}.log` 与 `jobs.json`。

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

当前 ConPress 9 题 smoke：

- 详见 `docs/analysis/conpress_probe_diagnostics.md`
- 裸 prompt 会诱导 Nemo 续写 `Question 4/5/6`，不能作为有效结论。
- chat template 后模型会按 `Problem A/B/C` 推理，但不遵守 `Answer A/B/C` 硬格式。
- 离线重解析显示 `3` 个 pack 中约 `2` 个可切出有用轨迹，另 `1` 个图形/asy 类 pack 明显失败。
- 需要先修 parser 和过滤图形题，再扩大到 `30-60` 题 probe。

JustRL-Nemotron 裸 prompt 50 条 probe：

- 产物：`data/rollout/math_sft/justrl_nemotron_probe50_len8192/raw/raw_samples.merged.jsonl`
- 该 probe 使用 `scripts/rollout_math_sft.py`，当时未套 tokenizer chat template；因此以下 `<think>` 统计不能作为模型真实格式能力结论。
- 样本数：`50`
- raw parser 正确：`33/50`
- 明显 parser 假阴性较多，粗看等价正确可能约 `40-42/50`
- 输出 tokens：平均 `2346`，`p50=2072`，`p75=3528`，`p90=3894`，`p95=5115`，`max=5734`
- 未命中 `8192` 上限，说明本次主要不是 max token 截断。
- tag 形态：
  - `<think>`：`8/50`
  - `</think>`：`39/50`
  - 两者都有：`7/50`
  - 只有 `</think>`：`32/50`
  - 两者都没有：`10/50`
- 有 `</think>` 的样本中，summary 平均约 `262` words，`p50=237`，`p95=402`。
- 冗余明显：平均 `Wait/Hmm/Let me/verify/Alternatively` 类标记约 `48` 次；大量样本多次出现 Final Answer 或多个 boxed。
- 结论：该 probe 可说明模型在裸 prompt 下仍能生成较短于 2507 的轨迹，但不能说明 chat template 下的 `<think>` 稳定性。

JustRL-Nemotron chat-template 50 条 probe：

- 产物：`data/rollout/math_sft/justrl_nemotron_chat_probe50_len8192/raw/raw_samples.merged.jsonl`
- 已确认 `use_chat_template=true`，`system_prompt=None`，`assistant_prefill=None`。
- 样本数：`50`
- raw parser 正确：`31/50`
- `parse_ok=42/50`
- `boxed=42/50`
- 输出 tokens：平均 `2805.5`，`p50=2264.5`，`p75=3949.5`，`p90=4634.8`，`p95=5071.5`，`max=6099`
- 未命中 `8192` 上限。
- tag 形态：
  - `<think>`：`50/50`
  - `</think>`：`42/50`
  - 两者都有：`42/50`
- 有 `</think>` 的样本中，summary 平均约 `230.5` words，`p50=241.5`，`p95=327.9`。
- 结论：chat template 明显修复 `<think>` 起始边界，但闭合仍不是 `100%`，输出仍偏长，raw correct 没有明显改善；直接 raw SFT 仍不推荐。

Qwen3-4B-2507 Thinking 5k 数据：

- 原始样本数：`5000`
- 可用题面和 `EleutherAI/hendrycks_math` train 精确匹配：`1667`
- 未匹配：`3333`
- 匹配到的 MATH 子集只包含 Level 4/5：
  - Level 4：`678`
  - Level 5：`989`
- 原始 MATH train 分布明显不同：
  - Level 1：`564`
  - Level 2：`1348`
  - Level 3：`1592`
  - Level 4：`1690`
  - Level 5：`2303`
- 因此该子集不是自然难度分布，而是强偏 Level 4/5 的 hard subset。
- 使用 Qwen2.5-Math tokenizer 估算匹配子集长度：
  - 完整 assistant answer 平均：`9560.2` tokens
  - `p50=8315`
  - `p75=13020`
  - `p90=18754`
  - `p95=23343`
  - `max=31003`
  - `</think>` 后 summary 平均：`772.5` tokens
  - summary `p50=738`
  - summary `p95=1272`

结论：Qwen3-4B-2507 Thinking 的完整思维链过长，当前不适合作为 1.5B SFT 直接标签；`</think>` 后 summary 长度合理，可作为未来 clean-solution 候选，但该路线暂时搁置。

## 样本模式判断

JustRL 的主要问题：

- 长推理偏置明显，很多简单题也会输出大量 verification 和自我纠错。
- `<think>...</think>` 后常再写正式答案，导致 `double_boxed` 和重复答案很多。
- 约四成全量样本命中生成上限，说明很多轨迹并非自然结束。
- 对 SFT 直接蒸馏风险较高，容易把学生训练成更啰嗦、更容易截断的风格。
- 早期 JustRL / JustRL-Nemotron probe 中存在未套 chat template 的结果；凡涉及 `<think>` 出现率、闭合率、summary 边界的结论，必须回查具体 rollout 配置。

Nemotron 的主要问题：

- raw 模式格式不稳定，boxed 口径不够可靠。
- chat template 能稳定触发 `<think>`，但整体更长，截断比例没有下降。
- 预填“已经思考完”不能有效压缩推理，模型仍会继续展开，且大量样本继续生成 `</think>`。
- 能力不差，但作为直接 SFT label 仍需要过滤或压缩。

Qwen3-4B-2507 Thinking 的主要问题：

- 完整 `<think>` 思维链极长，Level 5 平均超过 `11k` tokens，不适配当前 1.5B / 4096 上下文 SFT。
- 数据难度分布只覆盖 MATH Level 4/5，无法替代自然 MATH 分布或通用 warmup 数据。
- `</think>` 后 summary 更接近 instruct-style clean solution，长度可控，但仍需 boxed / correct / 格式过滤后才能使用。

## 当前结论

- 如果目标是直接 SFT imitation，JustRL 和 Nemotron 的 raw rollout 都不适合无清洗使用。
- 如果目标是 Lightning-OPD，Nemotron 更适合作 teacher：OPD 只需要 teacher 在 student trajectory 上的 logprob，不需要直接模仿 Nemotron 的长输出风格。
- 当前更稳的路线是：由 `stage1_mix_long_sft/checkpoint-300` 生成 student trajectory，再用 Nemotron 对这些轨迹前向打 topK/logprob，最后从同一个 SFT ckpt 继续 OPD 训练。
- 之前 OPD 显著下降的主要原因不是 Nemotron 一定差，而是训练起点错用了 base model；应改为从 `outputs/stage1_mix_long_sft/checkpoint-300` adapter 继续训练。
- 当前 gap 诊断显示：Nemotron 与 SFT student 的 top16 support 有中等偏高重叠，但 teacher 对 student 实际 sampled token 的 logprob 多数更低；因此 topK support 蒸馏有尝试价值，而继续放大 topK=1 sampled-token OPD 风险较高。
- Qwen3-4B-2507 Thinking 完整轨迹暂不进入当前 SFT/DFT 主线；若后续使用，应优先只抽取 `</think>` 后 summary，而不是完整 thinking。
- JustRL-Nemotron 已重跑 chat-template 版；它适合继续作为 teacher 候选观察 summary-only 或 filtered trajectory，不适合直接 raw SFT。

## 后续建议

- SFT 蒸馏不要直接使用 teacher raw 长 COT，应先做长度过滤、boxed 完整性过滤、重复答案清理。
- 在扩大任意 teacher rollout 前，先做 50 条 chat-template smoke，并把 `use_chat_template`、`system_prompt`、`assistant_prefill` 写进 report。
- ConPress 路线不应按当前 parser 的 0 分否定 Nemo；应先支持 `Problem A/B/C` reasoning anchor、尾部 boxed summary 配对和无序逗号多解等价。
- Qwen3-4B-2507 Thinking hard-MATH 数据先搁置；后续若恢复，优先做 summary-only 提取与同题 mix-long / ConPress 对照。
- OPD 小实验优先使用 `max_steps=300`、constant LR、低 warmup、从 SFT ckpt 继续训。
- 若继续放大 OPD，建议额外比较 `distill_top_k=1`、`distill_top_k=16 + topk_kd_weight>0` 与 `topK reverse-KL`，观察 MATH500、boxed rate、输出长度和 gap 指标。
- 每次 teacher/rollout 选择都应保留 `report.json` 和本分析文档更新，避免结论只存在会话里。
