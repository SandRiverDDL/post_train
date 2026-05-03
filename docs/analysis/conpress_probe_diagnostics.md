# ConPress Probe 诊断

## 当前结论

- 裸 prompt 结果全部不再作为有效口径；JustRL-Nemotron / Nemotron 必须走 tokenizer chat template。
- 当前 JustRL-Nemotron 最稳配置是：`exclude_visual=true` + `pack_strategy=balanced_level` + `prompt_style=default`；Qwen3-4B 后续策略以 `docs/analysis/conpress_compression_policy.md` 为准。
- 不建议继续押 XML-ish 标签：one-shot 示例会被复制；无示例 XML 标签 `xml_used=0/14`，模型仍回到自然语言 `Question/Problem` 风格。
- `assistant_prefill="<think>\nQuestion 1:"` 会缩短输出，但明显提高串题，不作为默认配置。
- 当前瓶颈不是压缩强度，而是闭合、切分、答案配对与正确样本率。扩大 rollout 前应做多采样 + rejection。
- visual/asy 不再按“必然破坏格式”处理；Qwen3-4B 消融显示格式稳定，但应单独成桶、小比例强过滤混入。

## 当前有效配置

推荐 probe / 采样配置：

```text
questions_per_prompt=3
samples_per_prompt>=4  # 正式构造数据时用多采样
temperature=0.6
top_p=0.95
max_new_tokens=8192
max_model_len=12288
use_chat_template=true
exclude_visual=true  # 主文本桶默认；visual/asy 另建桶
pack_strategy=balanced_level
prompt_style=default
assistant_prefill=None
```

主文本桶 `balanced_level` 规则：

- 先排除 visual/asy，再打包主文本桶。
- 每个 pack 最多 `1` 道 Level 5。
- 其余 slot 用 Level 1-4 填充。
- 多余 Level 5 直接丢弃或留到下一批，不组成 hard-heavy pack。

visual 分桶规则：

- 命中 `[asy]`、`[/asy]`、`\begin{asy}`、`\includegraphics`、`diagram`、`figure shows`、`the figure`、`tikzpicture`、`graph ... shown below` 时，不进入主文本桶。
- visual/asy 样本不一刀切丢弃；单独成桶，强过滤后按 `5%-10%` 混入。
- 当前 50 题 probe 过滤 `5/50`：`3099`、`6664`、`5723`、`400`、`6353`。

## JustRL-Nemotron 核心实验

同题池：

- `data/rollout/math_sft/justrl_nemotron_chat_probe50_len8192/query_pool.jsonl`
- 单题 chat baseline：`data/rollout/math_sft/justrl_nemotron_chat_probe50_len8192/raw/raw_samples.merged.jsonl`

| 实验 | 目录 | 题数 | 闭合 | correct | parse | boxed | format | pack/single | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| N=2 | `data/probes/conpress_n2_justrl_nemotron_chat_probe50/` | 50 | 18/25 | 25/50 | 43/50 | 43/50 | 0.480 | 0.787 | 压缩弱，6/25 pack 反而更长 |
| N=3 | `data/probes/conpress_n3_justrl_nemotron_chat_probe50/` | 48 | 11/16 | 24/48 | 33/48 | 33/48 | 0.688 | 0.459 | 压缩强，但失败 pack 一次损失 3 题 |
| N=3 no-visual | `data/probes/conpress_n3_justrl_nemotron_chat_probe50_no_visual/` | 45 | 8/15 | 18/45 | 30/45 | 30/45 | 0.667 | 0.465 | 过滤后连续重打包产生 hard-heavy pack，正确率下降 |
| N=3 no-visual balanced | `data/probes/conpress_n3_justrl_nemotron_chat_probe50_no_visual_balanced_level/` | 42 | 10/14 | 24/42 | 33/42 | 33/42 | 0.571 | 0.550 | 当前最佳默认候选 |
| XML one-shot | `data/probes/conpress_n3_justrl_nemotron_chat_probe50_no_visual_balanced_level_xml_oneshot/` | 42 | 11/14 | 17/42 | 36/42 | 36/42 | 0.786 | 0.527 | 示例污染，模型复制示例答案 |
| XML no-example | `data/probes/conpress_n3_justrl_nemotron_chat_probe50_no_visual_balanced_level_xml_tags/` | 42 | 8/14 | 22/42 | 35/42 | 35/42 | 0.643 | 0.556 | XML compliance 为 0，不优于自然语言 |
| Prefill Q1 | `data/probes/conpress_n3_justrl_nemotron_chat_probe50_no_visual_balanced_level_prefill_q1/` | 42 | 9/14 | 23/42 | 32/42 | 32/42 | 0.643 | 0.481 | 更短但串题显著增加 |

补充：

- 单题 chat 输出 words 均值 `2805.5`，p95 `5071.5`，max `6099`。
- N=3 原始压缩约 `54.1%`，但未闭合 pack 是主要损失源。
- balanced 后正确率与 parse 提升，但压缩减弱到约 `45.0%` 长度下降。
- prefill Q1 将 pack/single ratio 降到 `0.481`，但 `cross_talk_block_rate` 从 balanced 的 `0.286` 升到 `0.524`。

## Qwen3-4B Non-Thinking 教师实验

同题池：

- 单题教师 rollout：`data/rollout/teacher_sft/qwen3_4b_non_thinking_step500_chat_nt_probe60_len8192/`
- ConPress：`data/probes/conpress_n3_qwen3_4b_non_thinking_step500_probe60_len8192/`

配置：

```text
model=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
questions_per_prompt=3
temperature=0.8
top_p=1.0
top_k=32
max_new_tokens=8192
max_model_len=9216
use_chat_template=true
chat_template_enable_thinking=false
exclude_visual=true
pack_strategy=balanced_level
prompt_style=default
assistant_prefill=None
```

结果：

- 60 题同题池过滤 visual 后剩 `51` 题，形成 `17` 个 pack。
- 格式稳定性明显好于 1.5B Nemotron：`format_ok=17/17`，`parse=51/51`，`boxed=51/51`，`cross_talk=0/51`。
- 正确率 `48/51 = 94.1%`；同 51 题的单题教师 rollout 为 `49/51 = 96.1%`，正确率损失很小。
- 3 个错误全是 Level 5；Level 2-4 为 `34/34`，Level 5 为 `14/17`。
- pack 长度 words：mean `1028.4`，min `323`，p50 `902`，p90 `1764`，p95 `3118`，max `3118`。
- per-question 长度 words：mean `342.8`，p50 `219`，p90 `713`，p95 `1031`，max `2270`。
- 同 51 题单题教师 rollout words：mean `2053.3`，p50 `2045`，p90 `4114`，p95 `4436`，max `4952`。
- 以 word 口径估算，per-question 压缩到单题的 `16.7%`，约 `6.0x` 压缩。

结论：

- Qwen3-4B non-thinking 对自然语言 ConPress 的指令遵循显著更稳；不需要 XML 或 one-shot。
- 目前可作为冷启动短 CoT 数据候选，但仍应保留过滤：`format_ok && parse_ok && boxed && correct`，并优先排查 Level 5 错误样本。
- 如果正式扩 rollout，主文本桶建议仍保留 `exclude_visual + balanced_level`，并用 `N=3` 作为默认；visual/asy 另建桶强过滤混入，不要为了极少数长尾把 SFT max length 开到 8192。

## Qwen3-4B visual/asy 消融

目录：

- `data/probes/conpress_qwen3_4b_asy_ablation_n3_probe36/n3_len8192_one_visual/`

设计：

- 36 题，12 个 N=3 pack。
- 每个 pack 固定 `1` 道 visual/asy + `2` 道普通题。
- 使用 Qwen3 chat template，`chat_template_enable_thinking=false`。

结果：

- 总体：`format_ok=12/12`、`parse=36/36`、`boxed=36/36`、`correct=25/36`。
- visual 子集：`correct=8/12`、`parse=12/12`、`boxed=12/12`。
- normal 子集：`correct=17/24`、`parse=24/24`、`boxed=24/24`。
- pack tokens mean/p90/p95/max：`821 / 1804 / 2076 / 2076`。
- per-question tokens mean/p90/p95/max：`273.7 / 635 / 1012 / 1609`。

结论：

- 4B 教师下，visual/asy 不再表现为格式杀手。
- 错误主要来自图形语义、读图、数点、阴影面积、坐标比例或选项图形理解，而不是 parser 切分失败。
- 本地 MATH500 有 `42/500 = 8.4%` visual/asy，dev200 有 `24/200 = 12%`；完全不训练这类题不合理。
- 新策略：visual/asy 从主文本桶拆出，正确且格式稳定的样本按 `5%-10%` 混入。

## 难度与闭合

难题越多，结构越不稳定。当前 50 题 probe 上：

```text
no-visual 连续打包：
closed packs 平均 level = 2.92
unclosed packs 平均 level = 4.33
unclosed packs 中 Level 5 = 13/21 题

balanced_level：
每 pack 最多 1 个 Level 5 后：
correct 从 18/45 提升到 24/42
parse 从 30/45 提升到 33/42
closed 从 8/15 提升到 10/14
```

解释：

- hard-heavy pack 会让模型持续处在高熵推理状态，边界 token、`</think>`、尾部答案配对这些软约束更容易被推迟或错过。
- balanced_level 不是提升模型格式能力，而是降低每个 pack 的推理压力。

## Prompt 结论

保留自然语言 prompt：

```text
Please solve each question independently.
Do not restate the problems and do not create new problems.
Use the same numbering in your response: Question 1, Question 2, Question 3.
For each question, provide concise reasoning and end with Final answer: \boxed{...}.
```

不推荐：

- XML one-shot：小模型容易复制示例，污染答案。
- XML no-example：没有真实 XML compliance，仍回退自然语言结构。
- `assistant_prefill="<think>\nQuestion 1:"`：压短但强化 Q1 局部模式，串题增加。

可作为多采样分支：

- prefill Q1 可产生更短候选；只在闭合、可切分、正确且更短时保留。

## Parser 与过滤口径

当前 parser 合理口径：

- 优先按 XML 标签解析，但只有 `<think>` 后很快进入 `<question_A>` 才认 XML。
- XML 晚出现时视为 copied example，忽略其后的 boxed，避免污染 tail pairing。
- 默认 fallback 使用 `Question/Problem/Q` 和 `Answer/Solution` anchor。
- 全文最后 N 个 boxed 用于尾部答案配对，block 内最后 boxed 作为 fallback。
- 支持 `\dfrac -> \frac` 与无序逗号多解等价。

训练样本过滤建议：

- 必须 chat template。
- 主文本桶必须排除 visual/asy；visual/asy 另建桶，不再一刀切删除。
- 必须 `</think>` 闭合。
- 必须可切出对应题段。
- 对应题段或尾部答案必须 boxed/parse_ok/correct。
- 丢弃明显串题、重复 summary、示例复制、答案配对不稳定样本。

## OpenMath-Nemotron 旧结果

旧 90 题 probe 仍作为弱参考：

- 目录：`data/probes/conpress_n3_probe90_nemotron_chat/`
- `format_ok_rate=0.7000`
- `reasoning_split_ok_rate=0.8333`
- `answer_pairing_ok_rate=0.7667`
- `question_correct_rate=0.6222`
- `question_parse_ok_rate=0.8000`
- `avg_output_tokens=2970.9` / pack
- `avg_question_tokens=660.6`

早期结论仍成立：

- 裸 prompt 无效。
- 对 1.5B / 旧 Nemotron，图形/asy 题显著破坏多题稳定性；对 Qwen3-4B，最新消融显示格式稳定，但正确率和图形语义仍需单独评估。
- 显式 delimiter 与 XML-like 强格式都不如模型自然的 `Question/Problem` 风格可靠。

## 下一步

1. 使用当前推荐配置跑 `samples_per_prompt=4-8` 的小规模 probe。
2. 对每题做 rejection：优先保留正确、闭合、可切分、短的轨迹。
3. 统计 retained rate、平均长度、p95 长度、按 level/type 的保留率。
4. retained rate 稳定后再扩大到更大 MATH train 子集。
