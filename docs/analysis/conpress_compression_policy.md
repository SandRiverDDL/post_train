# ConPress 压缩数据策略

## 当前结论

- Qwen3-4B Non-Thinking 教师已能稳定遵循自然语言 N=3 ConPress 格式；默认不需要 XML、one-shot 或 assistant prefill。
- 压缩数据不能只看格式，正式入训仍以 `correct + parse_ok + boxed + format_ok` 为硬过滤。
- `asy/visual` 不应再一刀切丢弃；应从主文本桶中拆出单独 visual 桶，强过滤后按小比例混入。
- Level 5 尤其是参数约束、根个数、有理函数、复杂代数题容易产生长尾和结构风险；应使用更小打包比或更强 rejection。

## 推荐分桶

主文本桶：

- 题面不含 `[asy]`、`\begin{asy}`、`\includegraphics`、`diagram`、`figure shows`、`the figure`、`tikzpicture` 等 visual marker。
- 默认 `N=3`，使用 `balanced_level`，每个 pack 最多 1 道 Level 5。
- 用作高置信短 CoT 主数据。

visual/asy 桶：

- 保留含图形或 Asymptote 代码的题，不进入主文本桶。
- 默认也可先用 `N=3` probe；正式数据只收 `correct + parse_ok + boxed + format_ok + exact_n_boxed`。
- 建议混入比例先接近 MATH500 分布：`5%-10%`。
- 依赖读图、数点、阴影面积、选项图形的错题不直接丢弃，进入 hard visual candidate 池，后续用 N=1 或更强 teacher 生成。

Level 5 / hard 桶：

- 不建议全 Level 5 使用 `N=3`。
- 当前 36 题 Level5-only probe 显示：`N=3` 为 `correct=26/36`、`parse=32/36`；`N=2` 修正 prompt 后为 `correct=27/36`、`parse=32/36`。
- 推荐：L1-L4 默认 `N=3`；L5 默认 `N=2` 或 “每 pack 最多 1 个 L5”；长尾题降到 `N=1`。

## 关键实验事实

Qwen3-4B Non-Thinking 60 题 no-visual balanced：

- `format_ok=17/17`
- `parse=51/51`
- `boxed=51/51`
- `correct=48/51`
- per-question 平均长度约为同题单题 rollout 的 `16.7%`

Qwen3-4B Non-Thinking visual/asy 消融：

- 实验：36 题，12 个 N=3 pack，每个 pack 固定 1 道 visual/asy + 2 道普通题。
- 总体：`format_ok=12/12`、`parse=36/36`、`boxed=36/36`、`correct=25/36`。
- visual 子集：`correct=8/12`、`parse=12/12`、`boxed=12/12`。
- 本地 MATH500 中 visual/asy 占比为 `42/500 = 8.4%`；dev200 中为 `24/200 = 12%`。

解释：

- 这次 4B 实验不支持“visual/asy 会必然破坏格式”的说法。
- visual/asy 的主要风险是题面理解和图形语义，而不是 parser 格式。
- `math-verify` 能过滤最终答案错误，但不能证明正确样本的图形推理过程没有噪声；因此 visual 样本应单独成桶、小比例混入，而不是完全过滤。

## 默认过滤

所有桶共同硬过滤：

- 必须使用 tokenizer chat template；Qwen3 Non-Thinking 必须 `enable_thinking=false`。
- 必须能按题号切出对应 block。
- 每题必须有可解析的 `\boxed{...}`。
- 最终答案必须由本地 parser + `math-verify` 判定正确。
- 丢弃串题、示例复制、答案配对不稳定、多个互相冲突 boxed、截断样本。

长度建议：

- 先以 per-question p95 控制 SFT max length；不要为了极少数长尾直接把主训练 max length 开到 8192。
- `>3000 tokens` 的正确样本可保留到长样本桶；错误、无 boxed 或 parse fail 的长尾直接丢弃。

## 当前推荐执行口径

1. 先按主文本桶生成主数据：`N=3 + balanced_level + multi-sample rejection`。
2. 单独扩大 visual/asy probe，确认 retained rate 后按 `5%-10%` 混入。
3. Level 5 单独统计 retained rate；若 `N=3` 掉正确率，降到 `N=2` 或 `N=1`。
4. 最终训练集报告必须分桶统计数量、正确率、长度 p95、Level/type 分布和 visual 占比。
