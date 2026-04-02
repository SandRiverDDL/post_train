# RSR 设计说明

这份文档固定当前仓库内 `RSR` 数据构造与筛选的实现口径，避免后续实验时逐步偏移。

## 目标

从四个 `UWNSL` 数学轨迹数据集中构造一份 `RSR` 候选池，再按 `RSR` 分数和轨迹数量优先级筛出最终的 stage1 SFT 数据集。

当前固定使用的数据源：

- `UWNSL/MATH_training_split_short_cot`
- `UWNSL/MATH_training_split_long_cot`
- `UWNSL/MATH_training_split_distill_small_teacher`
- `UWNSL/MATH_training_split_distill_large_teacher`

## 两步流程

### 第一步：构造 RSR 候选池

- 按 `problem` 做题面对齐。
- 标准化只做：
  - 去首尾空白
  - 连续空白折叠
  - 换行视作空格
- v1 只做标准化后的精确匹配，不做编辑距离或模糊匹配。
- 每个 `problem` 保留全部可用轨迹，而不是提前只选一条。
- 每条轨迹会：
  - 规范化成统一 `SFTRecord`
  - 校验 `\boxed{}`、尾答可解析、尾答一致性
  - 在当前 stage1 base/student 模型下计算 `RSR`
  - 若 `prompt + solution` 超过 `max_seq_length`，默认直接丢弃，不参与打分

候选池一行对应一个 `problem`，其中包含：

- `question`
- `trajectory_count`
- `available_sources`
- `best_source_by_rsr`
- `best_rsr`
- `trajectories[]`

`trajectories[]` 中每条轨迹至少包含：

- `source_name`
- `source_dataset`
- `solution`
- `final_answer`
- `solution_tokens`
- `prompt_tokens`
- `scored_solution_tokens`
- `truncated`
- `avg_rank_clip`
- `avg_surprisal`
- `rank_surprisal_ratio`

### 第二步：筛选最终 SFT 数据集

- 每个 `problem` 最终只保留一条轨迹。
- 默认优先级：
  1. `trajectory_count` 更高的问题优先
  2. 同 bucket 内按最优轨迹 `RSR` 更小优先
  3. 若 `RSR` 相同，`solution_tokens` 更短优先
  4. 若仍相同，按固定来源顺序 `large > small > short > long`
- 若设置长度阈值，则先在单题内过滤轨迹，再从该题剩余轨迹里选 best。
- 可选地通过 `allowed_sources` 只允许从指定来源里选轨迹，例如先排除 `long` 做对照。
- 若该题所有轨迹都被长度过滤掉，则该题被丢弃。

## RSR 计算口径

当前实现遵循官方 `Rank-Surprisal Ratio` 仓库的主定义：

- prompt 使用当前仓库的 `build_sft_prompt(question)`
- 只对 completion token 计算分数
- 每个 token 计算：
  - `nll`
  - `rank_clip`
- 单条轨迹分数定义为：

`rank_surprisal_ratio = sum(rank_clip) / sum(nll)`

其中：

- `rank_clip_r` 当前默认是 `100`
- `rank` 先算 token 在词表中的名次，再截断到 `rank_clip_r`

## 当前已知交集事实

按当前标准化精确匹配统计，四源 `problem` 覆盖分布为：

- `4轨`: `4288`
- `2轨`: `1344`

其中两种 `2轨` 组合分别是：

- `short + long`: `1094`
- `small + large`: `250`

这意味着 v1 不需要模糊匹配也足够构造一份高质量候选池。

## 输出与报告

第一步输出：

- `data/stage1/rsr/candidates.jsonl`
- `data/stage1/rsr/candidates.report.json`
- `data/stage1/rsr/candidates.unmatched.preview.jsonl`

第二步输出：

- `data/stage1/rsr/selected_train.jsonl`
- `data/stage1/rsr/selected_train.report.json`

最终筛选报告必须包含：

- `trajectory_count` 分布
- `chosen_source_counts`
- `chosen_source_ratio`
- 长度过滤淘汰数量
- 超长轨迹的总数、默认舍弃数量与舍弃比例

## 默认假设

- `RSR` 打分模型使用当前 stage1 的 base/student 模型，不使用已训 short-cot 模型。
- `drop_truncated` 默认开启，超长轨迹默认丢弃而不是截断参与 `RSR`。
- v1 不做同题多轨训练。
- v1 不做 fuzzy match。
- 如果同一题不同来源在校验后仍出现非等价尾答，则整题丢弃。
