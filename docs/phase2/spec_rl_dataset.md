# Phase 2 数据规格

## 目标

为 GRPO 提供一份短、干净、可验证、适合 rollout 的冻结训练工件。

## 默认数据源

- `openai/gsm8k` `train(main)`

当前 GRPO 主线不再默认使用 `NuminaMath`，因为对 `1.7B/4B` 冷启动 policy 来说，reward 过于稀疏。

## 冻结工件

- `data/grpo/train_grpo_gsm8k_2k_short.jsonl`
- `data/grpo/train_grpo_gsm8k_tiny_short.jsonl`（单卡调试工件）

GRPO 不直接复用 SFT 工件路径。

## 样本选择

默认规则：

- 总量：`2000`
- 随机采样
- `response token length <= 128`
- `final_answer` 可稳定抽取
- 可映射到 boxed 协议

tiny 调试规则：

- 总量：`64`
- 随机采样
- `response token length <= 128`
- 仅用于快速验证 reward 与 rollout 是否工作

## 字段约定

每条记录至少包含：

```json
{
  "id": "...",
  "question": "...",
  "prompt": "...",
  "final_answer": "...",
  "source": "gsm8k",
  "problem_type": "Arithmetic"
}
```

可选调试字段：

- `response_tokens`
- `reference_solution`

## 原则

- 优先短 response，降低 rollout 难度
- 优先选择 reward 更密集、答案更短的题源
- 主线优先保证 `1.7B` 单卡可训练，而不是追求更难数学题覆盖

## 当前实现

- 数据准备入口：[prepare_grpo_data.py](/home/chy/code/active/rl/scripts/prepare_grpo_data.py)
- 离线打分入口：[score_grpo_candidates.py](/home/chy/code/active/rl/scripts/score_grpo_candidates.py)
- 训练子集选择入口：[select_grpo_scored_subset.py](/home/chy/code/active/rl/scripts/select_grpo_scored_subset.py)

当前数据链为：

- `candidates`
- `scored`
- `select`

当需要引入外部候选源扩充训练集时，当前新增一条并行链路：

- `filter_big_math_dataset.py`
- `merge_train_corpora.py`
- `select_merged_corpus.py`

约定：

- `filter_big_math_dataset.py` 当前只支持 `open-r1/Big-Math-RL-Verified-Processed`
- `quintile_2` 通过 HF `config` 选择，不是样本字段
- 过滤后先输出标准化 JSONL 工件，再在最终合并阶段统一去重
- manifest YAML 当前同时声明：
  - `inputs`
  - `dedup_against`
  - `merge_output`
  - `selection`
  - `final_output`
- `merge_train_corpora.py` 只负责生成去重后的 pool 工件
- `select_merged_corpus.py` 负责按 `target_size + per_source min/max + weight` 生成最终训练子集

其中 `select_grpo_scored_subset.py` 默认输出全部满足阈值的样本；只有在显式传入 `target-size` 时，才会在过滤后的结果上继续做可选抽样。

## 中等难度筛选

当 `GSM8K short` 候选集较大时，GRPO 主线优先使用经 `SFT` policy 离线打分后的中等难度子集，而不是直接随机抽样。

当前默认打分规则：

- 后端：`vLLM`
- 每题采样：`4` 次
- 采样参数：`temperature=0.8`，`top_p=0.95`
- 判分协议：strict boxed

当前默认筛选规则：

- `0.25 <= correct_rate <= 0.50`
- `parse_rate >= 0.50`

打分结果应先全量持久化为 scored 工件，再由 `select_grpo_scored_subset.py` 生成全量 filtered 工件；若有实验预算约束，可在同一脚本中追加 `target-size` 生成最终训练子集。

近期实验补充：

- 当筛选带宽收紧到 `0.25 <= correct_rate <= 0.50` 时，当前 `GSM8K short` 候选池只剩约 `705` 条样本。
- 这说明 `response token length <= 256` 可能过严，后续应优先验证将候选长度上限放宽到 `384`，而不是直接回退到更宽的难度带宽。
- 当 `num_samples_per_problem = 4` 时，`correct_rate` 仅能取 `{0, 0.25, 0.5, 0.75, 1.0}`，因此不推荐使用过细的难度阈值。
- 当前已确认：如果训练集是按旧模型或旧 prompt 打分筛出来的 middiff 工件，那么继续直接复用到新模型/新 prompt 的 GRPO 线上，容易放大 late-stage drift。
- 但这不是当前唯一根因；即使切到新模型线，若仍在同一批静态窄 middiff 样本上持续训练，也会继续复现 “early checkpoint 优于 final”。
- 因此 middiff 工件应视为**与 scoring model + prompt version 绑定**的实验工件，而不是可跨模型直接复用的通用训练集。
- 现阶段更推荐把 `scored/select` 当成短程 GRPO 的一轮工件，而不是一次筛完后长时间反复训练到底。

## 非目标

- v1 不追求覆盖最大难度
- v1 不引入第二套独立 RL 数据格式
- v1 不做模糊近似去重
- v1 不做复杂语义级 proof / explanation 分类
