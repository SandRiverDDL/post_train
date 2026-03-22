# State

## 当前阶段

- `SFT` 阶段已经完成，当前只作为冷启动背景，不再是主阻塞。
- 当前主任务是 `Phase 2 / GRPO`，主干路线为 `Unsloth GRPO + 单卡优先 + QLoRA continuation`。
- `TRL GRPO` 的排障代码与历史结论保留在独立分支，主干不再继续维护双实现。
- 评测配置已从 `configs/sft.yaml` 拆出，当前默认评测配置见 `configs/eval.yaml`。
- 正式评测入口仍为 `scripts/eval_dataset.py`，默认模式为本地 boxed 协议评测；`lm-evaluation-harness` 原生 benchmark 仅保留为可选对照模式，不能再直接当作 Phase 2 正式结论。
- GRPO 数据工件链当前为 `prepare_grpo_data -> score_grpo_candidates -> select_grpo_scored_subset`，其中 `select_grpo_scored_subset` 默认输出全部符合阈值样本，传入 `target-size` 时再做可选抽样。

## 当前有效基线

- 开发模型已分成两条：
  - `Qwen3-1.7B-Base`
  - `Qwen3-0.6B-Base`（用于快速迭代）
- 当前冷启动模型：
  - `outputs/sft-qwen3-1.7b`
  - `outputs/sft-qwen3-0.6b-v2`
- 当前 GRPO 基座：
  - `Qwen/Qwen3-1.7B-Base`
  - `Qwen/Qwen3-0.6B-Base`
- 当前主训练配置见 [grpo_base.yaml](/home/chy/code/active/rl/configs/grpo_base.yaml)
- 当前最近一轮主训练数据：`data/grpo/train_grpo_gsm8k_3857_short_middiff.jsonl`
- 当前更窄难度带宽实验已验证：`0.25 <= correct_rate <= 0.50` 时，现有 `response token <= 256` 候选池仅剩约 `705` 条样本，说明候选长度上限可能过严。
- 当前已新增一条并存的快速迭代链：`Qwen3-0.6B-Base + prompt_v2`，配置入口为：
  - [sft_qwen3_0.6b_v2.yaml](/home/chy/code/active/rl/configs/sft_qwen3_0.6b_v2.yaml)
  - [eval_qwen3_0.6b_v2.yaml](/home/chy/code/active/rl/configs/eval_qwen3_0.6b_v2.yaml)
  - [grpo_base_qwen3_0.6b_v2.yaml](/home/chy/code/active/rl/configs/grpo_base_qwen3_0.6b_v2.yaml)

`prompt_v2` 定义为英文 instruction-first 模板：

```text
Please reason step by step, and put your final answer within \boxed{}.

Question:
...

Solution:
```

SFT 结论只保留两点：

1. `1.7B + LoRA` 可以稳定学会 `Final answer: \boxed{...}` 协议。
2. 当前 SFT 基线在正式 boxed 评测下可作为 GRPO 冷启动，但已经不算低，后续 RL 的可提升空间有限。

## 当前 GRPO 实现状态

已完成：

- [scripts/train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py) 可在真实 GPU 环境启动并完成一轮训练。
- reward 已收敛为单一 `combined_reward`，实现见 [grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)。
- `GRPO` 配置已支持 `_base_` 继承，实验 yaml 可只保留差异字段。
- `wandb` 与 `train_log.jsonl` 监控已接通。
- 正式评测默认使用本地 prompt + `math-verify` 提取与等价判定，并输出 `format_success_rate / parse_success_rate / normalized_accuracy` 及其 `stderr`。
- 候选集离线难度打分与子集选择脚本已加入：
  - [score_grpo_candidates.py](/home/chy/code/active/rl/scripts/score_grpo_candidates.py)
  - [select_grpo_scored_subset.py](/home/chy/code/active/rl/scripts/select_grpo_scored_subset.py)

当前 reward 口径：

- `correct_relaxed: +1.0`
- `wrong: 0.0`
- `parse fail: -0.1`
- `strict_boxed_bonus: 0.0`
- `length penalty: 0.0`

这些 reward 项系数已经暴露在 yaml 中，可直接做实验切换。

## 当前最重要的问题

当前问题不是“GRPO 跑不起来”，也不再是“评测器把模型误判”，而是：

- **在正式 boxed 评测口径下，GRPO 的有效收益窗口很短，early checkpoint 往往优于 final；继续在同一批静态 middiff 数据上训练会出现 late-stage drift。**

当前已确认的现象：

- `1.7B` 线上，`checkpoint-25` / `checkpoint-50` 多次优于更晚 checkpoint 或 final
- `1.7B + beta=0.001` 线上，`checkpoint-25 = 0.805`，而 `final@100step = 0.775`
- `0.6B v2` 线上，`SFT = 0.460`，`checkpoint-25 = 0.455`，而 `final@50step = 0.435`
- 在这些退化 run 中，常见形态不是 `format/parse` 崩溃，而是：
  - `format_success` 持平或更高
  - `parse_success` 持平或更高
  - `normalized_accuracy` 反而下降

当前判断：

1. 这更像是真实的训练目标漂移，而不是 benchmark bug。
2. 共同根因不是单个 `beta` 或单个模型大小，而是：**静态窄数据集 + 终局二值 reward + 训练过头**。
3. “按当前模型与 prompt 重新打分筛题” 仍然重要，但它不能单独解释全部问题；即使切到 `0.6B + prompt_v2`，仍然复现了相同的 late-stage drift 模式。
4. 当前 GRPO 更像是在重排解题风格：前期修正部分题意理解或格式问题，后期则逐步伤害泛化。
5. `checkpoint selection / early stopping` 已经不是优化细节，而是当前实验设计的一部分。

补充：

- 已完成一次 `SFT vs GRPO official dev200` 的逐题互换分析，详细案例与结论见 [experiment_error_swaps.md](/home/chy/code/active/rl/docs/phase2/experiment_error_swaps.md)。

## 关键证据

最近几轮日志与评测共同支持：

- `format/parse` 经常早早接近饱和
- `0.6B` 线上即使 `frac_reward_zero_std` 不算特别高，也仍然会发生 late-stage drift
- `1.7B` 与 `0.6B` 都出现“早期 checkpoint 更好、后期回落”的相同模式

说明：

- rollout 和协议链路不是当前主故障点
- 当前也不能只用 `frac_reward_zero_std` 一个指标解释全部退化
- benchmark 主链也不是当前首要怀疑对象；同一条 official 评测链已经能够稳定区分 `checkpoint-25` 与 final
- 更核心的问题是：**当前 reward 只能区分最终对错，无法区分“稳定正确的轨迹”和“偶然拿到 1 分的脆弱轨迹”**

当前最需要验证的是：

- 短程训练 + 更密 checkpoint 评测是否比长程固定 step 更稳
- 是否应该把单次 GRPO 训练窗口进一步收紧到 `20~30 step`，并默认顺序评测 `checkpoint-10/20/25/30`
- 针对当前模型与 prompt 重新打分筛题，而不是沿用旧模型口径的 middiff 工件
- 候选池 `response token` 上限是否需要从 `256` 放宽到 `384`
- 是否需要引入更广但仍可验证的数据，而不是继续在静态窄 `GSM8K middiff` 上反复训练
- `eval_grpo_checkpoints.py` 是否需要改成走 `official` 评测链路；当前它仍是旧 harness 口径

## 当前优先级排序

1. 把 `checkpoint selection / early stopping` 视为主流程，不再默认 final 最优。
2. 缩短单次 GRPO 训练窗口，并把早停点搜索纳入默认流程。
3. 针对当前模型与 prompt 重新打分筛题，而不是沿用旧口径 middiff 工件。
4. 提高 GRPO 组内学习信号，而不是继续追 reward 系数微调。
5. 验证更广或更动态的候选池是否优于当前静态 `GSM8K middiff`。
6. `MATH500 test` 继续保留为辅助 benchmark，不作为当前主开发指标。

## 需要下一个会话优先理解的事实

1. 主干已经不是“修训练脚本能否启动”的阶段，而是“训练目标是否把模型拉偏”的阶段。
2. 当前最核心的问题是 **静态窄数据集 + 终局二值 reward 在短窗口后会把模型拉偏**。
3. 当前不应再把主要精力放在：
   - `TRL` 兼容
   - boxed 协议学习
   - reward 稀疏
4. 当前应重点检查：
   - 当前训练集是否按当前模型与 prompt 重新打分
   - 是否需要更短训练窗口与更密 checkpoint
   - 候选池长度过滤是否过严
   - 中间 checkpoint 是否优于最终点

## 文档口径

- 当前会话应优先读取 `STATE.md` 与 `TASK.md` 作为事实来源。
- `docs/phase1_mvp.md` 与 `docs/phase1_cold_start_plan.md` 仅保留历史背景，不再作为当前 Phase 2 的主执行文档。
