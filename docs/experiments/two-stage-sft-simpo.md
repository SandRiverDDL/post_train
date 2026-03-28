# 两阶段 SFT + SIMPO（暂停路线）

## 状态

这条路线当前暂停，不是项目的默认主线。

保留这份文档的目的：

- 记录已经试过的设计与口径
- 保留后续重新实验时的上下文
- 避免把旧方案继续写在现行 `docs/SPEC.md` 里误导执行

## 原方案目标

原目标是建立一个最小可运行的两阶段 `SFT + SIMPO` 数学后训练闭环。

原方案关注点：

1. 跑通 `stage1 SFT`
2. 跑通 `stage2 SFT`
3. 跑通 `SIMPO`
4. 用统一评测链路比较三个阶段结果

## 原方案定义

### Stage1 SFT

- 用一份基础数学 SFT 数据训练第一阶段模型。
- 当前数据口径：`UWNSL/MATH_training_split_short_cot`
- 当前首版训练规模：`2000`
- `dev`：从同源数据中切出 `200` 条并冻结
- 当前 checkpoint 选择口径：`2 epoch`、`save_steps=25`、训练结束后统一在 `dev200` 上按 `normalized_accuracy` 选 best checkpoint

### Stage2 SFT

- 定义为“在另一份 SFT 数据上继续训练一次”。
- 当前绑定到一条“清洗与筛选”数据管线，而不是模型采样管线。
- 当前主数据源：`qingy2024/OpenR1-Math-220k-Cleaned`
- 当前还保留一条 `hendrycks_math + long CoT` 对照线：
  - 从 `EleutherAI/hendrycks_math` 中筛 `level 4/5`
  - 与 `UWNSL/MATH_training_split_long_cot` 按标准化后的 `problem` 精确匹配
  - 默认过滤 `solution_tokens > 4096`
  - 与现有 short CoT 数据按 `long:short = 1:2` 混合成 `2000` 条
  - 当前默认从 `outputs/stage1_mix_long_sft/checkpoint-300` 继续训练
- 当前 MVP 配比：`50%` 随机 `stage1_train` + `35%` `math220k_short(<768)` + `15%` `math220k_long(<1380)`
- 当前 `math220k` 绑定规则：
  - 优先使用 `clean_problem`
  - `question_type == "MCQ"` 直接过滤
  - 尾部标准化为单个 canonical `\boxed{...}`，避免重复 boxed
- 当前 stage2 target dev 口径：
  - 不新增 YAML
  - 通过固定 `profile` 生成
  - 第一版只内置 `main150` 与 `short150`
- 当前 stage2 训练建议：
  - 从 `stage1 best checkpoint` 继续训练
  - 学习率优先用 `2e-5`
  - 避免继续使用 `1e-4` 这类对第二阶段过于激进的配置
- 当前实验现象：
  - `stage2` 会让模型更贴近 `math220k` 分布
  - 但未必带来中立 benchmark 的提升，当前要重点监控 `GSM8K` 这类 transfer 下降
- `Mix-Long` 现已转成独立的 `stage1_mix_long` 实验线，不再作为这里的 `stage2` 入口

### SIMPO

- 使用 `TRL`
- 训练器口径：`CPOTrainer(loss_type="simpo")`
- MVP 数据构造口径：
  - 以 `stage1 best checkpoint` 为采样模型
  - 从 `stage1` 同源但未用题目中抽 `800` 条 query
  - 每题采样 `4` 个 responses
  - 优先构造 `correct > incorrect`，不足时回退到 `correct > correct`
  - 目标 `500` 对 pair，但不足不报错
  - 会额外从 full pair 中固定抽出 `150` 对 `pilot` 子集
- 默认训练口径：
  - `TRL CPOTrainer(loss_type="simpo")`
  - `4bit + PEFT`
  - `beta=2.0`、`gamma=1.0`
  - 支持 step checkpoint 与 resume

## 当前评估

- 这条路线已经形成了最小可运行骨架
- 但尚未证明相对于当前主线更值得继续投入
- 暂停原因主要是：
  - `stage2` 的收益不稳定
  - `SIMPO` 训练成本较高
  - 当前需要把实验注意力收敛到 `on-policy SFT`
