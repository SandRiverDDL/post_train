# SPEC
这是已暂停路线，不是当前主线
## 目标

当前阶段的目标是建立一个最小可运行的两阶段 `SFT + SIMPO` 数学后训练闭环。

本阶段关注点：

1. 跑通 `stage1 SFT`
2. 跑通 `stage2 SFT`
3. 跑通 `SIMPO`
4. 用统一评测链路比较三个阶段结果

本阶段不是为了追求最终最优成绩，而是为了证明每个阶段都能被独立执行、评测和比较。

## 当前阶段定义

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
- 当前 MVP 配比：`50%` 随机 `stage1_train` + `35%` `math220k_short(<768)` + `15%` `math220k_long(<1380)`
- 当前 `math220k` 绑定规则：
  - 优先使用 `clean_problem`
  - `question_type == "MCQ"` 直接过滤
  - 尾部标准化为单个 canonical `\boxed{...}`，避免重复 boxed
- 后续可以把去重、去污染、多数据集配额控制接入这一阶段。
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

### SIMPO

- 当前使用 `TRL`。
- 当前训练器口径：`CPOTrainer(loss_type="simpo")`
- 当前 MVP 数据构造口径：
  - 以 `stage1 best checkpoint` 为采样模型
  - 从 `stage1` 同源但未用题目中抽 `800` 条 query
  - 每题采样 `4` 个 responses
  - 优先构造 `correct > incorrect`，不足时回退到 `correct > correct`
  - 目标 `500` 对 pair，但不足不报错
  - 会额外从 full pair 中固定抽出 `150` 对 `pilot` 子集
- 当前默认训练口径：
  - `TRL CPOTrainer(loss_type="simpo")`
  - `4bit + PEFT`
  - `beta=2.0`、`gamma=1.0`
  - 支持 step checkpoint 与 resume

## 当前模型与评测口径

- 开发模型：`Qwen/Qwen2.5-Math-1.5B`
- 正式 benchmark：`GSM8K`、`MATH-500`
- 正式评测支持两条链路：
  - `vllm_raw`：直接用 `vLLM` 生成，本地 parser + `math-verify` 判定
  - `lm_eval`：`lm-eval-harness` 负责编排推理，本地 parser + `math-verify` 判定
- 当前默认评测链路：`vllm_raw`

当前核心指标：

- `boxed_rate`
- `parse_success_rate`
- `normalized_accuracy`
- `avg_output_tokens`

## Prompt 原则

- SFT 使用更强约束的 prompt，优先学稳格式。
- 评测使用更短、接近公开基线的 prompt。
- 统一要求答案中必须包含 `\boxed{...}`。

## 当前边界

本阶段当前不做：

- 自动 early stopping
- 复杂题目选择策略
- 复杂轨迹筛选策略
- 多种偏好训练方法并存
- 为未来阶段预先设计复杂抽象

## 参数口径

- 当前推荐 prompt、LoRA、SIMPO 超参会在配置中维护。
- `configs/*.yaml` 是可执行参数真源。
- `SPEC.md` 只保留本阶段推荐口径与原则，不逐项同步所有细参数。
