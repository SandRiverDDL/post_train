# SPEC

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

### Stage2 SFT

- 定义为“在另一份 SFT 数据上继续训练一次”。
- 当前不把它绑定到自蒸馏、错题重采样或固定外部数据源。
- 后续可以把题目选择、轨迹选择、数据生成方式接入这一阶段。

### SIMPO

- 当前使用 `TRL`。
- 当前训练器口径：`CPOTrainer(loss_type="simpo")`
- 当前只要求能消费已有 preference 数据并完成训练闭环。
- preference 数据如何构造不在本阶段写死。

## 当前模型与评测口径

- 开发模型：`Qwen/Qwen2.5-Math-1.5B`
- 正式 benchmark：`GSM8K`、`MATH-500`
- 正式评测方式：`lm-eval-harness` 负责编排推理，本地 parser + `math-verify` 负责提取与等价判定

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
