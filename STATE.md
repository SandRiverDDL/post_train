# State

## 当前阶段

- `SFT` 阶段已经完成，当前只作为冷启动背景，不再是主阻塞。
- 当前主任务是 `Phase 2 / GRPO`，主干路线为 `Unsloth GRPO + 单卡优先 + QLoRA continuation`。
- `TRL GRPO` 的排障代码与历史结论保留在独立分支，主干不再继续维护双实现。
- 评测配置已从 `configs/sft.yaml` 拆出，当前默认评测配置见 `configs/eval.yaml`。

## 当前有效基线

- 开发模型：`Qwen3-1.7B-Base`
- 当前冷启动模型：`outputs/sft-qwen3-1.7b`
- 当前 GRPO 基座：`unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit` 本地 snapshot
- 当前主训练配置见 [grpo.yaml](/home/chy/code/active/rl/configs/grpo.yaml)
- 当前主训练数据：`data/grpo/train_grpo_gsm8k_1k_middiff.jsonl`

SFT 结论只保留两点：

1. `1.7B + LoRA` 可以稳定学会 `Final answer: \boxed{...}` 协议。
2. 当前 SFT 基线在 `GSM8K dev200` 上优于 base，可作为 GRPO 冷启动。

## 当前 GRPO 实现状态

已完成：

- [scripts/train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py) 可在真实 GPU 环境启动并完成一轮训练。
- reward 已收敛为单一 `combined_reward`，实现见 [grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)。
- `GRPO` 配置已支持 `_base_` 继承，实验 yaml 可只保留差异字段。
- `wandb` 与 `train_log.jsonl` 监控已接通。
- 候选集离线难度打分与子集选择脚本已加入：
  - [score_grpo_candidates.py](/home/chy/code/active/rl/scripts/score_grpo_candidates.py)
  - [select_grpo_scored_subset.py](/home/chy/code/active/rl/scripts/select_grpo_scored_subset.py)

当前 reward 口径：

- `correct_relaxed: +1.0`
- `parse fail: -0.5`

这些 reward 项系数已经暴露在 yaml 中，可直接做实验切换。

## 当前最重要的问题

当前问题不是“GRPO 跑不起来”，而是：

- **GRPO 训练指标很好，但外部 benchmark 没有稳定变好，且 `flexible-extract` 明显退化。**

最近一轮 `1K middiff` 训练后的 `GSM8K dev200` 结果：

```json
{
  "exact_match,strict-match": 0.67,
  "exact_match,flexible-extract": 0.57
}
```

对照：

- base：`strict 0.555 / flexible 0.47`
- SFT：`strict 0.645 / flexible 0.665`
- 当前 GRPO：`strict 0.67 / flexible 0.57`

当前判断：

1. `strict-match` 有小幅提升，但 `flexible-extract` 明显下降。
2. 这说明 RL 并非“完全无效”，而是把模型往**更适配 strict boxed reward** 的方向推了。
3. 当前问题更像是**目标函数错位 / 策略漂移**，不是数据解析失败或训练崩溃。
4. 当前已切到“reward 系数外置 + base/override 配置”的下一轮验证方案。

## 关键证据

最近一轮 [train_log.jsonl](/home/chy/code/active/rl/outputs/grpo-qwen3-1.7b/train_log.jsonl) 显示：

- `rewards/correct_rate` 均值约 `0.733`
- `rewards/parse_fail_rate` 均值约 `0.029`
- `rewards/format_rate` 均值约 `0.964`
- `completions/clipped_ratio` 均值约 `0.045`

说明：

- rollout 质量已经明显好于之前的 0.5K 随机 short 实验
- 当前不是 reward 稀疏，也不是 boxed 协议不稳定
- 问题更可能出在**reward 优化目标与外部评测目标不一致**

另外当前配置里：

- `loss_type: dr_grpo`
- `beta: 0.02`
- `temperature: 0.7`
- `top_p: 0.95`

当前最需要验证的是：

- 去掉 `strict_boxed_bonus` 是否能止住 `flexible-extract` 下滑
- 去掉 `length penalty` 是否能止住 `flexible-extract` 下滑
- 中间 checkpoint 是否优于最终 epoch checkpoint

## 当前优先级排序

1. 分析并修复“`strict` 升、`flexible` 降”的目标错位问题。
2. 用 reward 系数消融验证 `format/length` 是否在拉偏策略。
3. 继续使用离线难度筛选的数据路线，而不是回退到随机 `GSM8K short`。
4. `MATH500 test` 不是当前最高优先级；在 GRPO 指标方向明确前不应抢跑。

## 需要下一个会话优先理解的事实

1. 主干已经不是“修训练脚本能否启动”的阶段，而是“训练目标是否把模型拉偏”的阶段。
2. 当前最核心的问题是 **GRPO 优化目标和 benchmark 目标不一致**。
3. 当前不应再把主要精力放在：
   - `TRL` 兼容
   - boxed 协议学习
   - reward 稀疏
4. 当前应重点检查：
   - `format/strict bonus` 是否在拉偏策略
   - `length penalty` 是否在拉偏策略
   - 中间 checkpoint 是否优于最终点

## 文档口径

- 当前会话应优先读取 `STATE.md` 与 `TASK.md` 作为事实来源。
- `docs/phase1_mvp.md` 与 `docs/phase1_cold_start_plan.md` 仅保留历史背景，不再作为当前 Phase 2 的主执行文档。
