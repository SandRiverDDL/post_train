# State

## 当前阶段

当前仓库已切到“两阶段 `SFT + SIMPO`”主线，正在从最小可运行骨架走向真实实验验证。

## 当前有效基线

- 代码主包：`src/post_train/`
- 薄入口：`prepare_stage1_data.py`、`check_sft_data.py`、`train_sft.py`、`eval_model.py`、`prepare_simpo_data.py`、`train_simpo.py`
- 当前开发模型：`Qwen/Qwen2.5-Math-1.5B`
- 当前 `stage1` 数据口径：`UWNSL/MATH_training_split_short_cot`，首版规模 `2000`
- 当前 `SIMPO` 口径：`TRL CPOTrainer(loss_type="simpo")`

## 当前最重要的问题

1. 还缺少目标机器上的真实训练验证。
2. `stage2` 的去重与去污染仍未纳入 MVP。
3. `stage2` 的真实收益仍未被稳定验证；当前观测到它更贴近 `math220k`，但可能伤害 `GSM8K` transfer。
4. `SIMPO` 已有 MVP 数据构造链路，但尚未在目标机器上完成真实采样与训练验证。
5. `SIMPO` 当前训练成本仍高，当前优先验证 4bit + pilot + resume 是否能显著降本。
6. 远程数据下载在当前环境里不稳定。
7. `stage2` target dev 的 profile 构造链路刚建立，仍需在目标机器上实际生成并验证。

## 当前优先级

1. 在目标机器上验证 `stage1` 数据准备、训练、评测链路。
2. 重新生成 `stage2` 数据并确认 report 中不存在双 boxed 尾部。
3. 用较低学习率继续验证 `stage2`，优先观察 `math220k_dev_main150` 与 `GSM8K` 的此消彼长。
4. 在目标机器上验证 `stage2` 清洗与筛选管线。
5. 在目标机器上验证 `prepare_simpo_data -> train_simpo` 链路。
6. 在目标机器上验证 `SIMPO` 的 pilot/full 两档训练与 resume。
7. 基于真实实验结果继续收紧 `SPEC` 与配置。

## 当前 Stage1 口径

- 训练预算：`2 epoch`
- checkpoint 保存：每 `25` step 存一次
- 完成判定：训练结束后统一在冻结 `dev200` 上评测全部 checkpoints
- best 选择指标：`normalized_accuracy`

## 文档口径

- 稳定规则看 `AGENTS.md`
- 当前阶段设计看 `docs/SPEC.md`
- 结构分层看 `docs/ARCHITECTURE.md`
- 当前状态看 `docs/STATE.md`
