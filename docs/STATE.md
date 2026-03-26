# State

## 当前阶段

当前仓库已切到“两阶段 `SFT + SIMPO`”主线，正在从最小可运行骨架走向真实实验验证。

## 当前有效基线

- 代码主包：`src/post_train/`
- 薄入口：`prepare_stage1_data.py`、`check_sft_data.py`、`train_sft.py`、`eval_model.py`、`train_simpo.py`
- 当前开发模型：`Qwen/Qwen2.5-Math-1.5B`
- 当前 `stage1` 数据口径：`UWNSL/MATH_training_split_short_cot`，首版规模 `2000`
- 当前 `SIMPO` 口径：`TRL CPOTrainer(loss_type="simpo")`

## 当前最重要的问题

1. 还缺少目标机器上的真实训练验证。
2. `stage2` 的数据来源仍是抽象接口，尚未固定生成方案。
3. preference 数据构造尚未纳入仓库主线。
4. 远程数据下载在当前环境里不稳定。

## 当前优先级

1. 在目标机器上验证 `stage1` 数据准备、训练、评测链路。
2. 固定 `stage2` 数据来源。
3. 固定 preference 数据来源与构造方式。
4. 基于真实实验结果继续收紧 `SPEC` 与配置。

## 文档口径

- 稳定规则看 `AGENTS.md`
- 当前阶段设计看 `docs/SPEC.md`
- 结构分层看 `docs/ARCHITECTURE.md`
- 当前状态看 `docs/STATE.md`
