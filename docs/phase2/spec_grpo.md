# Phase 2 Trainer 规格

## 目标

实现一个最小但可解释的 `Unsloth GRPO` 训练基线，并优先保证单卡本地可调试与外部评测目标一致。

## 默认算法

- trainer：`Unsloth PatchFastRL("GRPO") + TRL GRPOTrainer`
- 当前配置：`dr_grpo`
- 备选：`dapo`

不推荐：

- 主干继续维护 `TRL + vLLM server/colocate` 双路径

## 训练形式

默认继续采用 QLoRA 风格训练：

- base model：`unsloth` 4bit 基座
- trainable params：LoRA adapter
- 不做全量微调
- rollout：优先不依赖独立 `vLLM server`

GRPO 在 Phase 1 的 SFT adapter 上继续训练：

- `SFT LoRA -> GRPO continuation`
- 优先直接从 `outputs/sft-qwen3-1.7b` 继续
- 训练数据默认切到 `GSM8K train(main)` 的短响应工件
- reward 默认使用单一 `combined_reward`
- 当前默认口径是 `relaxed correctness + weak strict boxed bonus`

## 当前实现

- 入口：[train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py)
- 配置：[grpo.yaml](/home/chy/code/active/rl/configs/grpo.yaml)
- 公共基线配置：`configs/grpo_base.yaml`
- tiny 调试配置：[grpo_tiny.yaml](/home/chy/code/active/rl/configs/grpo_tiny.yaml)
- GRPO 共享逻辑：[grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)
- 训练监控：默认 `train_log.jsonl`，当前主配置接 `wandb`

配置约定：

- 通过 `_base_` 继承公共配置
- 各实验 yaml 只保留差异字段
- reward 系数直接由 yaml 驱动

## tiny 调试路径

为单卡快速验证 reward 和 rollout 链路，仓库额外提供一条 tiny 调试路径：

- 训练配置：`configs/grpo_tiny.yaml`
- 训练数据：`data/grpo/train_grpo_gsm8k_tiny_short.jsonl`
- 目标：先验证训练有效，再决定是否跑完整 `2k` 配置

## 监控约定

- 默认仍写 `output_dir/train_log.jsonl`
- 可在配置中设置 `report_to: wandb` 启用 `wandb`
- 当前主配置实际使用 `wandb_mode: online`
- 如只做本地排障，仍推荐临时切回 `offline`

## 当前已知风险

当前最大的风险已经不是“能不能训练”，而是“训练把模型往哪里推”：

- strict boxed reward 与外部 `flexible-extract` 指标存在错位
- `format/length` 这类 shaping reward 可能把模型拉偏
- 最终 epoch checkpoint 不一定是外部 benchmark 最优点
- 已支持按 step 保存 checkpoint，并建议顺序扫 `dev200`

当前观察到的真实现象：

- `train_log.jsonl` 指标健康，`correct_rate` 高、`parse_fail_rate` 低
- 但 `GSM8K dev200` 上出现 `strict` 升、`flexible` 降

因此下一阶段优先级应是：

- 调整目标函数与约束
- 做中间 checkpoint 评测
- 继续分析 reward 与 eval 口径是否一致

## 非目标

- 主干不并存两套 GRPO trainer
- 主干不继续扩展 `TRL vLLM server/colocate` 兼容补丁
- v1 不实现 value model
