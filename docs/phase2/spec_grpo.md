# Phase 2 Trainer 规格

## 目标

实现一个最小但不落后的 `Unsloth GRPO` 训练基线，并优先保证单卡本地可调试。

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

## 当前实现

- 入口：[train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py)
- 配置：[grpo.yaml](/home/chy/code/active/rl/configs/grpo.yaml)
- tiny 调试配置：[grpo_tiny.yaml](/home/chy/code/active/rl/configs/grpo_tiny.yaml)
- GRPO 共享逻辑：[grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)
- 训练监控：默认 `train_log.jsonl`，可选 `wandb offline`

## tiny 调试路径

为单卡快速验证 reward 和 rollout 链路，仓库额外提供一条 tiny 调试路径：

- 训练配置：`configs/grpo_tiny.yaml`
- 训练数据：`data/grpo/train_grpo_gsm8k_tiny_short.jsonl`
- 目标：先验证训练有效，再决定是否跑完整 `2k` 配置

## 监控约定

- 默认仍写 `output_dir/train_log.jsonl`
- 可在配置中设置 `report_to: wandb` 启用 `wandb`
- 当前推荐 `wandb_mode: offline`

## 当前已知风险

当前最大的风险不再是 `trl/vllm` 兼容，而是：

- `Unsloth GRPO` 在当前单卡显存上的真实稳定性
- 现有 SFT adapter 与 `unsloth` 基座继续训练时的兼容性
- 跑通后日志、reward、工件口径是否与当前 Phase 2 约定保持一致

## 非目标

- 主干不并存两套 GRPO trainer
- 主干不继续扩展 `TRL vLLM server/colocate` 兼容补丁
- v1 不实现 value model
