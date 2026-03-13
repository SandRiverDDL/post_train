# Phase 2 Trainer 规格

## 目标

实现一个最小但不落后的 GRPO 训练基线。

## 默认算法

- trainer：`TRL GRPOTrainer`
- loss：`dapo`
- 备选：`dr_grpo`

不推荐：

- plain `grpo` 作为默认起点

## 训练形式

默认继续采用 QLoRA 风格训练：

- base model：4bit 量化加载
- trainable params：LoRA adapter
- 不做全量微调
- rollout backend：`vLLM server`

GRPO 在 Phase 1 的 SFT adapter 上继续训练：

- `SFT LoRA -> GRPO continuation`

## 接口边界

### v1 必须实现

- config 加载
- cold-start model 加载
- `GRPOConfig` 构建
- trainer 启动
- config / reward / train log 落盘

### 预留但不实现

- `loss_type: dapo | dr_grpo`
- `token_mask_strategy`
- `kl_control`
- `sft_mix_ratio`

其中：

- `token_mask_strategy` 预留给 HES / entropy-based 方法

## 当前实现

- 入口：[train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py)
- vLLM server 入口：[serve_grpo_vllm.py](/home/chy/code/active/rl/scripts/serve_grpo_vllm.py)
- 配置：[grpo.yaml](/home/chy/code/active/rl/configs/grpo.yaml)
- GRPO 共享逻辑：[grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)

## 当前已知风险

当前最主要风险不是 reward，而是 `TRL 0.24.x` 与当前 `vLLM 0.16.x` 的接口差异，以及真实 GPU 环境中的 server 权重同步稳定性。

当前实现约束：

- 基座默认走官方 `Qwen3-1.7B-Base`
- 不再维护 `unsloth` 的单独训练入口
- `vLLM` 当前只支持 `server` 模式，不支持 `colocate`

## reviewer 建议优先检查

- [train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py) 中：
  - `quantization_config`
  - `attn_implementation`
  - `vLLM server` 连接参数
- `trl` 导入阶段的 `vllm.sampling_params` 兼容
- `PeftModel` continuation 在 `vLLM` 权重同步后的稳定性

## 非目标

- v1 不实现 HES
- v1 不实现 value model
- v1 不并存多套 GRPO trainer
