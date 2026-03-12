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
- 配置：[grpo.yaml](/home/chy/code/active/rl/configs/grpo.yaml)
- GRPO 共享逻辑：[grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)

## 当前已知风险

当前最主要风险不是 reward，而是模型加载后的 dtype / 量化兼容问题。

真实 GPU 环境已观察到：

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

当前判断：

- `unsloth` 4bit 基座快照自带量化配置
- 该配置可能覆盖手动指定的 compute dtype
- HF + PEFT + TRL GRPO 路径中的 dtype 仍未完全统一

## reviewer 建议优先检查

- [train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py) 中：
  - `quantization_config`
  - `torch_dtype`
  - LoRA 参数 dtype 对齐
- 当前 `unsloth-bnb-4bit` 基座与 SFT adapter 的兼容性
- 是否需要改用标准 HF base 再挂现有 adapter

## 非目标

- v1 不实现 HES
- v1 不实现 value model
- v1 不并存多套 GRPO trainer
