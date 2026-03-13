# State

## 当前结论

SFT 阶段的核心阻塞已经解除。
GRPO 阶段已进入真实调试，当前主路径已切换到官方 `Qwen3-1.7B-Base + QLoRA continuation + vLLM server`。

当前最可信的结论是：

1. `LoRA + 1.7B` 可以学会 `Final answer: \boxed{...}` 协议。
2. 真正影响协议学习的主因不是 LoRA 挂载失败，也不是训练脚本整体失效，而是**训练数据形态**。
3. 相比原始长解答，`short response / short-CoT` 明显更适合作为当前阶段的 SFT 数据。

## 关键证据

### 1. protocol-only tiny-overfit 已成功

`train_tiny_overfit_protocol` 的结果：

```json
{
  "format_success,none": 1.0,
  "parse_success,none": 1.0,
  "normalized_accuracy,none": 0.6875
}
```

这说明：

- 当前训练实现可以把 boxed 协议学稳
- 1.7B + LoRA 不是“根本学不会”
- 之前的失败不能再归因为纯实现错误

### 2. short-CoT tiny-overfit 也成功

`train_tiny_overfit_shortcot` 的结果：

```json
{
  "format_success,none": 1.0,
  "parse_success,none": 1.0,
  "normalized_accuracy,none": 0.6875
}
```

这说明：

- 不需要退化到 answer-only 才能学会协议
- 只要推理保持短且干净，模型就能稳定输出 boxed 最终答案

### 3. 原始长解答 tiny-overfit 明显更差

`train_tiny_overfit` 的结果约为：

```json
{
  "format_success,none": 0.6875,
  "parse_success,none": 0.75,
  "normalized_accuracy,none": 0.25
}
```

这说明：

- 长解答会显著削弱协议学习
- 问题不只是“数据脏”，而是长推理本身会稀释 boxed 监督

### 4. 2k short response 训练集带来当前最好的 SFT 基线

训练集工程验收：

```json
{
  "format_success,none": 0.81,
  "parse_success,none": 0.81,
  "normalized_accuracy,none": 0.08
}
```

`GSM8K dev200` 对照：

- base `flexible-extract = 0.47`
- SFT `flexible-extract = 0.625`

这说明：

- `2k / response 64~256` 这条数据路线已经能同时带来
  - 更稳定的 boxed 协议
  - 明确的外部开发集提升

## 当前判断

对进入 GRPO 来说，当前最合理的 SFT 冷启动基线是：

- NuminaMath
- 分层抽样
- `2000` 条
- `response token length = 64~256`

而不是：

- 原始长解答 `3000` 条全量直接训练

## 仍未完成的部分

1. `MATH500 test` 的正式结果还可以补跑。
2. 训练集上的 `normalized_accuracy` 仍然不高，说明这版 SFT 更偏向“协议稳定 + 可用 benchmark 提升”，而不是对训练题强记忆。
3. GRPO 最小代码骨架已经收敛到单一路径：

- 官方 `Qwen3-1.7B-Base`
- `4bit + LoRA adapter continuation`
- `TRL GRPOTrainer`
- `vLLM server` rollout

当前判断：

- 主路径不再继续维护 `unsloth` 专用训练入口
- 当前最优先工作从“继续 patch dtype”切换为“验证 vLLM server 权重同步与 rollout 稳定性”
- Phase 2 当前最优先工作不是继续改 reward，而是先把 GRPO 训练真实跑通

## GRPO 调试上下文

### 已完成的 GRPO 实现

当前仓库已经有以下 Phase 2 最小代码骨架：

- [scripts/prepare_grpo_data.py](/home/chy/code/active/rl/scripts/prepare_grpo_data.py)
- [scripts/train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py)
- [src/rl/grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)
- [configs/grpo.yaml](/home/chy/code/active/rl/configs/grpo.yaml)
- [tests/test_grpo.py](/home/chy/code/active/rl/tests/test_grpo.py)

目前已经确认：

- GRPO 不再卡在脚本入口或 `GRPOConfig` 初始化
- `trl/peft` 的 `warnings_issued` / `add_model_tags` 兼容问题已修复
- `BitsAndBytesConfig` 导入错误已修复
- `bf16/use_cpu` 初始化校验已修复
- 代码在受限环境里已经可以推进到 `trainer.train()`，不再是启动即崩

### 当前真实阻塞

在用户的真实 GPU 环境中，`train_grpo.py` 仍稳定报同一类错误：

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

错误栈稳定落在：

- `peft/tuners/lora/layer.py`
- `result = self.base_layer(x, *args, **kwargs)`
- 上层对应 `Qwen3` 的 `gate_proj/down_proj`

这说明报错发生在：

- generation forward
- LoRA 包装层进入 base linear 之前
- 不是 reward 函数，不是答案解析，不是数据字段缺失

### 已知环境与关键事实

- 基座模型：
  - `unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit`
- 当前 cold start adapter：
  - `outputs/sft-qwen3-1.7b`
- `adapter_config.json` 显示：
  - `auto_mapping.parent_library = transformers.models.qwen3.modeling_qwen3`
  - `unsloth_fixed = true`
- 当前 SFT adapter 中可训练 LoRA 参数实际是 `torch.float32`
- 当前 GRPO 路径为了规避前一轮错误，已经尝试：
  - 显式传 `quantization_config`
  - 显式传 `torch_dtype`
  - `autocast_adapter_dtype=False`
  - 将 `requires_grad=True` 的参数强制转到推断的 `compute_dtype`

但根据真实报错看，这些改动仍没有让整条计算路径完全统一 dtype。

### 需要 reviewer 理解的判断

当前最重要的判断不是“reward 有没有问题”，而是：

1. 这个 `unsloth-bnb-4bit` 基座快照是否会用自带 `quantization_config` 覆盖我们手动设置的 compute dtype
2. 当前 SFT adapter 是否适合直接挂到这条 HF 4bit GRPO 路径上继续训练
3. 是否应该放弃继续和这个 `unsloth` 量化快照硬兼容，改成：
   - 切换到标准 HF Qwen3 base
   - 自己完全控制 `BitsAndBytesConfig`
   - 再挂现有 LoRA adapter

### reviewer 建议优先检查

- [scripts/train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py)
  - `load_model_and_tokenizer()` 中的 `quantization_config`
  - `torch_dtype`
  - LoRA 参数 dtype 对齐逻辑
- 当前基座模型快照和 SFT adapter 的兼容性
- 是否需要在 GRPO 路径显式调用 `prepare_model_for_kbit_training`
- 是否应直接更换 GRPO 基座加载策略，而不是继续 patch 当前 `unsloth-bnb-4bit` 快照

## 对下一阶段的影响

Phase 2 不应再把“如何学会 boxed 协议”当成主问题。

进入 GRPO 时应默认：

1. boxed 协议已由 `2k short response` 这条 SFT 基线提供
2. 重点转向：
   - 模型加载与量化/精度兼容
   - reward 设计
   - rollout 稳定性
   - base vs SFT vs GRPO 的对照

   
