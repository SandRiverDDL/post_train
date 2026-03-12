# State

## 当前结论

SFT 阶段的核心阻塞已经解除。

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
3. 进入 GRPO 前，reward contract 和 rollout 配置仍需单独设计。

## 对下一阶段的影响

Phase 2 不应再把“如何学会 boxed 协议”当成主问题。

进入 GRPO 时应默认：

1. boxed 协议已由 `2k short response` 这条 SFT 基线提供
2. 重点转向：
   - reward 设计
   - rollout 稳定性
   - base vs SFT vs GRPO 的对照

   
