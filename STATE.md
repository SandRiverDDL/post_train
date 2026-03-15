# State

## 当前结论

SFT 阶段的核心阻塞已经解除。
GRPO 阶段已进入真实调试，当前主路径已切换到 `Unsloth GRPO + 单卡优先`。

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

对当前项目来说，需要区分 `SFT` 和 `GRPO` 两条数据路线：

- `SFT` 冷启动基线仍然是：
  - NuminaMath
  - 分层抽样
  - `2000` 条
  - `response token length = 64~256`
- `GRPO` 主线训练数据已经切到：
  - `GSM8K train(main)`
  - full: `2000` 条
  - tiny: `64` 条
  - `response token length <= 128`

原因不是 boxed 协议问题，而是 `NuminaMath` 对 `1.7B/4B` 冷启动 policy 来说 reward 过于稀疏。

## 仍未完成的部分

1. `MATH500 test` 的正式结果还可以补跑。
2. 训练集上的 `normalized_accuracy` 仍然不高，说明这版 SFT 更偏向“协议稳定 + 可用 benchmark 提升”，而不是对训练题强记忆。
3. GRPO 主线策略已经改变：

- `TRL GRPO` 调试代码留在独立分支
- 主干切到 `Unsloth GRPO`
- 优先目标是单卡可调试，而不是继续兼容 `TRL + vLLM`

当前判断：

- 主干不再继续扩展 `TRL/vLLM` 兼容补丁
- 当前最优先工作从“修 TRL 路径”切换为“验证 Unsloth GRPO 单卡可运行性”
- Phase 2 的 reward 主口径已改为单一 `combined_reward`
- Phase 2 当前最优先工作仍是把 GRPO 训练真实跑通

## GRPO 调试上下文

### 已完成的 GRPO 实现

当前仓库已经有以下 Phase 2 最小代码骨架：

- [scripts/prepare_grpo_data.py](/home/chy/code/active/rl/scripts/prepare_grpo_data.py)
- [scripts/train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py)
- [src/rl/grpo.py](/home/chy/code/active/rl/src/rl/grpo.py)
- [configs/grpo.yaml](/home/chy/code/active/rl/configs/grpo.yaml)
- [tests/test_grpo.py](/home/chy/code/active/rl/tests/test_grpo.py)

目前已经确认：

- 数据准备、prompt、reward contract、训练日志落盘都已经具备
- `TRL` 路径的排障结论已经保存在独立分支
- 主干训练入口已经切换为 `Unsloth GRPO`

### 当前真实阻塞

当前阻塞已经从 `TRL/vLLM` 兼容问题切换为：

- `Unsloth GRPO` 在单卡真实 GPU 环境中的初始化与前几个 step 是否稳定
- 当前 `outputs/sft-qwen3-1.7b` 冷启动 adapter 是否能直接作为 `Unsloth GRPO` 的继续训练输入
- 跑通后日志、reward、配置快照是否仍符合现有工程约定

当前新增的调试策略：

- 保留完整配置 `configs/grpo.yaml`
- 额外提供单卡调试配置 `configs/grpo_tiny.yaml`
- GRPO 训练数据从 `NuminaMath` 切到 `GSM8K train(main)`，以降低 `1.7B` 上的 reward 稀疏问题
- full / tiny 都通过 `prepare_grpo_data.py` 生成独立的 `gsm8k` GRPO 工件
- 当前主配置的 loss 已从 `dapo` 切到 `dr_grpo`
- reward 已从 `correctness + parse + format` 三路加权，收敛为单一组合 reward：
  - `correct: +1.0`
  - `wrong: -0.2`
  - `parse fail: -0.2`
  - `format: +0.05`
  - `length: -1e-4 * cleaned_completion_tokens`
- GRPO 训练监控已支持可选 `wandb offline`，同时保留 `train_log.jsonl` 作为本地审计日志

### 已知环境与关键事实

- 当前 SFT 冷启动产物仍是：
  - `outputs/sft-qwen3-1.7b`
- 该 adapter 的 `base_model_name_or_path` 指向：
  - `unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit` 本地 snapshot
- 因此主干切回 `Unsloth` 路线后，冷启动资产与基座重新对齐
- 当前主线目标是单卡可调试，不再优先维护 `TRL + vLLM` 标准生态兼容
- 当前新增的数据筛选方向是：
  - 用 `vLLM` 对 `GSM8K short` 候选集做离线多次采样打分
  - 以 `correct_rate` 和 `parse_rate` 选择更适合 `1.7B` RL 的中等难度题
  - 打分结果全量持久化，后续抽样和控量走独立脚本，而不是重复采样

### 需要 reviewer 理解的判断

当前最重要的判断不是“是否继续救 `TRL` 路径”，而是：

1. `Unsloth GRPO` 是否能在单卡上稳定替代当前主线
2. 当前 SFT adapter 是否能直接继续训练，而不需要重新做 cold start
3. 主干是否能只保留一套 `Unsloth GRPO` 实现，而把 `TRL` 完全留在分支中

### reviewer 建议优先检查

- [scripts/train_grpo.py](/home/chy/code/active/rl/scripts/train_grpo.py)
  - `PatchFastRL("GRPO")`
  - 当前 SFT adapter 的加载路径
  - `Unsloth` 单卡参数设置
- 当前 `outputs/sft-qwen3-1.7b` 和 `unsloth` 基座的兼容性
- 跑通后的日志与工件是否仍符合现有 Phase 2 约定

## 对下一阶段的影响

Phase 2 不应再把“如何学会 boxed 协议”当成主问题。

进入 GRPO 时应默认：

1. boxed 协议已由 `2k short response` 这条 SFT 基线提供
2. 重点转向：
   - 单卡训练可运行性
   - reward 设计
   - rollout 稳定性
   - base vs SFT vs GRPO 的对照

   
