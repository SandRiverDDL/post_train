# Review

## 结论

当前问题的核心不是 `math-verify`、strict boxed 解析器或 `lm-eval` 本身，而是：

1. 当前 SFT 目标与后续 GRPO 冷启动目标不一致
2. 模型主要学到了“继续写长而杂的推理”，没有稳定学会 `Final answer: \boxed{...}` 协议
3. 训练集上的低 `format_success/parse_success` 主要是训练目标设计问题，不再应把生成长度截断视为主因

当前状态下，不建议直接进入 GRPO。否则 reward 解析会长期不稳定。

## 直接证据

### 1. 训练集工程验收结果已经说明格式没有学稳

用户给出的 `train_sft` strict 结果：

```json
{
  "alias": "train_sft",
  "format_success,none": 0.3,
  "parse_success,none": 0.3,
  "normalized_accuracy,none": 0.02
}
```

这说明：

- 只有约 `30%` 的样本输出了 strict boxed 格式
- 只有约 `30%` 的样本能被 strict 提取出答案
- 在训练分布上重新生成时，正确率也只有约 `2%`

这组信号应视为真实问题，而不是展示层误报。

后续用户已将训练改为 response-only loss 并重训一次，结果变为：

```json
{
  "format_success,none": 0.32,
  "parse_success,none": 0.32,
  "normalized_accuracy,none": 0.04
}
```

这说明：

- response-only loss 方向是对的
- 但它只带来了小幅改善
- 当前问题不能只归结为“loss 没 mask”或“样本太长”
- 仍然存在更关键的 prompt / target 结构错配

### 2. 评测生成长度不是当前主因

配置见 [configs/sft.yaml](/home/chy/code/active/rl/configs/sft.yaml)：

- `eval_max_new_tokens: 512`

对 `data/train_sft.jsonl` 的统计结果：

- 训练样本数：`3000`
- `solution` token 长度 `min/mean/p50/p90/p95/p99/max = 10 / 421.2 / 365 / 757 / 829 / 911 / 974`
- `solution > 512` 的样本有 `919` 条，占比 `30.6%`

这原本说明“截断”值得怀疑，但用户随后已实际将 `eval_max_new_tokens` 提高到 `1536`，结果仍然约为：

```json
{
  "format_success,none": 0.26,
  "parse_success,none": 0.26,
  "normalized_accuracy,none": 0.02
}
```

因此可以更新判断：

- 生成截断可能存在，但**不是主因**
- 即便给足生成长度，模型仍然没有稳定学会 boxed 协议
- 当前主问题仍然是训练目标本身

### 3. boxed 答案在训练序列里的位置过于靠后

当前数据构造见 [src/rl/data.py](/home/chy/code/active/rl/src/rl/data.py)：

- 原始 `raw_solution` 大体保留
- 只在尾部通过 `ensure_boxed_final_answer()` 补一个 `Final answer: \boxed{...}`

对 `data/train_sft.jsonl` 的统计结果：

- 整条训练文本长度 `min/mean/p50/p90/p95/p99/max = 256 / 526.0 / 472 / 868 / 944 / 1008 / 1024`
- `Final answer` 前缀平均出现在序列的 `96.8%` 位置
- `Final answer` 尾部平均只有 `15.1` tokens

这说明：

- 模型每个样本的大部分监督都在长推理正文
- 与 GRPO 解析最相关的 boxed 收尾只占末尾极小一段
- 这非常不利于学出稳定协议

### 4. 训练目标风格非常杂

训练数据中的 `solution` 首行高频分布包括：

- `## Solution`
- `## Solution.`
- `Solution`
- `Solution.`
- `$$`
- `1. **Understanding the Problem:**`
- `Solution 1`

这说明当前 target 风格并不统一。虽然尾部 boxed 行已统一，但正文风格仍然很杂，模型自然更容易学成“继续写各种长推理”，而不是“稳定地按协议收尾”。

进一步说，当前 prompt / completion 的结构本身也不够干净：

- prompt 固定是：

```text
Question:
...

Solution:
```

- 但 completion 的首行经常又是：
  - `## Solution`
  - `Solution.`
  - `Answer.`
  - `1. **Understanding the Problem:**`

这会形成类似：

```text
Question:
...

Solution:
## Solution
...
Final answer: \boxed{...}
```

这种重复结构不利于模型学出清晰、稳定的回答协议。

### 5. 当前 prompt contract 过弱，没有显式要求 boxed 协议

当前训练 prompt 只提供：

```text
Question:
...

Solution:
```

但没有明确告诉模型：

```text
最后一行必须严格写成：
Final answer: \boxed{...}
```

这会导致模型更容易把 boxed 当成“某些样本的一种写法”，而不是必须遵守的输出协议。

### 6. 当前训练实现原本大概率是在整段文本上计算 loss，而现在已部分修正

训练入口见 [scripts/train_sft.py](/home/chy/code/active/rl/scripts/train_sft.py)：

- [scripts/train_sft.py](/home/chy/code/active/rl/scripts/train_sft.py) 将 `Question + Solution` 直接拼成一个 `text`
- 然后把该字段直接交给 `SFTTrainer`
- 当前没有看到 completion-only mask，也没有 prompt / response 分段监督

原始实现中，训练大概率是在整段文本上做 next-token loss，包括：

- `Question:` 前缀
- 问题正文
- `Solution:` 前缀
- 长推理正文
- 最后的 boxed 收尾

这不代表“Question 部分最重要”，但意味着 boxed 收尾的监督占比被明显稀释了。

用户后续已经改成 response-only loss，并带来了小幅提升，因此这个方向应保留；但它不是唯一根因。

## 已基本排除的方向

### 1. 不是训练数据尾部格式清洗失败

根据当前 `check_sft_data.py` 验收结果，训练数据本身已经接近：

- `empty_final_answer=0`
- `boxed_rate` 接近 `1.0`
- `parse_success_rate` 接近 `1.0`
- `consistent_rate` 接近 `1.0`

因此问题不在“训练集里没有 boxed”，而在“模型没有稳定学会生成它”。

### 2. 不是 LoRA 完全没生效

从已有样本看，LoRA 模型在部分样本上能输出：

```text
Final answer: \boxed{...}
```

所以不能简单归因为“adapter 没加载”。

## 根因判断

当前最合理的根因是以下四项叠加：

1. 数据构造只统一了尾部 boxed 协议，但保留了大量原始长解答，导致 boxed 监督信号过弱
2. prompt 没有显式声明 boxed 协议是必须遵守的输出 contract
3. prompt 与 completion 的结构不够干净，存在 `Solution:` / `## Solution` 一类重复或冲突
4. 当前使用的数据形态更适合“长解答模仿”，不适合作为“GRPO 冷启动协议学习”的唯一数据来源

换句话说，当前 SFT 更像是在做“长解答模仿”，而不是“GRPO 冷启动协议学习”。

## 对 implementer 的建议

### 优先级 1：不要把“原始长 solution + 尾部 boxed”作为 GRPO 冷启动唯一 SFT 数据

建议至少做一个更适合协议学习的版本：

- `answer-only`
  - target 只有 `Final answer: \boxed{...}`
- `short-CoT`
  - 保留极短推理，再接 `Final answer: \boxed{...}`
- `mixed`
  - 一部分 full solution
  - 一部分 short-CoT / answer-only

如果目标是尽快进入 GRPO，推荐优先 `answer-only` 或 `short-CoT`。

这里的建议不是第一步就彻底更换主数据集，而是：

- 优先从现有 Numina 数据派生一个“短、干净、显式 boxed 协议”的冷启动版本
- 只有当这条路线仍然学不稳协议时，再考虑更换主数据源

### 优先级 2：训练时只对回答部分计算 loss

这条的含义不是“当前一定错误地只学 Question”，而是：

- 不要让 `Question:`、问题正文、`Solution:` 前缀和回答正文拥有同等 loss 权重
- 更合理的是只对 response token 计算 loss

实现方向：

- 将训练样本拆成 `prompt` 与 `response`
- 使用 completion-only masking 或等价机制
- 保证 loss 主要落在模型生成的回答部分

这会显著增加 boxed 协议在训练目标中的相对权重。

### 优先级 3：强化 prompt contract，并清理 completion 头部结构

建议：

1. 在 prompt 中显式加入 boxed 协议要求，例如：

```text
Question:
{question}

Solution:
请给出必要推理，最后一行必须严格写成：
Final answer: \boxed{...}
```

2. 在数据预处理中移除 completion 头部重复结构，例如：
   - `## Solution`
   - `Solution.`
   - `Answer.`

目标不是清洗掉推理，而是避免出现：

```text
Solution:
## Solution
```

这类对模型不友好的重复模式。

### 优先级 4：新增 tiny-overfit sanity check

这是当前最应该尽快补上的诊断工具。

建议：

1. 取 `8~16` 条训练样本
2. 在极小数据上训练到明显过拟合
3. 直接看模型对这批样本的生成

判定标准：

- 如果 tiny-overfit 后仍不能稳定输出 `Final answer: \boxed{...}`
- 则问题不是数据集泛化难度，而是 pipeline 仍存在硬错误或严重结构错配

这一步应优先于继续大规模调超参数。

### 优先级 5：把 Phase 1 SFT 目标拆成两个阶段

推荐顺序：

1. 先训练一个“协议稳定版”模型
   - 核心目标：稳定输出 `Final answer: \boxed{...}`
2. 再考虑加入更长推理数据
   - 核心目标：在不破坏 boxed 协议的前提下补充推理能力

对于 GRPO 冷启动，第一阶段更重要。

### 优先级 6：重新审视训练长度策略

当前长度策略见 [scripts/prepare_data.py](/home/chy/code/active/rl/scripts/prepare_data.py) 和 [configs/sft.yaml](/home/chy/code/active/rl/configs/sft.yaml)：

- 训练长度过滤：`256 ~ 1024`
- 训练 `max_seq_length: 1024`

建议尝试：

- 降低训练样本上限，例如 `768` 或更低
- 单独导出一个“短样本协议版”训练集
- 把长样本留到第二阶段或单独实验

原因不是 1024 一定错误，而是当前 boxed 尾部长期贴在序列最末端，对协议学习不友好。

## 建议的实施顺序

1. 新增 tiny-overfit sanity check
2. 修改 prompt contract，显式要求 boxed 协议
3. 清理 completion 头部的重复结构
4. 新增一个短目标数据集版本
5. 继续保留 response-only loss
6. 重新训练一个最小协议模型
7. 再对比：
   - train strict format
   - GSM8K dev200 strict / relaxed
   - 少量 MATH500 spot check
8. 只有当 boxed 协议稳定后，再进入 GRPO

## 一句话总结

当前问题不是“数据清洗坏了”，而是“你把 NuminaMath 当成长解答模仿数据用了，但 GRPO 冷启动更需要协议稳定数据”。实现上应优先修正：

- tiny-overfit 诊断
- prompt contract
- completion 结构清理
- 训练目标长度与风格
- response-only loss
- 冷启动数据形态

而不是先继续调 LoRA 超参数或直接进入 GRPO。
