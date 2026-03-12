# Phase 1：SFT + GSM8K / MATH500 Eval

## 目标

构建一个可信的最小闭环：

`NuminaMath SFT -> GSM8K dev200 -> MATH500 test`

本阶段证明两件事：

1. 数据准备、训练、评测链路可稳定跑通。
2. SFT 模型在外部开发集上相对 base 有可观测提升。

本阶段不宣称“通用数学能力已经显著提升”。

## 模型

- 开发模型：`unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit`
- 最终实验模型：`unsloth/Qwen3-8B-Base-bnb-4bit`

Phase 1 仅使用 1.7B 模型。

## 数据设计

### Train

训练集使用 `nlile/NuminaMath-1.5-RL-Verifiable`。

- 固定采样：`3000`
- `seed`：`42`
- 长度过滤：`256 ~ 1024` token
- 长度基于最终训练文本计算，而不是原始字段裸长度

按 `problem_type` 分层抽样：

- Algebra：750
- Geometry：600
- Number Theory：500
- Combinatorics：450
- Logic and Puzzles：250
- Calculus：180
- Inequalities：180
- Other：90

实现要求：

1. `problem_type` 匹配允许做鲁棒归一化，不依赖完全相同字符串。
2. 优先使用 NuminaMath 原生 `answer` 字段作为 `final_answer`。
3. 若缺少推理文本但存在答案，仍需构造最小合法样本。

### Dev

开发集使用 `gsm8k`，固定 `200` 条。

- 可做长度过滤
- 用于日常调参与快速对比
- 不作为最终 benchmark 汇报口径
- 一旦生成 `data/eval/gsm8k_dev200.jsonl`，后续实验应复用该工件，不再随 tokenizer 或长度阈值变化重新采样
- `gsm8k_dev200.jsonl` 的主要作用是冻结样本索引；答案解析和 metric 仍然交给 `lm-eval` 原生 task

### Test

测试集使用 `HuggingFaceH4/MATH-500` 全量。

约束：

1. 不允许按长度裁剪。
2. 只允许字段映射和答案格式标准化。
3. 只在阶段收尾时跑正式结果。
4. 正式评测时使用 `lm-eval` 原生 `hendrycks_math500` task，而不是自写答案解析

## 数据格式

训练记录统一为：

```json
{
  "id": "...",
  "question": "...",
  "solution": "...",
  "final_answer": "...",
  "source": "numinamath|gsm8k|math500"
}
```

统一约束：

```text
Final answer: \boxed{...}
```

训练数据中的 `solution` 必须以该格式结束。

说明：

1. 统一 boxed 协议主要服务于训练和后续 RL。
2. benchmark 正确率评测不再依赖自写 `final_answer` 解析，而是使用 `lm-eval` 原生 task。

## 关键脚本

### 数据准备

入口：`scripts/prepare_data.py`

输出：

- `data/train_sft.jsonl`
- `data/eval/gsm8k_dev200.jsonl`
- `data/eval/math500_test.jsonl`

要求：

1. 训练和开发集按最终格式文本做长度过滤。
2. 测试集不做长度过滤。
3. 每个 split 打印至少 3 条样本用于人工抽查。
4. `dev/test` 导出的本地文件主要是冻结样本工件，不再作为 benchmark evaluator 的真值来源。

### 数据验收

入口：`scripts/check_sft_data.py`

要求：

1. 在每次开始 SFT 前运行。
2. 至少检查：
   - `boxed_rate`
   - `parse_success_rate`
   - `consistent_rate`
   - `empty_final_answer`
   - `problem_type_counts`
3. 如果 `empty_final_answer > 0`，或 `boxed_rate / parse_success_rate / consistent_rate` 明显异常，则停止训练，先修数据。

### 训练

入口：`scripts/train_sft.py`

要求：

1. 使用 `unsloth + LoRA + TRL SFTTrainer`
2. 使用共享格式化逻辑构造训练文本
3. 训练前打印至少 3 条样本
4. 输出 adapter 到本地目录

补充说明：

- Phase 1 当前已确认需要把 SFT 拆成 protocol cold start 与 short-CoT 两阶段
- 具体实施方案见 [`docs/phase1_cold_start_plan.md`](/home/chy/code/active/rl/docs/phase1_cold_start_plan.md)

当前默认配置见 [`configs/sft.yaml`](/home/chy/code/active/rl/configs/sft.yaml)。

### 评测

正式入口：`scripts/eval_dataset.py`

要求：

1. 统一通过 `lm-evaluation-harness` 执行推理。
2. `GSM8K dev200` 使用原生 `gsm8k_cot_zeroshot` task，并通过冻结样本索引保证可重复。
3. `MATH500 test` 使用原生 `hendrycks_math500` task。
4. 支持通过 `batch_size` 打开批量并发评测。
5. 评测时至少打印 3 条样本。

说明：

- benchmark 正确率由 `lm-eval` 原生 task 负责，避免自写 evaluator 的 dirty work。
- `GSM8K dev200` 和 `MATH500 test` 使用同一个评测入口，靠 `--dataset` 切换。
- 如果后续还要统计 boxed 格式合规率，应单独作为 format-check 指标，不与 benchmark 正确率耦合。
- LoRA/SFT 产物在进入 GRPO 前，还必须在 `data/train_sft.jsonl` 上跑一次工程验收；这一步继续复用同一个评测入口，但指标解释与 benchmark 不同。

### GRPO 前验收

进入 GRPO 前至少完成三类检查：

1. `train` 工程验收：
   - 在 `data/train_sft.jsonl` 上评测 `format_success / parse_success / normalized_accuracy`
   - 目标是确认 `Final answer: \boxed{...}` 协议已经学会，reward 可稳定解析
2. `dev` 泛化检查：
   - 在 `GSM8K dev200` 上对比 base 与 SFT
   - 主看答案正确率，不把原生 `strict-match` 作为唯一结论
3. 样本人工抽查：
   - 检查是否存在明显截断、模板错配、或只会输出格式而不会解题的情况

说明：

- `train` 评测用于确认工程链路和 reward contract，不用于宣称泛化能力。
- 是否进入 GRPO，不取决于某一个 benchmark 分数，而取决于 `train` 解析稳定性和 `dev` 相对提升是否同时成立。

## 默认配置

当前 `configs/sft.yaml` 默认值：

```yaml
model_name: unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit
seed: 42
train_dataset: data/train_sft.jsonl
eval_dataset: data/eval/gsm8k_dev200.jsonl
test_dataset: data/eval/math500_test.jsonl
output_dir: outputs/sft-qwen3-1.7b
max_seq_length: 1024
learning_rate: 2.0e-4
epochs: 2
batch_size: 2
gradient_accumulation_steps: 4
lora_rank: 16
eval_max_new_tokens: 256
```

## 成功标准

Phase 1 完成条件：

1. `prepare_data -> check_sft_data -> train_sft -> eval` 全链路跑通。
2. base 和 SFT 模型都能在 `GSM8K dev200` 上完成评测。
3. SFT 在 `GSM8K dev200` 上的 `normalized_accuracy` 高于 base。
4. 最终对 `MATH500 test` 跑一次正式评测，并与 base 一起保存结果。

## 当前状态

截至 2026-03-12，Phase 1 的 SFT MVP 已基本完成。

当前最可信的 1.7B SFT 路线为：

- 训练集：NuminaMath 分层抽样 `2000`
- 训练样本：`response token length = 64~256`
- 目标：短 response / short-CoT + `Final answer: \boxed{...}`

当前结果：

- `train_sft_2k_short` 工程验收：
  - `format_success ≈ 0.81`
  - `parse_success ≈ 0.81`
- `GSM8K dev200`
  - base `flexible-extract ≈ 0.47`
  - SFT `flexible-extract ≈ 0.625`

这说明：

1. 当前 SFT 基线已经能够稳定学习 boxed 协议。
2. 相比 base，SFT 在外部开发集上已有明确提升。
3. 当前阶段的主问题已从“如何学会协议”切换为“如何在此基础上设计 GRPO”。

仍待补充：

- `MATH500 test` 的正式结果可作为 Phase 1 收尾指标补跑。

## 非目标

本阶段不做：

1. 完整实验追踪平台
2. 大规模 benchmark 编排
3. 复杂 reward 设计
4. 过早追求 RL SOTA
