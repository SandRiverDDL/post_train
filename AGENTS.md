# 项目名称

Qwen3 数学后训练项目

## 语言规范

始终使用中文回答与编写代码注释。

## 全局规则

遵守同目录下的 `global_rules.md`。

## 开发环境

* 优先使用 `.venv/bin/python` 运行和检查代码。
* 若 `.venv` 不可用，使用 `uv run python`。
* 除非用户明确要求，禁止重新安装依赖或重新创建环境。

## 项目目标

实现一个完整的 LLM 数学后训练流程。当前阶段只实现最小可运行闭环：

1. SFT
2. Benchmark 评测
3. 后续再接 GRPO

优先建立**可信、可运行、可扩展的工程基线**，再逐步扩大规模。

## 模型策略

* 开发模型：`Qwen/Qwen3-1.7B-Base`
* 最终实验模型：`Qwen/Qwen3-8B-Base`

开发阶段优先使用 1.7B，以保证调试速度和迭代效率。

## 数据与评测基线

* `train`：`nlile/NuminaMath-1.5-RL-Verifiable`
* `dev`：`gsm8k`（固定 200 条）
* `test`：`HuggingFaceH4/MATH-500`（全量）

约束：

1. `train` 与 `dev` 可以进行长度过滤。
2. `test` 不允许按长度裁剪，只允许字段映射和格式标准化。
3. NuminaMath 训练集按 `problem_type` 分层采样，优先使用原始 `answer` 字段作为最终答案。

## 输出格式

训练、开发和评测数据统一使用尾部格式：

```
Final answer: \boxed{...}
```

该格式同时用于：

* GRPO reward 解析
* strict eval

## 必跑检查

数据准备完成后必须运行：

```bash
python scripts/check_sft_data.py
```

期望指标：

* `empty_final_answer = 0`
* `boxed_rate ≈ 1.0`
* `parse_success_rate ≈ 1.0`
* `consistent_rate ≈ 1.0`

若不满足，必须先修复数据再继续训练。

## 评测约定

正式评测入口：

```
scripts/eval_dataset.py
```

规则：

1. 正式结果使用 **strict boxed 协议**。
2. `GSM8K dev200` 与 `MATH500 test` 统一通过该入口评测，通过 `--dataset` 区分。
3. 评测框架使用 `lm-evaluation-harness`，数学答案比对使用 `math-verify`。
4. `data/eval/gsm8k_dev200.jsonl` 一旦确认可用即视为**冻结工件**，不得因 tokenizer 或长度配置变化重新生成。
5. LoRA / SFT 模型在进入 GRPO 前，必须在 `data/train_sft.jsonl` 上完成工程验收，至少检查：

   * `format_success`
   * `parse_success`
   * `normalized_accuracy`

   以确保 reward 解析链路稳定。

## 项目结构

```
configs/
scripts/
src/
docs/
data/
tests/
```

* `scripts/`：训练、数据准备、评测入口
* `src/rl/`：配置、数据处理、答案解析、评测集成
* `docs/`：阶段设计文档
* `tests/`：最小单元测试

## Codex 工作规则

1. 优先保证当前阶段的**最小可运行闭环**，不要提前实现未来阶段功能。
2. 文档必须与当前实现保持一致。

## 完成定义

一次实现变更只有在满足以下条件时才算完成：

1. 代码、配置与文档口径一致。
2. 已运行相关最小验证命令并记录结果。
3. 未引入重复实现（duplicate logic）。
4. 若改动影响训练或评测链路，必须更新相关说明文档。

