# 项目名称

Qwen3 数学后训练项目

## 语言规范

永远用中文回答以及写注释。

## 开发环境

- 此仓库使用项目本地虚拟环境。
- 如果存在，则优先使用`.venv/bin/python`进行检查和执行。
- 如果 `.venv` 不可用，请优先使用 `uv run python`。
- 除非有明确要求，否则请勿重新安装依赖项或重新创建环境。
- 在更改环境之前，请先检查`.venv`、`pyproject.toml`和锁定文件。

## 项目目标

本项目实现一个完整的 LLM 数学后训练流程，当前聚焦数学任务的最小可运行闭环：

1. SFT
2. Benchmark 评测
3. 后续再接 GRPO

目标是先做出可信、可运行、可扩展的工程基线，再逐步扩大规模。

## 模型策略

- 开发模型：`Qwen/Qwen3-1.7B-Base`
- 最终实验模型：`Qwen/Qwen3-8B-Base`

开发阶段优先使用 1.7B 模型，保证迭代速度和调试效率。

## 数据与评测基线

- `train`：`nlile/NuminaMath-1.5-RL-Verifiable`
- `dev`：`gsm8k`，固定 `200` 条
- `test`：`HuggingFaceH4/MATH-500` 全量

约束：

1. `train/dev` 可以做长度过滤。
2. `test` 不允许按长度裁剪，只允许字段映射和格式标准化。
3. NuminaMath 训练集按 `problem_type` 分层采样，优先使用数据集原生 `answer` 字段作为最终答案来源。

## 输出格式

训练、开发和正式评测数据统一为以下尾部格式：

```text
Final answer: \boxed{...}
```

该格式用于后续 GRPO reward 解析，也用于 strict eval。

## 工程原则

1. 优先实现最小可运行闭环，不为未来需求提前抽象复杂框架。
2. 复用优先于重写：benchmark、答案比对、标准评测优先复用 `lm-evaluation-harness` 与 `math-verify`；只有在 boxed 协议或 reward 链路确有缺口时，才补充自定义实现。
3. 共享逻辑必须收敛到 `src/rl/`；`scripts/` 只保留流程编排，不复制业务逻辑。
4. 禁止为同一职责长期维护两套实现，例如两套答案解析器、两套评测入口、两套数据格式转换逻辑。
5. 关键实验参数必须配置化，禁止把训练/评测关键口径硬编码在脚本正文中。
6. 任何影响数据格式、答案解析、评测口径的改动，必须附最小验证步骤；未验证前，不进入长时间训练或下一阶段实验。
7. 重构必须以减少重复、提高可验证性为目标；如果当前需求只服务一个调用点，优先保持简单实现。

## 必跑检查

准备数据后，先运行：

```bash
python scripts/check_sft_data.py
```

期望至少满足：

- `empty_final_answer=0`
- `boxed_rate` 接近 `1.0`
- `parse_success_rate` 接近 `1.0`
- `consistent_rate` 接近 `1.0`

不满足时先修数据，不继续训练。

## 评测约定

- 正式评测入口：`scripts/eval_dataset.py`

规则：

1. 正式结果使用 strict boxed 协议。
2. `GSM8K dev200` 和 `MATH500 test` 都通过同一个评测入口执行，靠 `--dataset` 区分。
3. 评测统一走 `lm-evaluation-harness`，数学答案比对统一走 `math-verify`。
4. `data/eval/gsm8k_dev200.jsonl` 一旦确认可用，应视为冻结工件，不随 tokenizer 或长度配置变化重新生成。
5. LoRA/SFT 产物在进入 GRPO 前，必须先在 `data/train_sft.jsonl` 上做工程验收，至少检查 `format_success`、`parse_success`、`normalized_accuracy`，确认 reward 解析链路稳定。

## 项目结构

```text
configs/
scripts/
src/
docs/
data/
tests/
```

- `scripts/`：训练、数据准备、评测入口
- `src/rl/`：配置、数据、答案解析、评测集成
- `docs/`：阶段设计文档
- `tests/`：最小单测

## Codex 工作规则

1. 先读仓库结构和阶段文档。
2. 先保证最小闭环，再做扩展。
3. 文档要与当前实现保持一致，避免“代码一个版本，文档一个版本”。

## 完成定义

一次实现变更只有在以下条件满足时才算完成：

1. 代码、配置、文档口径一致。
2. 相关最小验证命令已运行并记录结果。
3. 未引入重复实现或无必要的新抽象。
4. 若改动影响训练/评测链路，已更新相应工件说明或状态文档。
