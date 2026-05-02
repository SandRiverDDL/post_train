# Eval Runbook

这份文档只放模型评测相关命令和常用覆盖方式。

## 默认评测

```bash
.venv/bin/python scripts/eval_model.py --config configs/eval/default.yaml
```

默认固定走 `vllm` 后端上的本地评测链路。

当前内部统一由 `post_train.eval.EvalRunner` 编排；CLI 参数、输出路径和 `result.json/raw.json` 格式保持兼容。

## 常用覆盖

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval/default.yaml \
  --model outputs/stage2_sft/checkpoint-75 \
  --tasks gsm8k \
  --batch-size 16 \
  --max-new-tokens 512
```

按单数据集临时覆盖：

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval/default.yaml \
  --model outputs/stage1_sft/checkpoint-50 \
  --dataset data/eval/math500_dev200.jsonl \
  --batch-size 6 \
  --max-lora-rank 32 \
  --output outputs/eval
```

当前主线开发集：

- `data/eval/math500_dev200.jsonl`
- 由 `scripts/prepare_eval_data.py global-dev` 从 `HuggingFaceH4/MATH-500` 按 `level` 分层抽样生成

## AIME 评测

运行合并版 `AIME24/AIME25` sampled pass@1：

```bash
.venv/bin/python scripts/eval_model.py --config configs/eval/aime.yaml
```

如果只想单独跑 `AIME24`：

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval/aime.yaml \
  --tasks aime24
```

## 说明

- 数学生成型任务默认使用手动 `batch_size`
- 评测 LoRA adapter 时，`max_lora_rank` 必须大于等于训练时的 `lora_rank`
- 结果默认落到 `outputs/eval/<模型路径>/<task>/result.json` 与 `raw.json`
- 终端默认只打印核心字段；完整 metrics 仍保留在结果 JSON 中
- checkpoint selection 会复用同一个 vLLM backend runner，只切换 LoRA adapter
- `AIME24/AIME25` 默认不在 `configs/eval/default.yaml` 里，需要用单独配置
- `AIME24/AIME25` 当前通过同一条评测链路支持每题多采样与 `pass@1`

## 结果总表维护

评测结束后刷新总表：

```bash
.venv/bin/python scripts/report_eval_results.py
```

默认读取：

- `outputs/eval/**/result.json`
- `docs/analysis/eval_metadata.yaml`
- metadata 中列出的历史手动记录，例如 `tmp/results.md` 与 `tmp/new_results.md`

默认写入：

- `docs/analysis/eval_registry.jsonl`
- `docs/analysis/eval_leaderboard.md`

维护规则：

- 不手改 `eval_registry.jsonl` 和 `eval_leaderboard.md`，它们是生成文件
- `eval_registry.jsonl` 保留扫描到的全量结果
- `eval_metadata.yaml` 是人工展示白名单，只写重要模型、效果好的 checkpoint、关键 baseline 与明确要归档的坏例
- 手动维护 `eval_metadata.yaml` 里的 `method / label / role / ignore / notes`
- 未写入 metadata 的模型默认只进 registry，不进 leaderboard
- 同模型同任务同时存在正式 `result.json` 与手动记录时，正式 `result.json` 优先
- `math500_dev200` 这类 dev-only 结果只进入 Task Details，不进入 Main Results
