# Eval Runbook

这份文档只放模型评测相关命令和常用覆盖方式。

## 默认评测

```bash
.venv/bin/python scripts/eval_model.py --config configs/eval/default.yaml
```

默认会走 `vllm_raw` runner。

## 常用覆盖

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval/default.yaml \
  --runner vllm_raw \
  --model outputs/stage2_sft/checkpoint-75 \
  --tasks gsm8k \
  --batch-size 6 \
  --max-lora-rank 32 \
  --max-new-tokens 512 \
  --limit 200
```

按单数据集临时覆盖：

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval/default.yaml \
  --runner vllm_raw \
  --model outputs/stage1_sft/checkpoint-50 \
  --dataset data/eval/global_dev_math500_150.jsonl \
  --batch-size 6 \
  --max-lora-rank 32 \
  --output outputs/eval
```

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
- `max_batch_size` 只在显式使用 `batch_size=auto` 时才有意义
- 评测 LoRA adapter 时，`max_lora_rank` 必须大于等于训练时的 `lora_rank`
- 两条链路复用同一个评测 prompt，只比较编排层差异
- 结果默认落到 `outputs/eval/<模型路径>/<task>/result.json` 与 `raw.json`
- 终端默认只打印核心字段；完整 metrics 仍保留在结果 JSON 中
- `AIME24/AIME25` 默认不在 `configs/eval/default.yaml` 里，需要用单独配置
- `AIME24/AIME25` 当前只在 `vllm_raw` 下支持每题多采样与 `pass@1`
