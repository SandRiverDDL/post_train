# Runbook

## 1. 准备 stage1 与 benchmark 数据

```bash
.venv/bin/python scripts/prepare_stage1_data.py \
  --train-size 2000 \
  --dev-size 200
```

默认输出：

- `data/stage1_train.jsonl`
- `data/stage1_dev200.jsonl`
- `data/eval/gsm8k_test.jsonl`
- `data/eval/math500_test.jsonl`

## 2. 检查 SFT 数据

```bash
.venv/bin/python scripts/check_sft_data.py --input data/stage1_train.jsonl
```

期望至少满足：

- `boxed_rate` 接近 `1.0`
- `parse_success_rate` 接近 `1.0`
- `empty_final_answer = 0`

## 3. 运行 stage1 SFT

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage1_sft.yaml
```

## 4. 评测模型

```bash
.venv/bin/python scripts/eval_model.py --config configs/eval.yaml
```

常用覆盖：

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval.yaml \
  --model outputs/stage1_sft \
  --tasks gsm8k \
  --batch-size 6 \
  --max-new-tokens 256
```

按单数据集临时覆盖：

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval.yaml \
  --model outputs/stage1_sft \
  --dataset data/eval/gsm8k_test.jsonl \
  --batch-size 6 \
  --output outputs/eval_stage1_gsm8k.json
```

说明：

- 数学生成型任务默认使用手动 `batch_size`
- `max_batch_size` 只在显式使用 `batch_size=auto` 时才有意义

## 5. 运行 stage2 SFT

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage2_sft.yaml
```

说明：

- `stage2` 输入数据当前不在仓库里自动生成
- 只要准备好 `data/stage2_train.jsonl`，即可复用同一个训练入口

## 6. 运行 SIMPO

```bash
.venv/bin/python scripts/train_simpo.py --config configs/simpo.yaml
```

说明：

- `SIMPO` 训练依赖现成的 preference JSONL
- 当前不在仓库里内置 preference 构造逻辑
