# Stage1 Runbook

这份文档只覆盖基础数据准备、`stage1` 训练和 checkpoint 选择。

## 准备 stage1 与 benchmark 数据

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

如需额外准备评测数据：

```bash
.venv/bin/python scripts/prepare_global_dev.py
.venv/bin/python scripts/prepare_math220k_dev.py --profile main150
.venv/bin/python scripts/prepare_aime_eval.py --year 24
.venv/bin/python scripts/prepare_aime_eval.py --year 25
```

## 检查 SFT 数据

```bash
.venv/bin/python scripts/check_sft_data.py --input data/stage1_train.jsonl
```

期望至少满足：

- `boxed_rate` 接近 `1.0`
- `parse_success_rate` 接近 `1.0`
- `empty_final_answer = 0`

## 运行 stage1 SFT

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage1/sft.yaml
```

默认行为：

- 训练 `2 epoch`
- 每 `25` step 保存 checkpoint
- 按长度分桶组 batch
- 每 `5` step 打印一次训练日志

## 选择 stage1 最优 checkpoint

```bash
.venv/bin/python scripts/select_sft_checkpoint.py \
  --eval-config configs/eval/default.yaml \
  --train-output-dir outputs/stage1_sft \
  --dataset data/eval/global_dev_math500_150.jsonl \
  --runner vllm_raw \
  --batch-size 6
```

默认产物：

- `outputs/stage1_sft/dev_eval/dev_ranking.json`
- `outputs/stage1_sft/dev_eval/best_checkpoint.json`
