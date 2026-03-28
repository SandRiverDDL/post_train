# Stage1 Runbook

这份文档只覆盖基础数据准备、`stage1` 训练和 checkpoint 选择。

## 准备 stage1 数据

```bash
.venv/bin/python scripts/prepare_stage1_data.py \
  --train-size 2000 \
  --dev-size 200
```

默认输出：

- `data/stage1_train.jsonl`
- `data/stage1_dev200.jsonl`

Mix-Long stage1 实验线：

```bash
.venv/bin/python scripts/prepare_stage2_mix_long_data.py --config configs/stage1/mix_long_data.yaml
```

默认规则：

- 数据源固定为 `UWNSL/Mix-Long_long_0.2_short_0.8`
- 直接使用过滤后的全量样本
- 当前默认删除 `solution_tokens > 4096` 的样本
- 如需临时缩小规模，可追加 `--sample-size N`

如需准备 benchmark 与其他评测数据：

```bash
.venv/bin/python scripts/prepare_eval_data.py benchmarks
.venv/bin/python scripts/prepare_eval_data.py global-dev
.venv/bin/python scripts/prepare_eval_data.py math220k-dev --profile main150
.venv/bin/python scripts/prepare_eval_data.py aime --year 24
.venv/bin/python scripts/prepare_eval_data.py aime --year 25
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
- 当前 `configs/stage1/sft.yaml` 默认开启 `PROFIT`
- `PROFIT` 会屏蔽 gold probability 小于 `profit_threshold` 的 completion token，不参与 loss

Mix-Long stage1 实验线训练：

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage1/mix_long_sft.yaml
```

这条线的默认差异：

- `max_seq_length = 5120`
- `batch_size = 4`
- `gradient_accumulation_steps = 8`
- 其余仍保持 stage1 的 `2 epoch` 与 checkpoint 选择口径

## 选择 stage1 最优 checkpoint

```bash
.venv/bin/python scripts/select_sft_checkpoint.py \
  --eval-config configs/eval/default.yaml \
  --train-output-dir outputs/stage1_sft \
  --dataset data/eval/global_dev_math500_150.jsonl \
  --batch-size 6
```

默认产物：

- `outputs/stage1_sft/dev_eval/dev_ranking.json`
- `outputs/stage1_sft/dev_eval/best_checkpoint.json`
- `outputs/stage1_sft/dev_eval/checkpoint-*/result.json`
- `outputs/stage1_sft/dev_eval/checkpoint-*/raw.json`
