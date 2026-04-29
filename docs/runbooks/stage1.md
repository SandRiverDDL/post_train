# Stage1 Runbook

这份文档只覆盖基础数据准备、`stage1` 训练和 checkpoint 选择。

## 准备 stage1 数据

```bash
.venv/bin/python scripts/prepare_stage1_data.py \
  --train-size 2000 \
  --dev-size 200
```

默认输出：

- `data/stage1/train.jsonl`
- `data/stage1/dev200.jsonl`

Mix-Long stage1 实验线：

```bash
.venv/bin/python scripts/prepare_stage2_mix_long_data.py --config configs/stage1/mix_long_data.yaml
```

Math220K stage1 对照线：

```bash
.venv/bin/python scripts/prepare_stage1_math220k_data.py --config configs/stage1/math220k_data.yaml
```

RSR 候选池与二段筛选：

```bash
.venv/bin/python scripts/prepare_stage1_rsr_candidates.py --config configs/stage1/rsr_candidates.yaml
.venv/bin/python scripts/select_stage1_rsr_dataset.py --config configs/stage1/rsr_select.yaml
```

RSR 默认规则：

- 当前默认 `drop_truncated: true`
- `prompt + solution` 超过 `max_seq_length` 的轨迹会直接丢弃，不再截断参与打分
- 候选池报告会记录超长轨迹的舍弃比例
- `rsr_select.yaml` 可选设置 `allowed_sources`，例如只从 `short/small/large` 中选，先排除 `long`

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

其中 `global-dev` 当前固定语义是：

- 从 `HuggingFaceH4/MATH-500` 重新取数
- 按原始 `level` 字段做 `dev200` 分层抽样
- 默认输出 `data/eval/math500_dev200.jsonl`

## 检查 SFT 数据

```bash
.venv/bin/python scripts/check_sft_data.py --input data/stage1/train.jsonl
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
 CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_sft.py --config configs/stage1/mix_long_sft.yaml
```

Math220K stage1 对照线训练：

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage1/math220k_sft.yaml
```

RSR 筛选后的 stage1 训练：

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage1/rsr_sft.yaml
.venv/bin/python scripts/collect_results.py
```

这条线使用独立配置：

- `train_dataset = data/stage1/rsr/selected_train.jsonl`
- `output_dir = outputs/stage1_rsr_sft`

这条线的默认差异：

- 其余超参与主 `stage1` 配置保持一致，便于把差异集中在数据集本身

## 选择 stage1 最优 checkpoint

```bash
.venv/bin/python scripts/select_sft_checkpoint.py \
  --eval-config configs/eval/default.yaml \
  --train-output-dir outputs/stage1_sft \
  --dataset data/eval/math500_dev200.jsonl \
  --batch-size 6
```

默认产物：

- `outputs/stage1_sft/dev_eval/dev_ranking.json`
- `outputs/stage1_sft/dev_eval/best_checkpoint.json`
- `outputs/stage1_sft/dev_eval/checkpoint-*/result.json`
- `outputs/stage1_sft/dev_eval/checkpoint-*/raw.json`
- `outputs/stage1_sft/run_summary.json`
- `experiments/registry.jsonl`
