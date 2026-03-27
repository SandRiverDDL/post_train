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

如果要从 `MATH-500` 随机抽一个全局 dev：

```bash
.venv/bin/python scripts/prepare_global_dev.py
```

默认输出：

- `data/eval/global_dev_math500_150.jsonl`

如果要从 `Math220K` 生成固定 profile 的 stage2 target dev：

```bash
.venv/bin/python scripts/prepare_math220k_dev.py --profile main150
```

当前内置 profile：

- `main150`
- `short150`

默认输出：

- `data/eval/math220k_dev_main150.jsonl`
- `data/eval/math220k_dev_main150.report.json`

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

当前默认会：

- 训练 `2 epoch`
- 每 `25` step 保存一个 checkpoint
- 按长度分桶组 batch
- 每 `5` step 打印一次训练日志

## 4. 选择 stage1 最优 checkpoint

```bash
.venv/bin/python scripts/select_sft_checkpoint.py \
  --eval-config configs/eval.yaml \
  --train-output-dir outputs/stage1_sft \
  --dataset data/eval/global_dev_math500_150.jsonl \
  --runner vllm_raw \
  --batch-size 6 
```

默认产物：

- `outputs/stage1_sft/dev_eval/dev_ranking.vllm_raw.json`
- `outputs/stage1_sft/dev_eval/best_checkpoint.json`

## 5. 评测模型

```bash
.venv/bin/python scripts/eval_model.py --config configs/eval.yaml
```

默认会走 `vllm_raw` runner。

常用覆盖：

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval.yaml \
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
  --config configs/eval.yaml \
  --runner vllm_raw \
  --model outputs/stage1_sft/checkpoint-50 \
  --dataset data/eval/global_dev_math500_150.jsonl \
  --batch-size 6 \
  --max-lora-rank 32 \
  --output outputs/eval_stage1_global_dev.json
```

说明：

- 数学生成型任务默认使用手动 `batch_size`
- `max_batch_size` 只在显式使用 `batch_size=auto` 时才有意义
- 评测 LoRA adapter 时，`max_lora_rank` 必须大于等于训练时的 `lora_rank`
- 两条链路复用同一个评测 prompt，只比较编排层差异
- 结果文件会带上 runner 后缀，如 `gsm8k.vllm_raw.json`

## 6. 准备 stage2 数据

```bash
.venv/bin/python scripts/prepare_stage2_data.py --config configs/stage2_data.yaml
```

默认产物：

- `data/stage2_train.jsonl`
- `data/stage2_train.report.json`

当前 MVP 规则：

- `50%` 来自 `stage1_train`
- `35%` 来自 `math220k_short(<768)`
- `15%` 来自 `math220k_long(<1380)`
- `math220k` 中 `question_type == "MCQ"` 直接过滤
- `math220k` 的 `solution` 会被标准化为单个 `\boxed{...}` 尾部，不再直接重复追加 boxed

## 7. 运行 stage2 SFT

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage2_sft.yaml
```

说明：

- `stage2` 输入数据由 `prepare_stage2_data.py` 自动生成
- 训练入口仍复用同一个 `train_sft.py`
- 默认同样开启按长度分桶与 `5` step 日志
- 当前推荐 `learning_rate=2e-5`
- `stage2` 的主观察口径是：
  - 是否在 `math220k_dev_main150` 上更贴近目标分布
  - 是否同时伤害 `GSM8K` 这类 transfer benchmark

## 8. 运行 SIMPO

先准备 full preference 数据：

```bash
.venv/bin/python scripts/prepare_simpo_data.py --config configs/simpo_data.yaml
```

默认产物：

- `data/simpo/query_pool.jsonl`
- `data/simpo/raw_samples.jsonl`
- `data/simpo/train.jsonl`
- `data/simpo/train.pilot.jsonl`
- `data/simpo/train.report.json`

当前默认规则：

- 使用 `stage1 best checkpoint` 采样
- 从 `stage1` 同源未用题中抽 `800` 条 query
- 每题采样 `4` 个 responses
- 优先构造 `correct > incorrect`
- 不足时回退到 `correct > correct`
- 目标 `500` 对 pair，但不足不报错
- 会额外从 full 数据中固定抽出 `150` 对 `pilot` 子集，优先保留 `correct > incorrect`

如果只想先跑便宜的 pilot 数据准备：

```bash
.venv/bin/python scripts/prepare_simpo_data.py --config configs/simpo_data.pilot.yaml
```

先跑 pilot 训练：

```bash
.venv/bin/python scripts/train_simpo.py --config configs/simpo.pilot.yaml
```

再跑 full 训练：

```bash
.venv/bin/python scripts/train_simpo.py --config configs/simpo.yaml
```

如需从 checkpoint 继续：

```bash
.venv/bin/python scripts/train_simpo.py \
  --config configs/simpo.yaml \
  --resume-from-checkpoint outputs/simpo/checkpoint-10
```

说明：

- `SIMPO` 默认从 `stage1 best checkpoint` 起训
- 当前默认 `beta=2.0`、`gamma=1.0`
- 当前默认启用 `4bit + PEFT`
- 当前默认 `logging_steps=5`、`save_steps=25`
