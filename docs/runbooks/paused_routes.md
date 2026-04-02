# 暂停路线 Runbook

这份文档只保留暂停路线的可执行命令。路线背景、收益判断和暂停原因看：

- [两阶段 SFT + SIMPO（暂停路线）](../experiments/two-stage-sft-simpo.md)

## Stage2 数据准备

```bash
.venv/bin/python scripts/prepare_stage2_data.py --config configs/stage2/data.yaml
```

默认产物：

- `data/stage2/default/train.jsonl`
- `data/stage2/default/train.report.json`

当前 MVP 规则：

- `50%` 来自 `stage1_train`
- `35%` 来自 `math220k_short(<768)`
- `15%` 来自 `math220k_long(<1380)`
- `math220k` 中 `question_type == "MCQ"` 直接过滤
- `math220k` 的 `solution` 会被标准化为单个 `\boxed{...}` 尾部

hendrycks_math + long CoT 对照线：

```bash
.venv/bin/python scripts/prepare_stage2_hendrycks_long_data.py \
  --config configs/stage2/hendrycks_long_data.yaml
```

默认规则：

- 只保留 `hendrycks_math` 中 `level in {4,5}` 的题
- 与 `UWNSL/MATH_training_split_long_cot` 按标准化后的 `problem` 精确匹配
- 当前默认过滤 `solution_tokens > 4096` 的 long CoT
- 最终按 `long:short = 1:2` 组装到 `2000` 条
- 会额外写 unmatched 预览，便于检查匹配率

## Stage2 SFT

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage2/sft.yaml
```

说明：

- `stage2` 输入数据由 `prepare_stage2_data.py` 生成
- 训练入口仍复用 `train_sft.py`
- 当前推荐 `learning_rate=2e-5`

hendrycks_math + long CoT 对照线训练：

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage2/hendrycks_long_sft.yaml
```

## 准备 SIMPO 数据

准备 full preference 数据：

```bash
.venv/bin/python scripts/prepare_simpo_data.py --config configs/simpo/data.yaml
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
- 会额外从 full 数据中固定抽出 `150` 对 `pilot` 子集

如果只想先跑便宜的 pilot 数据准备：

```bash
.venv/bin/python scripts/prepare_simpo_data.py --config configs/simpo/data_pilot.yaml
```

## 运行 SIMPO

先跑 pilot 训练：

```bash
.venv/bin/python scripts/train_simpo.py --config configs/simpo/pilot.yaml
```

再跑 full 训练：

```bash
.venv/bin/python scripts/train_simpo.py --config configs/simpo/train.yaml
```

如需从 checkpoint 继续：

```bash
.venv/bin/python scripts/train_simpo.py \
  --config configs/simpo/train.yaml \
  --resume-from-checkpoint outputs/simpo/checkpoint-10
```

说明：

- `SIMPO` 默认从 `stage1 best checkpoint` 起训
- 当前默认 `beta=2.0`、`gamma=1.0`
- 当前默认启用 `4bit + PEFT`
- 当前默认 `logging_steps=5`、`save_steps=25`
