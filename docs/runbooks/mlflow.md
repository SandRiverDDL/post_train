# MLflow Runbook

本项目使用 MLflow 管理实验索引、参数、指标和小型报告。大文件仍保存在 `data/` 与 `outputs/`。

## 启动 UI

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 0.0.0.0 --port 5000
```

默认环境变量：

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
export MLFLOW_EXPERIMENT_NAME=post_train
```

临时禁用：

```bash
export POST_TRAIN_DISABLE_MLFLOW=1
```

## 当前接入范围

- `scripts/train_sft.py`
- `scripts/train_grpo.py`
- `scripts/eval_model.py`
- `scripts/prepare_lightning_opd_data.py`
- `scripts/rollout_math_sft.py`

训练输出目录会写入：

```text
outputs/<run>/mlflow_run.json
```

评测时如果模型目录存在 `mlflow_run.json`，会把指标写回对应训练 run。

## Artifact 口径

MLflow 只记录小文件：

- `result.json`
- `report.json`
- `run_summary.json`
- 配置与数据准备摘要

以下大文件不复制进 MLflow，只记录路径：

- checkpoint
- `raw.json`
- `train.jsonl`
- `raw_rollouts.jsonl`
- teacher logits JSONL

## 导入已有评测结果

```bash
.venv/bin/python scripts/import_existing_to_mlflow.py --eval-root outputs/eval
```

## 迁移服务器

除 Git 之外，需要同步：

- `mlruns/`
- 关键 `data/`
- 关键 `outputs/`

具体目录看 `docs/runbooks/migration.md`。
