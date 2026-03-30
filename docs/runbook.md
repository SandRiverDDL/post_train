# Runbook

先看这些口径文档：

- 当前阶段设计：[SPEC](SPEC.md)
- 当前有效基线：[STATE](STATE.md)
- 系统分层与职责：[ARCHITECTURE](ARCHITECTURE.md)

## 当前主线 Quickstart

当前主线是 `on-policy SFT`，基础起点是 `outputs/stage1_sft_5000/checkpoint-200`。

### 1. 准备单轮 on-policy 数据

```bash
.venv/bin/python scripts/prepare_on_policy_sft_data.py --config configs/on_policy/data.yaml
```

默认会：

- 对 `data/stage1_train_5000.jsonl` 做 full-harvest
- 每题采样 `4` 个 responses
- 保留全量 `raw_samples`
- 导出 `any_correct_shortest` 与 `mixed_only_shortest` 两套 retained

### 2. 训练单轮 on-policy SFT

```bash
.venv/bin/python scripts/train_sft.py --config configs/on_policy/sft.yaml
.venv/bin/python scripts/train_sft.py --config configs/on_policy/sft_mixed.yaml
```

### 3. 评测单轮模型

```bash
.venv/bin/python scripts/eval_model.py \
  --config configs/eval/default.yaml \
  --model outputs/on_policy_sft/round1 \
  --dataset data/eval/global_dev_math500_150.jsonl \
  --batch-size 6 \
  --output outputs/eval
```

默认结果会落到：

- `outputs/eval/on_policy_sft/round1/global_dev_math500_150/result.json`
- `outputs/eval/on_policy_sft/round1/global_dev_math500_150/raw.json`

### 4. 自动运行多轮 on-policy

```bash
.venv/bin/python scripts/run_on_policy_loop.py --config configs/on_policy/loop.yaml
```

当前默认主线 [configs/on_policy/loop.yaml](/home/chy/code/active/post_train/configs/on_policy/loop.yaml) 是 `512 prompt / round`；如果要跑论文迁移版 `opSFT`，优先看 [on_policy.md](/home/chy/code/active/post_train/docs/runbooks/on_policy.md) 里的 `loop_opsft_smallpool.yaml`。

## 按任务跳转

- `stage1` 基础数据与训练：[runbooks/stage1.md](runbooks/stage1.md)
- `on-policy` 单轮与循环：[runbooks/on_policy.md](runbooks/on_policy.md)
- 通用评测与 AIME：[runbooks/eval.md](runbooks/eval.md)
- 暂停路线 `stage2 / SIMPO`：[runbooks/paused_routes.md](runbooks/paused_routes.md)

## 使用约定

- 这里的 `runbook` 只放可执行步骤和常用命令。
- 当前主线优先看 `on_policy.md`，不要从暂停路线开始执行。
- benchmark、`global_dev`、`AIME`、`math220k_dev` 统一通过 `scripts/prepare_eval_data.py` 准备。
- 评测结果默认按模型路径和任务名分目录，不再平铺堆在 `outputs/eval/` 根目录。
- 若文档与实现冲突，以 `AGENTS.md`、`SPEC.md`、`STATE.md` 为准，并优先修正文档。
