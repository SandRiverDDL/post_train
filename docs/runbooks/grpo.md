# GRPO Runbook

这份文档覆盖当前服务器上的 `GRPO / DAPO-lite` 数据准备、训练、checkpoint 选择与 benchmark 的最小运行路径。

## 数据准备

```bash
.venv/bin/python scripts/prepare_grpo_data.py --config configs/grpo/data.yaml
```

默认行为：

- 从 `rd211/Big-Math-RL-Verified-Filtered` 的 `train` split 读取题目
- 仅保留 `0.25 < llama8b_solve_rate < 0.5` 的样本
- 与本地 anchor 题池混合
- 先做 exact dedup，再按 `anchor_share` 采样混合

默认产物：

- `data/grpo/train.jsonl`
- `data/grpo/train.report.json`

## 训练

保守单卡配置：

```bash
.venv/bin/python scripts/train_grpo.py --config configs/grpo/train_cheap.yaml
```

单卡 3090 配置：

```bash
.venv/bin/python scripts/train_grpo.py --config configs/grpo/train_3090.yaml
```

当前双卡 `DAPO-lite` 默认入口：

```bash
.venv/bin/python scripts/run_grpo_3090_dapo_server.py
```

指定 trainer / vLLM GPU 与端口：

```bash
.venv/bin/python scripts/run_grpo_3090_dapo_server.py \
  --config configs/grpo/train_3090_dapo_server_resume766.yaml \
  --trainer-gpu 1 \
  --vllm-gpu 2 \
  --vllm-port 8001
```

当前双卡默认行为：

- trainer 与 `vLLM server` 分卡运行
- 默认优先：
  - trainer `GPU 6`
  - `vLLM server` `GPU 7`
- 若 `6/7` 空余显存不足，再回退到其他空闲卡对
- 已有健康且匹配的 `vLLM server` 会优先复用，不重复冷启动
- 使用 `--vllm-port` 时，wrapper 会把实际 `vLLM server` 地址通过 `GRPO_VLLM_SERVER_BASE_URL` 传给 trainer，避免 trainer 仍按 yaml 里的旧端口连接
- 如果 `127.0.0.1:8000` 已被占用但 `/health/` 不可用，入口会直接报错；先用 `ss -ltnp 'sport = :8000'` 确认残留进程归属，再手动关闭

当前 `DAPO-lite` 默认配置：

- `reward_config: configs/grpo/reward_dapo.yaml`
- `loss_type: dapo`
- `epsilon: 0.2`
- `epsilon_high: 0.28`
- `beta: 0.0`
- `mask_truncated_completions: true`
- `max_completion_length: 768`
- `soft_overlong.weight: 0.05`
- `soft_overlong.cache_tokens: 192`

题级统计：

- 需要题级 rollout 摘要时，在训练配置中设置 `question_stats_path`
- 当前记录本地 JSONL，不写入 W&B
- 每行对应一次 reward batch 中的一道题，包含：
  - `global_step`
  - `question_id`
  - `num_generations`
  - `num_correct`
  - `correct_rate`
  - `parse_success_rate`
  - completion 平均长度摘要
- 该文件用于后续筛选 hard / easy / mixed 题，不保存完整回答文本

注意：

- 当前实现会在训练前做 preflight
- 没有 GPU 会直接失败
- `use_vllm=true` 时要求当前环境中的 `TRL / vLLM` 组合与实现兼容
- 训练启动时会打印 `grpo.adapter.*`，用于确认：
  - 当前是否从 adapter checkpoint 继续训练
  - 当前是否真的挂上 LoRA
  - 当前 LoRA 参数数量与底模路径

## checkpoint 选择

在 dev 集上批量评 checkpoint：

```bash
.venv/bin/python scripts/select_sft_checkpoint.py   --eval-config configs/eval/default.yaml   --train-output-dir outputs/grpo_3090_dapo_server   --dataset data/eval/math500_dev200.jsonl   --batch-size 16   --max-new-tokens 768
```

当前行为：

- 按 `normalized_accuracy` 选 best checkpoint
- 结果写到：
  - `<train_output_dir>/dev_eval/dev_ranking.json`
  - `<train_output_dir>/dev_eval/best_checkpoint.json`
- 当前评测日志会打印 `eval.model_resolution.*`，用于确认：
  - `pretrained`
  - `enable_lora`
  - `lora_local_path`
- ranking / best summary 里也会保留 `model_resolution`

注意：

- `max-new-tokens` 会直接影响 dev 结果；如果给太小，容易因为截断导致误判 checkpoint 质量
- 当前 `math500_dev200` 只有 `200` 题，单次标准误差较高，不能用很小的分差下强结论
- 同一 checkpoint 不需要重复 eval 多次；当前评测在 `samples_per_problem=1` 时基本是确定性的，真正值得重复的是多训练 seed。
- `scripts/select_sft_checkpoint.py` 现在会在同一底模下一次性创建并复用单个 `vLLM` 实例，只切换不同 checkpoint 的 `LoRARequest`。

## benchmark 评测

单个模型评测：

```bash
.venv/bin/python scripts/eval_model.py   --config configs/eval/default.yaml   --model outputs/grpo_3090_dapo_server/checkpoint-30   --tasks math500 gsm8k   --max-new-tokens 768   --batch-size 16
```

当前输出目录口径：

- `outputs/eval/<模型路径>/<task>/result.json`
- `outputs/eval/<模型路径>/<task>/raw.json`

## 单轮 workflow

把训练、dev 选 best checkpoint、benchmark 与 baseline 对比串起来：

```bash
.venv/bin/python scripts/run_experiment_workflow.py --config configs/workflow/grpo_dapo_server.yaml
```

当前 workflow v1 会：

1. 训练 `GRPO`
2. 在 `math500_dev200` 上选 best checkpoint
3. 评 `math500` 与 `gsm8k`
4. 写 `workflow_summary.json` 或 `workflow_failure.json`

## 当前判读建议

- 先看 `math500_dev200` 做粗筛
- 再看 `math500_test + gsm8k_test` 做最终判断
- 若 dev 分数只差几个点以内，优先结合：
  - `boxed_rate`
  - `parse_success_rate`
  - `completions/clipped_ratio`
  - `entropy`
  一起判断，不要只看单个 accuracy 数字
