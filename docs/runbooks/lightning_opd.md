# Lightning-OPD 数据准备

当前只实现 offline 数据准备，不实现 trainer。

默认入口：

```bash
runs/lightning_opd/prepare_candidate.sh
```

默认配置：

```bash
configs/lightning_opd/candidate.yaml
```

修改参数优先改 YAML；临时覆盖可以继续用环境变量：

```bash
SAMPLE_SIZE=10 TOP_K=4 OUT=data/lightning_opd/smoke_candidate_n10_topk4 \
  runs/lightning_opd/prepare_candidate.sh
```

默认口径：

- prompt 源：`data/on_policy_loop/query_strategy/candidate_pool.jsonl`
- prompt 数：`1000`
- student rollout：`outputs/stage1_mix_long_sft/checkpoint-300`
- student base：`Qwen/Qwen2.5-Math-1.5B` 本地 snapshot
- teacher：`hbx/JustRL-DeepSeek-1.5B` 本地 snapshot
- teacher topK：`32`
- shard：默认 `GPU0=3`、`GPU1=4`

输出结构：

```text
data/lightning_opd/candidate/
  prompts.jsonl
  train.jsonl
  report.json
  shards/
    shard0-of-2/
      raw_rollouts.jsonl
      teacher_topk.jsonl
    shard1-of-2/
      raw_rollouts.jsonl
      teacher_topk.jsonl
  logs/
    shard0.log
    shard1.log
```

`train.jsonl` 保存固定 response、`input_ids`、`response_mask`、`teacher_token_logprobs`、`teacher_topk_token_ids` 和 `teacher_topk_logprobs`。根目录 `report.json` 统一记录 shard 行数、长度统计和 shape 校验。

## Qwen3 ConPress ASFT 数据准备

Qwen3 student 不能直接复用默认 Lightning-OPD 裸 prompt。必须使用训练/评测一致口径：

- `prompt_style=justrl_math`
- Qwen3 chat template
- `system_prompt=""`
- `assistant_prefill="<think>\n\n</think>\n\n"`

当前 2000 条 query source 已提前渲染：

```text
data/lightning_opd/query_sources/conpress_asft_qwen3_chat_math2000_seed42.jsonl
```

当前配置：

```text
configs/lightning_opd/conpress_asft_qwen3_4b_teacher_math2000.yaml
```

本轮运行命令：

```bash
tmux new-session -d -s conpress_asft_opd_math2000_4gpu_20260504 \
  'cd /mnt/dataset/fengshuwen/post_train && \
   CONFIG=configs/lightning_opd/conpress_asft_qwen3_4b_teacher_math2000.yaml \
   GPUS=0,1,2,4 \
   STAGES=prompts,shard,merge \
   PYTHONPATH=src bash runs/lightning_opd/prepare_candidate.sh \
   2>&1 | tee logs/eval/conpress_asft_opd_math2000_4gpu_20260504.log'
```

输出目录：

```text
data/lightning_opd/conpress_asft_qwen3_4b_teacher_math2000_20260504
```

注意事项：

- ASFT LoRA rank 为 `32`。vLLM 默认 `max_lora_rank=16` 会报 `LoRA rank 32 is greater than max_lora_rank 16`；当前代码会从 `adapter_config.json` 自动读取 rank 并传给 vLLM。
- student vLLM rollout 后会显式 shutdown 并释放 CUDA cache，再加载 teacher 做 sampled-token logprob，避免同一进程内显存残留。
- 当前 teacher 使用 `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`，BF16 HF forward，不使用 4bit。标准 sampled-token OPD 只需要 `teacher_token_logprobs`；`top_k=0` 时不保存 teacher topK。

当前状态：

- student raw rollout 已完成，四个 shard 各 `500` 条。
- BF16 teacher sampled-token forward 已完成，四个 shard 各 `500` 条。
- 此前 teacher scoring 在 full-vocab dense `log_softmax` 临时张量处 OOM；当前代码已改为 sampled-token `target_logit - logsumexp(logits)`，不再 materialize 完整 `[B, L, V]` log-prob 张量。
- 合并产物 `train.jsonl` 共 `2000` 行，`shape_errors=0`，`top_k_values=[0]`。

teacher-only resume 示例：

```bash
tmux new-session -d -s conpress_asft_opd_math2000_teacher_retry_20260504 \
  'cd /mnt/dataset/fengshuwen/post_train && \
   CONFIG=configs/lightning_opd/conpress_asft_qwen3_4b_teacher_math2000.yaml \
   GPUS=0,1,2,4 \
   STAGES=teacher,merge \
   TEACHER_BATCH_SIZE=1 \
   TOP_K=0 \
   TEACHER_LOAD_IN_4BIT=0 \
   PYTHONPATH=src bash runs/lightning_opd/prepare_candidate.sh \
   2>&1 | tee logs/eval/conpress_asft_opd_math2000_teacher_retry_20260504.log'
```

## Lightning-OPD 训练

默认 baseline 使用原始 Lightning-OPD 语义：只用轨迹 token 的 `teacher_token_logprobs`，不启用 topK 辅助蒸馏。

```bash
.venv/bin/python scripts/train_sft.py --config configs/lightning_opd/train_candidate.yaml
```

如果要启用 topK 辅助蒸馏，`distill_top_k` 不能超过离线保存的 `teacher_topK`。当前 `data/lightning_opd/candidate/train.jsonl` 保存的是 `32`。

```bash
.venv/bin/python scripts/train_sft.py \
  --config configs/lightning_opd/train_candidate.yaml \
  --set distill_top_k=8 \
  --set topk_kd_weight=0.1
```

当前 ConPress ASFT 不过滤版训练：

```bash
GPUS=0,1,2,4 CONFIG=configs/lightning_opd/train_conpress_asft_qwen3_4b_teacher_math2000.yaml \
  PYTHONPATH=src bash runs/sft/train_sft_ddp.sh
```

已完成自动流水线：

```text
tmux conpress_asft_lightning_opd_train_eval_20260504:
  1. 等待 teacher forward 完成
  2. 校验 train.jsonl 为 2000 行
  3. 四卡训练 Lightning-OPD
  4. 四卡并行 dev best ckpt 选择
  5. 用 best checkpoint 跑 MATH500/GSM8K
```

结果：

- best checkpoint：`outputs/lightning_opd_conpress_asft_qwen3_4b_teacher_math2000_unfiltered/checkpoint-30`
- dev200：acc `0.630`，boxed `0.785`，avg tokens `459.3`
- MATH500：acc `0.678`，boxed `0.824`，avg tokens `394.2`
- GSM8K：acc `0.8135`，boxed `0.9750`，avg tokens `138.5`
- 相比 ConPress ASFT 起点 `checkpoint-175` 的 MATH500/GSM8K `0.654 / 0.8006` 有准确率正收益，但 MATH500 boxed rate 从 `0.850` 降到 `0.824`。

如果需要直接调用单步 CLI，也可以显式传配置：

```bash
.venv/bin/python scripts/prepare_lightning_opd_data.py \
  --config configs/lightning_opd/candidate.yaml \
  --mode prompts
```
