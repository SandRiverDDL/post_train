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

如果需要直接调用单步 CLI，也可以显式传配置：

```bash
.venv/bin/python scripts/prepare_lightning_opd_data.py \
  --config configs/lightning_opd/candidate.yaml \
  --mode prompts
```
