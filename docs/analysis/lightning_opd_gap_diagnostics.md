# Lightning OPD Gap 诊断

## 背景

本诊断用于判断当前 Lightning OPD 数据中，Nemotron teacher 对 `stage1_mix_long_sft/checkpoint-300` student trajectory 的 token-level 信号是否健康。

相关产物：

- OPD 训练集：`data/lightning_opd/nemotron_candidate/train.jsonl`
- 诊断脚本：`scripts/analyze_lightning_opd_gap.py`
- 诊断报告：`data/lightning_opd/nemotron_candidate/student_teacher_gap_report.json`
- student：`outputs/stage1_mix_long_sft/checkpoint-300`
- teacher：`OpenMath-Nemotron-1.5B`

运行命令：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src .venv/bin/python scripts/analyze_lightning_opd_gap.py \
  --dataset data/lightning_opd/nemotron_candidate/train.jsonl \
  --student-model outputs/stage1_mix_long_sft/checkpoint-300 \
  --adapter-base-model /mnt/dataY/fsw/cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2 \
  --output data/lightning_opd/nemotron_candidate/student_teacher_gap_report.json \
  --batch-size 1 \
  --device cuda \
  --load-in-4bit \
  --logprob-chunk-size 128 \
  --overlap-top-k 16
```

说明：

- 该分析不重新 rollout，只对固定 `input_ids/response_mask` 做 student 前向。
- teacher logprob / teacher topK 已在 `train.jsonl` 中离线保存。
- student 前向使用 4bit，数值有轻微量化误差，但趋势足够明确。

## 关键统计

训练集规模：

- 样本数：`1000`
- response token 数：`676030`

teacher/student sampled-token logprob gap：

- `teacher_logprob > student_logprob`：`202219 / 676030 = 29.91%`
- `teacher_logprob < student_logprob`：`410124 / 676030 = 60.67%`
- 相等或近似相等：`63687 / 676030 = 9.42%`
- `delta_mean = teacher_logprob - student_logprob = -0.1750`
- `delta_p10 = -0.1806`
- `delta_p90 = 0.0012`
- `teacher_logprob_mean = -0.2575`
- `student_logprob_mean = -0.0826`

按样本聚合：

- `per_sample_delta_mean_p10 = -0.3494`
- `per_sample_delta_mean_p50 = -0.1657`
- `per_sample_delta_mean_p90 = -0.0874`

top16 overlap：

- `overlap_intersection_mean = 9.84 / 16`
- `overlap_ratio_mean = 0.6148`
- `overlap_ratio_p10 = 0.4375`
- `overlap_ratio_p50 = 0.625`
- `overlap_ratio_p90 = 0.8125`
- `overlap_full_ratio = 0.00013`

## 解释

### 1. teacher 和 student 的候选 token 并非完全错开

top16 平均交集约 `9.84/16`，说明 Nemotron 和当前 SFT student 在局部 token 候选上有中等偏高重叠。

这意味着问题不是“teacher tokenizer / thinking pattern 完全不兼容”，也不是 teacher topK 对 student 完全没有参考价值。

### 2. 当前 sampled-token OPD 信号整体偏负

60.67% 的 response token 上，teacher 对 student 实际生成 token 的 logprob 低于 student 自己的 logprob。

按样本聚合后，`per_sample_delta_mean_p90` 仍为负，说明这不是少数异常样本拖低，而是整体趋势。

因此当前 `distill_top_k=1` 的 OPD 训练实际经常在惩罚 student 已经高置信的原始 trajectory，而不是提供明确的替代 token 分布。

### 3. MATH500 轻微下降符合该信号形态

当前 OPD 结果：

- `SFT ckpt300 MATH500 = 0.702`
- `OPD Nemo MATH500 = 0.690`
- `GSM8K` 基本不变
- `boxed_rate` 没有崩

这更像 token-level teacher preference 不匹配导致的轻微负迁移，而不是格式崩坏或 think pattern 彻底对不上。

## 当前结论

- Nemotron teacher 和当前 SFT student 的 top16 support 有足够重叠，topK 蒸馏可以尝试。
- 但当前 topK=1 sampled-token OPD 信号明显偏负，不适合继续按原样放大。
- 最合理的下一步不是继续训更多 epoch，而是改 loss：从 sampled-token OPD 转向 teacher topK support 内的 sparse loss。

## 建议

短期实验：

- 启用 `distill_top_k=16`。
- 不再只依赖 sampled-token `teacher_logprob - student_logprob`。
- 优先实现或测试 `topk_reverse_kl`，也可先用已有 topK KD 做低风险对照。

推荐对照：

```yaml
# 当前 baseline
distill_top_k: 1
topk_kd_weight: 0.0
opd_weight: 1.0

# 低风险 topK KD 对照
distill_top_k: 16
topk_kd_weight: 0.2
opd_weight: 0.2

# Revisiting OPD 风格对照
distill_top_k: 16
topk_loss_type: reverse_kl
topk_reverse_kl_weight: 0.5
opd_weight: 0.0-0.2
```

注意：

- 当前代码已有 topK KD，但还没有严格的 topK reverse-KL。
- 如果继续使用 sampled-token OPD，应先按 gap report 过滤掉 teacher 明显不认可的 token 或样本。
- 后续每换 teacher、student 或 rollout 数据，都应重新跑本 gap 诊断。
