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

## 2026-05-05 Qwen3-1.7B SFT150 -> Instruct2507 OPD 记录

这轮实验不同于上面的 Nemotron 路线：

- student 起点：`outputs/stage1_qwen3_1p7b_bf16_lora_qwen3_4b_instruct2507_math_correct_len4096_sft_ddp3_b2_acc6_bs36/checkpoint-150`
- teacher：`Qwen/Qwen3-4B-Instruct-2507`
- rollout 数据：`data/lightning_opd/qwen3_1p7b_sft150_math_nontrunc_top16_instruct2507/train.jsonl`
- HF dataset：`data/lightning_opd/qwen3_1p7b_sft150_math_nontrunc_top16_instruct2507/hf_dataset_opd_top1`
- 训练配置：`configs/lightning_opd/train_qwen3_1p7b_sft150_math_nontrunc_top16_lr1e5.yaml`
- 训练输出：`outputs/lightning_opd_qwen3_1p7b_sft150_math_nontrunc_top16_lr1e5`

训练事实：

- 数据共 `6329` 条，source_correct 约 `5267` true、`1062` false。
- OPD 训练只用 sampled-token teacher logprob；保存的 top16 主要用于分析，不参与当前 loss。
- LR 使用 `1e-5`，LoRA rank32，三卡 DDP，有效 batch `36`，共 `176` optimizer steps。
- 训练日志无 OOM/Traceback；结束时只有 NCCL destroy warning。

gap 解释：

- 训练日志中 teacher avg logprob 均值约 `-0.365`，student avg logprob 均值约 `-0.216`，teacher-student gap 约 `-0.149`。
- 这不是直接 BUG 证据。当前轨迹来自 student rollout，且 student 已经过硬标签 SFT，对自身 token 更尖锐是正常现象。
- 该现象意味着 sampled-token OPD 的信号仍偏“约束/降温”，不是 teacher 主动给出更高概率轨迹；最终是否有效必须看 dev/benchmark，而不能只看 avg logprob。

dev 选择结果：

- 三卡 dev200 选择已完成，输出目录为 `outputs/lightning_opd_qwen3_1p7b_sft150_math_nontrunc_top16_lr1e5/dev_eval_parallel/`。
- `max_new_tokens=2048` 下 best 为 `checkpoint-10`：acc `0.635`，boxed `0.720`，avg output tokens `596.1`。
- 后续 checkpoint 快速退化：`checkpoint-20/30/40/50` acc 为 `0.540/0.425/0.405/0.385`，boxed 为 `0.590/0.460/0.450/0.435`；`checkpoint-176` acc `0.505`，boxed `0.555`。
- 判读：这轮 sampled-token OPD 在当前数据/学习率下很快损伤 boxed/parse 格式，不能按训练终点使用。由于 dev 选择用 `2048` generation cap，若要和正式 MATH500/GSM8K 对齐，应至少用 `3584` 或 `4096` 对 `checkpoint-10`、SFT 起点和少数后期 ckpt 做复核。

长度复核：

- `max_new_tokens=4096/max_seq_length=5096` 完整扫描目录为 `outputs/lightning_opd_qwen3_1p7b_sft150_math_nontrunc_top16_lr1e5/dev_eval_parallel_gen4096_len5096/`。
- 4096 口径下 `checkpoint-10` 仍最好：acc `0.675`，boxed `0.775`，parse `0.780`，tokenizer cap4096 `47/200=23.5%`。
- 后续 checkpoint 仍明显退化且更长：`checkpoint-50` acc `0.455`、boxed `0.500`、cap4096 `105/200=52.5%`；`checkpoint-150` acc `0.590`、boxed `0.655`、cap4096 `73/200=36.5%`；`checkpoint-176` acc `0.560`、boxed `0.605`、cap4096 `80/200=40.0%`。
- `checkpoint-10/50/150` 又用 `max_new_tokens=8192/max_seq_length=9216` 复核，目录为 `outputs/lightning_opd_qwen3_1p7b_sft150_math_nontrunc_top16_lr1e5/dev_eval_8192_three/`。三者 acc 为 `0.680/0.575/0.640`，boxed 为 `0.790/0.620/0.700`，cap8192 为 `44/200`、`88/200`、`62/200`。
- 8192 只把 `checkpoint-10` 从 `0.675` 提到 `0.680`，说明剩余长尾多数不是“需要更多 token 才能答对”的高质量推理，而是重复检查、反复改写、正确 boxed 后继续生成或无 boxed 收束。

EOS 与格式诊断：

- Qwen3 tokenizer EOS 为 `<|im_end|>`，但当前 OPD `train.jsonl` 中 `response_text_contains_<|im_end|>=0/6329`，`response_text_ends_<|im_end|>=0/6329`，`masked_last_eos=0/6329`。
- 普通 SFT tokenization 会在 completion 末尾追加 EOS 并监督；Lightning-OPD 数据路径是 `prompt_ids + response_ids`，当前没有追加 EOS。
- 这会削弱“最终 boxed 后停止”的监督，可能放大尾部重复；但从 4096/8192 复核看，主要失稳仍来自 sampled-token OPD 信号偏负、离线 student trajectory 与 teacher preference 不够接近、数据中约 `16.8%` source incorrect，以及 LR `1e-5` 对 LoRA OPD 偏激进。

当前判读：

- 这轮不能证明 Lightning OPD 方法本身无效，但证明本地配置不满足稳定条件：6K 级 SFT/OPD 数据、1.7B student、4B teacher、top1 sampled-token、无 EOS 监督、较高 LR 组合后，模型在极早期后开始变长和格式失稳。
- 若继续该路线，优先级应是重新生成带 EOS 的 OPD 数据、只保留 `source_correct && boxed && parse_ok && length` 合理样本、把 LR 降到 `2e-6~3e-6`，并加入 SFT anchor 或 topK support loss；不建议继续扩大当前 raw top1 OPD。
