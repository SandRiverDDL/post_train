# Eval Leaderboard

本文件由 `scripts/report_eval_results.py` 自动生成；不要手动编辑表格。

未被 `docs/analysis/eval_metadata.yaml` 显式跟踪的结果只写入 registry，不进入本表；当前隐藏 2 行。

## Main Results

| method | label | math500 | gsm8k | aime24 | aime25 | notes |
|---|---|---:|---:|---:|---:|---|
| SFT | SFT mix-long ckpt300 | 0.7020 | - | 0.1021 | 0.0667 | 当前 SFT 基线 |
| Lightning-OPD | ConPress ASFT OPD unfiltered ckpt30 | 0.6780 | 0.8135 | - | - | ConPress ASFT ckpt175 起点；2000 条不过滤 sampled-token OPD；dev200 选 ckpt30；MATH500/GSM8K 高于 ASFT，但 MATH500 boxed 下降 |
| ASFT-topK | ConPress Qwen3-1.7B ASFT-topK ckpt175 | 0.6540 | 0.8006 | - | - | ConPress 成功压缩数据 QLoRA ASFT-topK；dev200 选 ckpt175；GSM8K 优于 SFT/DFT |
| SFT | ConPress Qwen3-1.7B SFT ckpt175 | 0.6520 | 0.7741 | - | - | ConPress 成功压缩数据 no-think 重训；dev200 选 ckpt175 |
| DFT | ConPress Qwen3-1.7B DFT ckpt50 | 0.6320 | 0.7665 | - | - | ConPress 成功压缩数据 no-think 重训；dev200 选 ckpt50；准确率低于 SFT |

## Task Details

### math500

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.7020 | 0.0205 | 0.9680 | 280.9 | 500 | SFT | SFT mix-long ckpt300 | outputs/eval/stage1_mix_long_sft/checkpoint-300/math500/result.json |
| 0.6780 | 0.0209 | 0.8240 | 394.2 | 500 | Lightning-OPD | ConPress ASFT OPD unfiltered ckpt30 | outputs/eval/lightning_opd_conpress_asft_qwen3_4b_teacher_math2000_unfiltered/checkpoint-30/math500/result.json |
| 0.6540 | 0.0213 | 0.8500 | 333.8 | 500 | ASFT-topK | ConPress Qwen3-1.7B ASFT-topK ckpt175 | outputs/eval/stage1_conpress_qwen3_1p7b_asft_topk/checkpoint-175/math500_test/result.json |
| 0.6520 | 0.0213 | 0.8400 | 327.7 | 500 | SFT | ConPress Qwen3-1.7B SFT ckpt175 | outputs/eval/stage1_conpress_qwen3_1p7b_sft/checkpoint-175/math500_test/result.json |
| 0.6320 | 0.0216 | 0.9060 | 239.5 | 500 | DFT | ConPress Qwen3-1.7B DFT ckpt50 | outputs/eval/stage1_conpress_qwen3_1p7b_dft/checkpoint-50/math500_test/result.json |

### gsm8k

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.8135 | 0.0107 | 0.9750 | 138.5 | 1319 | Lightning-OPD | ConPress ASFT OPD unfiltered ckpt30 | outputs/eval/lightning_opd_conpress_asft_qwen3_4b_teacher_math2000_unfiltered/checkpoint-30/gsm8k/result.json |
| 0.8006 | 0.0110 | 0.9848 | 109.7 | 1319 | ASFT-topK | ConPress Qwen3-1.7B ASFT-topK ckpt175 | outputs/eval/stage1_conpress_qwen3_1p7b_asft_topk/checkpoint-175/gsm8k_test/result.json |
| 0.7741 | 0.0115 | 0.9773 | 108.8 | 1319 | SFT | ConPress Qwen3-1.7B SFT ckpt175 | outputs/eval/stage1_conpress_qwen3_1p7b_sft/checkpoint-175/gsm8k_test/result.json |
| 0.7665 | 0.0117 | 0.9864 | 91.3 | 1319 | DFT | ConPress Qwen3-1.7B DFT ckpt50 | outputs/eval/stage1_conpress_qwen3_1p7b_dft/checkpoint-50/gsm8k_test/result.json |

### aime24

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.1021 | 0.0414 | 0.7896 | 528.0 | 30 | SFT | SFT mix-long ckpt300 | outputs/eval/stage1_mix_long_sft/checkpoint-300/aime24/result.json |

### aime25

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.0667 | 0.0313 | 0.9167 | 496.3 | 30 | SFT | SFT mix-long ckpt300 | outputs/eval/stage1_mix_long_sft/checkpoint-300/aime25/result.json |

### math500_dev200

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.6350 | 0.0341 | 0.9500 | 295.4 | 200 | SFT | SFT mix-long ckpt300 | outputs/eval/stage1_mix_long_sft/checkpoint-300/math500_dev200/result.json |
