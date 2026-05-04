# Eval Leaderboard

本文件由 `scripts/report_eval_results.py` 自动生成；不要手动编辑表格。

未被 `docs/analysis/eval_metadata.yaml` 显式跟踪的结果只写入 registry，不进入本表；当前隐藏 41 行。

## Main Results

| method | label | math500 | gsm8k | aime24 | aime25 | notes |
|---|---|---:|---:|---:|---:|---|
| GRPO | GRPO DAPO ckpt660 | 0.7240 | 0.8605 | 0.1083 | 0.0708 | 当前 MATH500 最好主线结果 |
| SFT | SFT mix-long ckpt300 | 0.7020 | 0.8628 | 0.1021 | 0.0667 | 当前 SFT 基线 |
| Lightning-OPD | OPD Nemo from SFT ckpt300 ep1 | 0.6900 | 0.8628 | - | - | max_new_tokens=2048 后 MATH500=0.690 |
| ASFT-topK | Qwen3-1.7B BF16 LoRA ASFT-topK all-clean ckpt50 | 0.6800 | 0.8120 | - | - | 全量 Mix-Long clean<2048；dev200 选 ckpt50；MATH500/GSM8K 优于同口径 SFT/DFT，但仍有尾部 boxed 重复风险 |
| DFT | Qwen3-1.7B BF16 LoRA DFT all-clean ckpt25 | 0.6560 | 0.7923 | - | - | 全量 Mix-Long clean<2048；dev200 选 ckpt25；分数略高于 SFT 且输出显著更短 |
| SFT | Qwen3-1.7B BF16 LoRA SFT all-clean ckpt100 | 0.6440 | 0.7885 | - | - | 全量 Mix-Long clean<2048；dev200 选 ckpt100；GSM8K 存在明显 Final Answer 重复刷屏 |
| Base | Qwen2.5-Math-1.5B base | 0.3240 | 0.6975 | - | - | 历史手动记录 |
| Base | Qwen2.5-Math-1.5B base | - | - | 0.0854 | 0.0417 | 原始基座模型 |

## Task Details

### math500

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.7240 | 0.0200 | 0.9700 | 275.1 | 500 | GRPO | GRPO DAPO ckpt660 | outputs/eval/grpo_3090_dapo_server/checkpoint-660/math500/result.json |
| 0.7020 | 0.0205 | 0.9680 | 280.9 | 500 | SFT | SFT mix-long ckpt300 | outputs/eval/stage1_mix_long_sft/checkpoint-300/math500/result.json |
| 0.6900 | 0.0207 | 0.9740 | 289.0 | 500 | Lightning-OPD | OPD Nemo from SFT ckpt300 ep1 | outputs/eval/lightning_opd_nemotron_from_stage1_ckpt300_ep1/math500/result.json |
| 0.6800 | 0.0209 | 0.9020 | 565.3 | 500 | ASFT-topK | Qwen3-1.7B BF16 LoRA ASFT-topK all-clean ckpt50 | outputs/eval/stage1_qwen3_1p7b_bf16_lora_mix_long_justrl_chat_lt2048_all_consistent_asft_topk/checkpoint-50/math500/result.json |
| 0.6560 | 0.0213 | 0.9460 | 329.5 | 500 | DFT | Qwen3-1.7B BF16 LoRA DFT all-clean ckpt25 | outputs/eval/stage1_qwen3_1p7b_bf16_lora_mix_long_justrl_chat_lt2048_all_consistent_dft/checkpoint-25/math500/result.json |
| 0.6440 | 0.0214 | 0.9400 | 555.4 | 500 | SFT | Qwen3-1.7B BF16 LoRA SFT all-clean ckpt100 | outputs/eval/stage1_qwen3_1p7b_bf16_lora_mix_long_justrl_chat_lt2048_all_consistent_sft/checkpoint-100/math500/result.json |
| 0.3240 | 0.0209 | 0.5240 | 228.7 | 500 | Base | Qwen2.5-Math-1.5B base | tmp/results.md |

### gsm8k

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.8628 | 0.0095 | 0.9970 | 178.4 | 1319 | Lightning-OPD | OPD Nemo from SFT ckpt300 ep1 | outputs/eval/lightning_opd_nemotron_from_stage1_ckpt300_ep1/gsm8k/result.json |
| 0.8628 | 0.0095 | 0.9977 | 175.4 | 1319 | SFT | SFT mix-long ckpt300 | tmp/results.md |
| 0.8605 | 0.0095 | 0.9992 | 175.4 | 1319 | GRPO | GRPO DAPO ckpt660 | outputs/eval/grpo_3090_dapo_server/checkpoint-660/gsm8k/result.json |
| 0.8120 | 0.0108 | 0.9864 | 610.6 | 1319 | ASFT-topK | Qwen3-1.7B BF16 LoRA ASFT-topK all-clean ckpt50 | outputs/eval/stage1_qwen3_1p7b_bf16_lora_mix_long_justrl_chat_lt2048_all_consistent_asft_topk/checkpoint-50/gsm8k/result.json |
| 0.7923 | 0.0112 | 0.9909 | 208.7 | 1319 | DFT | Qwen3-1.7B BF16 LoRA DFT all-clean ckpt25 | outputs/eval/stage1_qwen3_1p7b_bf16_lora_mix_long_justrl_chat_lt2048_all_consistent_dft/checkpoint-25/gsm8k/result.json |
| 0.7885 | 0.0112 | 0.9962 | 618.9 | 1319 | SFT | Qwen3-1.7B BF16 LoRA SFT all-clean ckpt100 | outputs/eval/stage1_qwen3_1p7b_bf16_lora_mix_long_justrl_chat_lt2048_all_consistent_sft/checkpoint-100/gsm8k/result.json |
| 0.6975 | 0.0126 | 0.8749 | 204.4 | 1319 | Base | Qwen2.5-Math-1.5B base | tmp/results.md |

### aime24

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.1083 | 0.0445 | 0.8125 | 530.8 | 30 | GRPO | GRPO DAPO ckpt660 | outputs/eval/grpo_3090_dapo_server/checkpoint-660/aime24/result.json |
| 0.1021 | 0.0414 | 0.7896 | 528.0 | 30 | SFT | SFT mix-long ckpt300 | outputs/eval/stage1_mix_long_sft/checkpoint-300/aime24/result.json |
| 0.0854 | 0.0288 | 0.7750 | 520.7 | 30 | Base | Qwen2.5-Math-1.5B base | outputs/eval/external/home/fsw/cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/aime24/result.json |

### aime25

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.0708 | 0.0336 | 0.9208 | 495.0 | 30 | GRPO | GRPO DAPO ckpt660 | outputs/eval/grpo_3090_dapo_server/checkpoint-660/aime25/result.json |
| 0.0667 | 0.0313 | 0.9167 | 496.3 | 30 | SFT | SFT mix-long ckpt300 | outputs/eval/stage1_mix_long_sft/checkpoint-300/aime25/result.json |
| 0.0417 | 0.0178 | 0.7479 | 528.4 | 30 | Base | Qwen2.5-Math-1.5B base | outputs/eval/external/home/fsw/cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/aime25/result.json |

### math500_dev200

| score | stderr | boxed | avg tokens | samples | method | label | result |
|---:|---:|---:|---:|---:|---|---|---|
| 0.6350 | 0.0341 | 0.9500 | 295.4 | 200 | SFT | SFT mix-long ckpt300 | outputs/eval/stage1_mix_long_sft/checkpoint-300/math500_dev200/result.json |

## Ignored / Archived

| task | score | method | label | reason |
|---|---:|---|---|---|
| math500 | 0.3500 | GRPO | grpo_4090/checkpoint-383 | 异常低分历史结果，暂不进入主表 |
| gsm8k | 0.7437 | Lightning-OPD | OPD Nemo from base mistake | 错误从 base 起训，保留作反例 |
| math500 | 0.5500 | Lightning-OPD | OPD Nemo from base mistake | 错误从 base 起训，保留作反例 |
