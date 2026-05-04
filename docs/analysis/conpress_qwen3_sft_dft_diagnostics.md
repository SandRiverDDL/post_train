# ConPress Qwen3 SFT/DFT/ASFT 诊断

日期：2026-05-04

## 背景

- 底模：`Qwen/Qwen3-1.7B` 本地 snapshot。
- 数据：ConPress 压缩成功 prompt 子集，训练样本原始 solution 均包含 `\boxed{...}`。
- 当前训练链路已修复 completion EOS 监督：tokenization 阶段追加 tokenizer `eos_token_id` 并参与 label。
- 当前正式 no-think 重训已完成：训练与评测两侧都使用 Qwen3 chat template，并写入 `assistant_prefill="<think>\n\n</think>\n\n"`。

## no-think 重训前诊断

| 模型 | checkpoint | eval prompt | max_new_tokens | acc | boxed | parse | avg tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| SFT | 75 | chat template | 2048 | 0.610 | 0.795 | 0.795 | 359.1 |
| DFT | 50 | chat template | 2048 | 0.515 | 0.605 | 0.600 | 763.8 |
| SFT | 75 | chat template + no-think prefill | 2048 | 0.590 | 0.860 | 0.860 | 346.2 |
| SFT | 75 | chat template + no-think prefill | 4096 | 0.590 | 0.910 | 0.910 | 395.3 |
| DFT | 50 | chat template + no-think prefill | 2048 | 0.600 | 0.915 | 0.915 | 289.7 |
| DFT | 50 | chat template + no-think prefill | 4096 | 0.600 | 0.945 | 0.945 | 327.7 |

这些 checkpoint 是在 no-think `assistant_prefill` 写入训练 YAML 前完成的，只用于定位 boxed/parse 异常来源。

## no-think 重训后结果

| 模型 | best ckpt | dev200 acc | dev200 boxed | dev200 avg tokens | MATH500 | GSM8K | benchmark boxed | benchmark avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 175 | 0.625 | 0.830 | 347.1 | 0.652 | 0.7741 | 0.840 / 0.977 | 327.7 / 108.8 |
| DFT | 50 | 0.605 | 0.885 | 281.6 | 0.632 | 0.7665 | 0.906 / 0.986 | 239.5 / 91.3 |

判读：

- ConPress SFT 并不弱，benchmark 结果已经接近或略高于此前 UWLS SFT 口径。
- ConPress DFT 输出更短、boxed 更高，但准确率低于 ConPress SFT，也低于 UWLS DFT 的相对收益。
- 因此当前问题不再是“没有学会 boxed”，而是 ConPress 数据分布和 DFT 训练信号没有给准确率带来正增益。

## 训练数据长度分布

统计文件：`data/stage1/conpress_qwen3_4b_nt_correct_compressed/train.jsonl`，使用 Qwen3 chat/no-think tokenization 统计 `prompt + solution`。

| n | mean | p50 | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 6660 | 690.8 | 369 | 1665 | 2455 | 4182 | 6313 |

| 阈值 | 数量 | 比例 |
|---:|---:|---:|
| >1024 | 1237 | 18.57% |
| >1536 | 758 | 11.38% |
| >2048 | 487 | 7.31% |
| >3072 | 179 | 2.69% |
| >4096 | 71 | 1.07% |

若后续为了 ASFT/OPD 稳定性改用 `max_seq_length=2048`，会丢弃约 `7.31%` 样本，不是小于 `5%`，但代价仍可接受。

## 剩余失败题 normal rollout

统计目录：`data/rollout/teacher_sft/qwen3_4b_nt_remaining_failed837_normal_spp1_max8192_20260504`。

口径：837 个 ConPress 失败 prompt，每题 1 条正常 eval prompt rollout，`max_new_tokens=8192`，`max_model_len=12288`，Qwen3 chat template + `enable_thinking=false`。

| n | mean tokens | p50 | p75 | p90 | max | finish=length | boxed | parse_ok | correct |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 837 | 5777.7 | 7869 | 8192 | 8192 | 8192 | 410 | 57.11% | 56.87% | 14.22% |

| level | n | mean tokens | p50 | trunc | correct |
|---|---:|---:|---:|---:|---:|
| L1 | 29 | 2359.1 | 1514 | 3.4% | 17.2% |
| L2 | 70 | 2653.1 | 1130 | 12.9% | 8.6% |
| L3 | 105 | 3263.1 | 2184 | 15.2% | 21.0% |
| L4 | 131 | 4998.8 | 5412 | 38.9% | 19.1% |
| L5 | 502 | 7140.0 | 8192 | 66.3% | 12.2% |

结论：该 hard-tail batch 不是高质量 SFT/ASFT/OPD 数据源。它可以作为失败分析或多 attempt 候选池，但只能抽取 `boxed && parse_ok && correct` 且长度合理的轨迹，不能 raw 混入训练。

## ASFT-topK 状态

- 配置：`configs/stage1/conpress_qwen3_1p7b_asft_topk.yaml`。
- 口径：`qlora_4bit`、`loss_mode=asft_topk`、`topK=32`、`kl_weight=0.03`、`max_seq_length=4096`。
- 首次四卡 `batch_size=2` OOM，日志归档为 `logs/train/conpress_qwen3_1p7b_asft_topk_4gpu_oom_bsz2.log`。
- 最终运行口径改为 `batch_size=1`、`gradient_accumulation_steps=8`、四卡有效 batch 仍为 `32`，日志为 `logs/train/conpress_qwen3_1p7b_asft_topk_4gpu.log`。
- 并行 dev200 选择 best 为 `checkpoint-175`：acc `0.645`，boxed `0.830`，avg tokens `368.4`。
- Benchmark：MATH500 `0.654`，GSM8K `0.8006`；boxed `0.850 / 0.9848`；平均输出 `333.8 / 109.7` tokens。

| 模型 | best ckpt | MATH500 | GSM8K | boxed | avg tokens |
|---|---:|---:|---:|---:|---:|
| SFT | 175 | 0.652 | 0.7741 | 0.840 / 0.977 | 327.7 / 108.8 |
| DFT | 50 | 0.632 | 0.7665 | 0.906 / 0.986 | 239.5 / 91.3 |
| ASFT-topK | 175 | 0.654 | 0.8006 | 0.850 / 0.9848 | 333.8 / 109.7 |

判读：ASFT-topK 没有显著提高 MATH500，但 GSM8K 明显强于 SFT/DFT，且没有 DFT 的准确率退化。

## OPD 数据准备状态

- 当前基于 ASFT `checkpoint-175` 准备 Lightning-OPD sampled-token 数据：`configs/lightning_opd/conpress_asft_qwen3_4b_teacher_math2000.yaml`。
- 输出目录：`data/lightning_opd/conpress_asft_qwen3_4b_teacher_math2000_20260504`。
- prompt source：`data/lightning_opd/query_sources/conpress_asft_qwen3_chat_math2000_seed42.jsonl`，共 2000 条 Hendrycks MATH train query。
- prompt 已提前渲染为 Qwen3 chat template，并追加 no-think prefill：`<think>\n\n</think>\n\n`。
- student：`outputs/stage1_conpress_qwen3_1p7b_asft_topk/checkpoint-175`；teacher：`Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`。
- 4 shard student raw rollout 已完成：每个 shard `500` 条，共 `2000` 条。
- BF16 teacher sampled-token forward 已完成：每个 shard `500` 条，共 `2000` 条。
- 已修复 vLLM LoRA rank 配置：ASFT adapter rank 为 `32`，必须传 `max_lora_rank=32`，不能使用 vLLM 默认 `16`。
- 当前 teacher scoring 已改为 sampled-token `target_logit - logsumexp(logits)`，本轮配置 `top_k=0`、`teacher_load_in_4bit=false`，不保存 teacher topK。
- 合并产物：`train.jsonl` 共 `2000` 行，`shape_errors=0`，`top_k_values=[0]`；response token 分布为均值 `630.8`、p50 `329`、p75 `790`、p90/p95/p99/max 均为 `2048`。

## OPD 训练结果

- 不过滤版训练配置：`configs/lightning_opd/train_conpress_asft_qwen3_4b_teacher_math2000.yaml`。
- 输出目录：`outputs/lightning_opd_conpress_asft_qwen3_4b_teacher_math2000_unfiltered`。
- 口径：起点为 ConPress ASFT `checkpoint-175`，四卡 `0,1,2,4`，`batch_size=1`、`gradient_accumulation_steps=8`，有效 batch `32`，`save_steps=10`，共 `63` optimizer steps。
- 并行 dev200 选择 best 为 `checkpoint-30`：acc `0.630`，boxed `0.785`，avg tokens `459.3`。
- Benchmark：MATH500 `0.678`，GSM8K `0.8135`；boxed `0.824 / 0.9750`；平均输出 `394.2 / 138.5` tokens。
- 训练日志显示 `avg_teacher_logprob` 多数低于 `avg_student_logprob`，sampled-token advantage 多为负；尽管最终准确率高于 ASFT 起点，但 MATH500 boxed rate 下降，说明不过滤 OPD 有收益但格式风险仍在。

| 模型 | best ckpt | MATH500 | GSM8K | boxed | avg tokens |
|---|---:|---:|---:|---:|---:|
| SFT | 175 | 0.652 | 0.7741 | 0.840 / 0.977 | 327.7 / 108.8 |
| DFT | 50 | 0.632 | 0.7665 | 0.906 / 0.986 | 239.5 / 91.3 |
| ASFT-topK | 175 | 0.654 | 0.8006 | 0.850 / 0.9848 | 333.8 / 109.7 |
| Lightning-OPD unfiltered | 30 | 0.678 | 0.8135 | 0.824 / 0.9750 | 394.2 / 138.5 |

## 证据

- 训练数据本身不是低 boxed 的直接原因：`solution_box_rate=1.0`，最后一行含 boxed 约 `99.67%`。
- 原始 eval 中缺 boxed 的样本几乎全部打满 `2048` token 上限，并且没有完成 `Final answer`。
- DFT 默认 chat eval 输出中 `<think>` 触发率为 `100%`，说明评测时仍在 thinking 模式。
- 加 no-think prefill 后，DFT boxed 从 `0.605` 恢复到 `0.915/0.945`，但 acc 只到约 `0.600`。

## 结论

- 低 boxed / parse 的历史主因是 Qwen3 thinking 模式、prompt 口径和生成长度共同导致的截断，不是训练 JSONL 普遍缺 boxed。
- no-think 重训后，ConPress SFT 的准确率可用；ConPress DFT 没有复现 UWLS 上“DFT 明显强于 SFT”的现象；ConPress ASFT-topK 主要收益体现在 GSM8K。
- 不过滤 Lightning-OPD 在 ConPress ASFT 起点上带来准确率正收益，但同时降低 MATH500 boxed rate、拉长输出；后续优先考虑过滤 echo/repetition/no-box/truncation、改 topK support 或更换 teacher，而不是直接扩大同一 sampled-token OPD。
- Qwen3-4B Non-Thinking teacher 可用于候选生成或 OPD 诊断，但 hard-tail 输出过长且正确率低，不能默认代表高质量教师分布。
