# ConPress Qwen3 SFT/DFT 诊断

日期：2026-05-04

## 背景

- 底模：`Qwen/Qwen3-1.7B` 本地 snapshot。
- 数据：ConPress 压缩成功 prompt 子集，训练样本原始 solution 均包含 `\boxed{...}`。
- 当前训练链路已修复 completion EOS 监督：tokenization 阶段追加 tokenizer `eos_token_id` 并参与 label。
- 下表中的 SFT/DFT checkpoint 是在 no-think `assistant_prefill` 写入训练 YAML 前完成的，因此只能作为诊断结果，不代表修复后重训结果。

## 关键结果

| 模型 | checkpoint | eval prompt | max_new_tokens | acc | boxed | parse | avg tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| SFT | 75 | chat template | 2048 | 0.610 | 0.795 | 0.795 | 359.1 |
| DFT | 50 | chat template | 2048 | 0.515 | 0.605 | 0.600 | 763.8 |
| SFT | 75 | chat template + no-think prefill | 2048 | 0.590 | 0.860 | 0.860 | 346.2 |
| SFT | 75 | chat template + no-think prefill | 4096 | 0.590 | 0.910 | 0.910 | 395.3 |
| DFT | 50 | chat template + no-think prefill | 2048 | 0.600 | 0.915 | 0.915 | 289.7 |
| DFT | 50 | chat template + no-think prefill | 4096 | 0.600 | 0.945 | 0.945 | 327.7 |

## 证据

- 训练数据本身不是低 boxed 的直接原因：`solution_box_rate=1.0`，最后一行含 boxed 约 `99.67%`。
- 原始 eval 中缺 boxed 的样本几乎全部打满 `2048` token 上限，并且没有完成 `Final answer`。
- DFT 默认 chat eval 输出中 `<think>` 触发率为 `100%`，说明评测时仍在 thinking 模式。
- 加 no-think prefill 后，DFT boxed 从 `0.605` 恢复到 `0.915/0.945`，但 acc 只到约 `0.600`。

## 结论

- 低 boxed / parse 的主因是 Qwen3 thinking 模式、prompt 口径和生成长度共同导致的截断，不是训练 JSONL 普遍缺 boxed。
- acc 没有随 boxed 恢复同步上升，说明当前 ConPress 数据或训练信号仍弱于 UWLS 服务器上的 math 数据；不能只靠修 box 期待准确率自动对齐。
- 后续 ConPress Qwen3 SFT/DFT 必须在训练和评测两侧统一 no-think prompt：`assistant_prefill="<think>\n\n</think>\n\n"`，或等价使用 tokenizer `enable_thinking=false`。
- 与 UWLS 结果比较前，需要同步另一台服务器的训练 YAML、数据样本和 raw eval；仅凭 leaderboard 数字无法判断差异来自数据、prompt、长度还是训练后端。
