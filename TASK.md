# Task

## 已完成

- [x] 跑通 `prepare_data -> train_sft -> eval_dataset` 的 SFT 最小闭环
- [x] 通过 tiny-overfit 证明 `1.7B + LoRA` 可以学会 `Final answer: \boxed{...}` 协议
- [x] 验证 `protocol-only` 和 `short-CoT` 都能稳定学会 boxed 协议
- [x] 验证原始长解答显著弱于 `protocol-only / short-CoT`
- [x] 新增 `2k / response 64~256` 训练集路线，并完成 1.7B SFT
- [x] 在 `train_sft_2k_short` 上完成工程验收，得到 `format_success≈0.81 / parse_success≈0.81`
- [x] 跑完整 `GSM8K dev200`
- [x] 在相同配置下补跑 base 模型的 `GSM8K dev200`，形成可解释对照
- [x] 确认当前 SFT 基线在 `GSM8K dev200 flexible-extract` 上优于 base（`0.625 > 0.47`）
- [x] 完成 GRPO v1 的最小代码骨架：数据准备、训练入口、reward contract、配置与最小单测
- [x] 修复 GRPO 初始化阶段的 `trl/peft` 兼容问题（`warnings_issued` / `add_model_tags`）
- [x] 修复 GRPO HF 4bit 路径的导入与精度参数初始化问题（`BitsAndBytesConfig`、`bf16/use_cpu`）
- [x] `TRL GRPO` 调试代码已转存到独立分支
- [x] 主干已切换为 `Unsloth GRPO` 路线

## 当前结论

- [x] 当前 SFT MVP 已完成
- [x] 当前最推荐的 SFT 冷启动数据形态是：`NuminaMath 2k + response token 64~256`
- [x] 当前不再推荐直接使用原始长解答 `3k` 作为 GRPO 冷启动唯一 SFT 数据
- [x] GRPO 主干路线已改为 `Unsloth + 单卡优先`
- [x] GRPO 主训练数据已从 `NuminaMath` 切到 `GSM8K train(main) + response token <= 128`

## 下一步

- [ ] 补跑 `MATH500 test` 正式结果，作为 Phase 1 收尾指标
- [ ] 固化 SFT 最佳配置到文档和结果记录
- [ ] 在单卡真实 GPU 环境验证 `Unsloth GRPO` 至少稳定跑过前几个训练 step
- [ ] 确认 `outputs/sft-qwen3-1.7b` 冷启动路径在 `Unsloth GRPO` 下可继续训练
- [ ] 验证切换到 `GSM8K train` 后，GRPO reward 稀疏问题是否明显缓解
- [ ] GRPO 跑通后，再补 Phase 2 对照：`base vs SFT vs GRPO`
