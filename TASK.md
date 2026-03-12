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

## 当前结论

- [x] 当前 SFT MVP 已完成
- [x] 当前最推荐的 SFT 冷启动数据形态是：`NuminaMath 2k + response token 64~256`
- [x] 当前不再推荐直接使用原始长解答 `3k` 作为 GRPO 冷启动唯一 SFT 数据
- [x] GRPO 已经不再卡在“脚本无法启动”的阶段，当前阻塞点收敛到模型加载后的 dtype/量化兼容问题

## 下一步

- [ ] 补跑 `MATH500 test` 正式结果，作为 Phase 1 收尾指标
- [ ] 固化 SFT 最佳配置到文档和结果记录
- [ ] reviewer 先审 `train_grpo.py` 的 dtype / 量化兼容问题，给出是否需要切基座的结论
- [ ] 在 reviewer 结论基础上修复 GRPO 真实 GPU 环境下的 `float != bfloat16` 报错
- [ ] 修复后重新验证 `scripts/train_grpo.py --config configs/grpo.yaml` 至少稳定跑过前几个训练 step
- [ ] GRPO 跑通后，再补 Phase 2 对照：`base vs SFT vs GRPO`
