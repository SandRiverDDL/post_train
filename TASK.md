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
- [x] 评测配置已拆分为 `configs/eval.yaml`，评测脚本默认不再依赖 `configs/sft.yaml`
- [x] 正式评测默认切换为本地 boxed/math-verify 链路，native benchmark 改为可选对照模式
- [x] GRPO 数据准备链已收敛为 `candidates -> scored -> select` 三阶段，`select_grpo_scored_subset.py` 默认输出全部符合阈值样本，传入 `target-size` 时再做可选抽样

## 下一步

- [ ] 分析并修复当前 `GRPO` 在正式 boxed 评测下 early checkpoint 优于 final 的问题
- [x] 落地下一轮 GRPO 的 `_base_` 配置继承，并支持短程消融
- [x] 增加按 step 保存 checkpoint 的能力，并补充顺序评测脚本
- [ ] 将短程 GRPO + 顺序评测 `checkpoint-10/20/25/30` 固化为默认实验流程
- [ ] 用当前模型与 prompt 重新打分筛题，避免继续复用旧模型口径的 middiff 工件
- [ ] 验证更窄的难度带宽、更高的 `num_generations`、更宽的候选长度上限是否能提高组内信号
- [ ] 评估是否需要把单次训练窗口收紧到 `20~30 step`，而不是继续默认 `50/100 step`
- [ ] 固化离线难度筛选流程，继续使用 scored/filtered 路线而不是回退随机 short 子集
- [ ] 修正 `eval_grpo_checkpoints.py`，使 checkpoint 顺序评测默认走 `official` 口径，而不是旧 harness benchmark
- [ ] 在目标函数问题收敛后，再补 Phase 2 对照：`base vs SFT vs GRPO`
- [ ] `MATH500 test` 留到 GRPO 方向稳定后再补跑
