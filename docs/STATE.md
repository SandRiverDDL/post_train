# State

## 当前阶段

当前仓库主线已转向 `on-policy SFT`，正在从已有 `stage1 SFT` 基线切换到新的数据生成与继续训练方向。

## 当前有效基线

- 代码主包：`src/post_train/`
- 薄入口：`prepare_stage1_data.py`、`prepare_eval_data.py`、`check_sft_data.py`、`train_sft.py`、`eval_model.py`、`prepare_on_policy_sft_data.py`、`run_on_policy_loop.py`、`prepare_simpo_data.py`、`train_simpo.py`
- 配置目录：`configs/eval/`、`configs/on_policy/`、`configs/stage1/`、`configs/stage2/`、`configs/simpo/`
- 当前开发模型：`Qwen/Qwen2.5-Math-1.5B`
- 当前 `stage1` 数据口径：`UWNSL/MATH_training_split_short_cot`，首版规模 `2000`
- 当前 on-policy 数据口径：固定使用 `data/stage1_train_5000.jsonl` 作为 query registry，并保留全量轨迹归档
- 当前基础起点：`outputs/stage1_sft_5000/checkpoint-200`
- 当前评测结果目录口径：`outputs/eval/<模型路径>/<task>/result.json` 与 `raw.json`
- 暂停中的候选路线：`docs/experiments/two-stage-sft-simpo.md`

## 当前最重要的问题

1. 还缺少在 `stage1_train_5000` seen-pool 上的真实首轮 self-refine 结果，对比 `mixed_only_shortest` 与 `any_correct_shortest` 两条线。
2. 还缺少修复后多轮实验结果来判断当前默认早停口径是否有效。
3. `AIME24/AIME25` sampled pass@1 评测刚接入，仍缺少目标机器上的真实运行验证。
4. 远程数据下载在当前环境里不稳定。
5. 暂停路线仍保留在仓库里，文档与执行主线必须持续区分清楚。

## 当前优先级

1. 在目标机器上跑通 `stage1_train_5000` query registry 的单轮数据准备，确认轨迹归档与双 selector 输出正确。
2. 观察 `all_correct / mixed / all_wrong`、长度统计与 holdout 结果，比较 `mixed_only_shortest` 与 `any_correct_shortest`。
3. 验证当前 `patience=3` 的默认早停口径是否稳健。
4. 若 seen-pool self-refine 仍无提升，再扩到新的 query 池或切换 mixed SFT 主线。

## 当前 Stage1 口径

- 训练预算：`2 epoch`
- checkpoint 保存：每 `25` step 存一次
- 完成判定：训练结束后统一在冻结 `dev200` 上评测全部 checkpoints
- best 选择指标：`normalized_accuracy`

## 文档口径

- 稳定规则看 `AGENTS.md`
- 当前阶段设计看 `docs/SPEC.md`
- 暂停路线看 `docs/experiments/two-stage-sft-simpo.md`
- 结构分层看 `docs/ARCHITECTURE.md`
- 当前状态看 `docs/STATE.md`
