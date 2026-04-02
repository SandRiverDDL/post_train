# GRPO 设计说明

这份文档固定当前仓库内 `GRPO` 路线的 v1 实现口径，目标是先构造一条稳定、易验证、可迁移到 24G 4090 的数学 RL 训练链路。

## 目标

- 基于 `TRL GRPOTrainer` 建一条独立于当前 `on-policy SFT` 主线的实验路线。
- 数据由两部分混合：
  - `rd211/Big-Math-RL-Verified-Filtered`
  - 一个本地 `anchor` 题池，当前默认是 `data/on_policy_loop/query_strategy/candidate_pool.jsonl`
- reward 只保留最小必要项：
  - 答案正确：`+1.0`
  - parse 失败：`-0.5`

## 数据口径

- `rd211` 当前固定使用：
  - 数据集：`rd211/Big-Math-RL-Verified-Filtered`
  - split：`train`
- 首版筛选规则：
  - `solve_rate_lower < llama8b_solve_rate < solve_rate_upper`
  - 默认区间：`(0.25, 0.5)`
- 题面去重统一复用现有 `normalize_question`：
  - 只做连续空白折叠与首尾空白清理
  - v1 只做 exact dedup，不做 fuzzy dedup
- `rd211` 与 `anchor` 跨池混合前，也按同一套 exact dedup 去掉重题。
- 若 `anchor` 去重后数量不足以支撑目标占比，则限制 `rd211` 采样数量，而不是放弃比例约束。

## 训练口径

- 默认 loss：`dr_grpo`
- 默认关闭 reward scaling：`scale_rewards=false`
- 默认使用 `importance_sampling_level=sequence`
- 首版不手写 `GAPO`
- 首版不把原始 `grpo` 作为默认基线；若要做对照，只通过改配置切 `loss_type`

## 资源与运行假设

- `train_cheap.yaml`
  - 面向低显存单卡保守设置
  - `num_generations=4`
  - `use_vllm=false`
- `train_4090.yaml`
  - 面向 24G 4090
  - `num_generations=8`
  - `use_vllm=true`
  - `vllm_mode=colocate`
- 当前仓库检测到 `trl 0.24.0` 搭配 `vllm 0.12.0` 时，`GRPOTrainer` 在导入阶段就存在兼容风险。
- 因此 v1 强制在训练前做 preflight：
  - 无 GPU 直接失败
  - `use_vllm=true` 时若不是 `vllm==0.10.2` 直接失败

## 日志口径

- `wandb` 只记录必要指标，不打开 TRL 全量自动日志。
- 只保留：
  - `loss`
  - `reward`
  - `reward_std`
  - `frac_reward_zero_std`
  - `reward/*`
  - `rewards/*`
  - `completions/mean_length`
  - `completions/clipped_ratio`
  - `entropy`
  - `clip_ratio/region_mean`
  - `learning_rate`
- 默认按最近 `5` 次 log event 做滚动平均后再上报。
