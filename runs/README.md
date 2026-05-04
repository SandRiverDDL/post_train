# 运行脚本入口

`runs/` 只放可复用 shell 编排；Python CLI 入口放 `scripts/`，核心逻辑放 `src/post_train/`。

## 当前保留入口

- `runs/sft/train_sft_ddp.sh`：多卡 SFT/DFT/ASFT/Lightning-OPD 训练入口。
- `runs/lightning_opd/prepare_candidate.sh`：Lightning-OPD prompt、student rollout、teacher forward、merge 编排。
- `runs/rollout/justrl_math_shards.sh`：JustRL MATH rollout 分片编排。
- `runs/rollout/watch_justrl_math_shards.sh`：rollout 分片日志监控。
- `runs/rollout/justrl_math_merge.sh` 与 `runs/rollout/justrl_math_select_sft.sh`：历史 rollout 产物的 merge/select 单步入口。

## 归档规则

- 一次性后台流水线、固定日期实验脚本、只服务某个 checkpoint 的评测脚本，移动到 `runs/archive/<date>/`。
- 归档脚本只用于复现历史命令，不作为当前推荐入口。
- 新增 shell 前先判断是否可以用现有 Python CLI + YAML + 环境变量表达；能表达就不要新增脚本。
