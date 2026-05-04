# 文档入口

本目录按“稳定规则、当前状态、运行手册、实验分析、论文参考”分层维护，避免把临时实验状态继续堆进同一个长文档。

## 当前优先阅读

- `AGENTS.md`：长期稳定规则与协作约束。
- `docs/STATE.md`：当前有效基线、已知问题和下一步重点。
- `docs/SPEC.md`：当前阶段目标、边界和不做什么。
- `docs/ARCHITECTURE.md`：系统分层、主链路和模块职责。
- `docs/analysis/eval_leaderboard.md`：当前评测总表，由 `scripts/report_eval_results.py` 生成。

## 分区口径

- `docs/runbooks/`：只写可复用操作步骤，例如训练、评测、GRPO、Lightning-OPD、MLflow、迁移。
- `docs/analysis/`：写已经发生的实验诊断和结果判读；长实验细节放这里，不放 `STATE.md`。
- `docs/refs/`：写论文、方法和外部资料笔记；不得混入“当前正在跑什么”的状态。
- `docs/design/`：写仍有参考价值的设计说明；过期路线应标注为历史方案。
- `docs/experiments/`：保留历史路线记录，不作为当前推荐入口。

## 维护规则

- 新实验完成后：先更新对应 `docs/analysis/*`，再在 `docs/STATE.md` 只同步一条当前结论。
- 新运行方式稳定后：写入 `docs/runbooks/*`；一次性 tmux 命令只作为复现实例，不写成默认入口。
- 评测结果只通过 `docs/analysis/eval_metadata.yaml` 和 `scripts/report_eval_results.py` 进入 leaderboard。
