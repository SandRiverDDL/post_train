# Experiments

这里不手写原始实验数值，统一使用机器产物：

- `experiments/registry.jsonl`：实验索引，每次训练或 loop 成功结束后追加一条
- `outputs/*/run_summary.json`：单次实验摘要
- `experiments/summary.md` / `experiments/summary.csv`：自动汇总结果

当前汇总口径：

- 主排序按 `math500` 与 `gsm8k` 的 `rank` 取平均后升序
- 只有两个 benchmark 都存在的实验进入 `Complete Runs`
- 缺失 benchmark 的实验进入 `Incomplete Runs`
- 历史回扫、缺少 `run_summary.json` 的旧实验会显示为 `source=discovered`
- 这类旧实验如果拿不到训练集路径，会显示 `train_dataset=unknown`
- `Parent` 在汇总表里使用短模型 ID 显示；完整路径仍保留在原始 `run_summary.json`

推荐流程：

1. 运行训练或自动 loop
2. 用 `scripts/collect_results.py` 汇总当前实验
3. 在 `docs/STATE.md` 里只写判断、结论和下一步，不再手抄原始数值

当前只要求 `stage1`、`on_policy`、`simpo` 接入自动登记；历史实验可按需补录。
