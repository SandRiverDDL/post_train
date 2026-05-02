# State

## 当前阶段

当前仓库的实际工作重心已切到 `GRPO / DAPO-lite` 实验，`on-policy SFT` 与旧的两阶段 `SFT + SIMPO` 保留为历史路线与候选路线，不再作为当前主线。

当前默认工作环境为服务器仓库 `~/dataY/post_train`；文档、代码、配置与实验结果都以服务器副本为准。

## 当前有效基线

- 代码主包：`src/post_train/`
- 当前正式开发与实验环境：服务器仓库 `~/dataY/post_train`
- 当前开发模型：`Qwen/Qwen2.5-Math-1.5B`
- 当前 `GRPO` 基础起点：`outputs/stage1_mix_long_sft/checkpoint-300`
- 当前 `GRPO` 训练数据：`data/grpo/train.jsonl`
- 当前 `GRPO` 双卡入口：`scripts/run_grpo_3090_dapo_server.py`
- 当前 `GRPO` 双卡编排：`src/post_train/grpo/server.py`
- 当前单轮实验编排入口：`scripts/run_experiment_workflow.py`
- 当前评测结果目录口径：`outputs/eval/<模型路径>/<task>/result.json` 与 `raw.json`
- 当前评测总表：`docs/analysis/eval_leaderboard.md`
- 当前评测机器索引：`docs/analysis/eval_registry.jsonl`
- 当前评测 metadata：`docs/analysis/eval_metadata.yaml`
- 当前 checkpoint 选择输出口径：`<train_output_dir>/dev_eval/dev_ranking.json` 与 `best_checkpoint.json`
- 当前 `select_sft_checkpoint.py` / `eval_model.py` 已补齐 `model_resolution` 诊断，可明确记录：
  - `pretrained`
  - `enable_lora`
  - `lora_local_path`
- 当前 `train_grpo.py` 已补齐 `grpo.adapter.*` 诊断，可明确记录：
  - 是否从 adapter checkpoint 继续训练
  - LoRA 参数数量
  - 训练时活跃 adapter 与底模路径

## 当前已确认事实

- ssh -R 反向隧道会出现“假活”状态：本地 ssh 进程仍在、服务器 127.0.0.1:17897 端口也可连接，但真实 HTTP 请求已经超时；代理是否真的可用，必须用 curl -I -m 15 https://www.google.com 这类实际请求验证，而不能只看进程或端口。
- `GRPO` 训练与评测链路都能确认 `LoRA` 已正确挂上，不是静默退化成 `base model`
- 先前某轮 dev 结果异常偏低，主因是评测时 `max_new_tokens=512` 导致大量截断，而不是 checkpoint 没挂 LoRA
- 当前 `stage1_mix_long_sft/checkpoint-300` 在 `math500_dev200` 上约为 `0.635 ~ 0.645`
- 当前 `grpo_3090_dapo_server/checkpoint-30` 在 `math500_dev200` 上约为 `0.63`
- 以 `math500_dev200` 的 `200` 题规模来看，这一档差异仍处在高噪声区间，不能据此认定 `GRPO` 已明显优于或劣于当前 SFT 基线
- 当前 JustRL rollout 初筛暴露明显长度问题：`max_new_tokens=2048` 下 raw response token p50 贴近 2048，存在大量正确 boxed 后继续输出并被截断的轨迹；这类样本不应无过滤地当作高质量 SFT 数据。
- 当前可用于快速对照的 JustRL/Mix-Long 同题 SFT 数据为 `data/rollout/math_sft/justrl_deepseek_1500/train.rollout_rejection_lt2000_matched321.jsonl` 与 `train.mix_long_lt2000_matched321.jsonl`，筛选口径是 JustRL 正确轨迹 `solution_tokens < 2000` 且能在 Mix-Long 中精确匹配同题 prompt。
- 当前 JustRL rollout 脚本已支持 `shard -> merge -> select` 一键链路，可用两个独立 vLLM 进程做数据并行；默认 MATH train 全量 7500 题、每题 1 轨迹、`max_model_len=4096`、`max_new_tokens=3650`，主入口为 `runs/rollout/justrl_math_shards.sh`，产物按 `shards/`、`raw/`、`sft/`、`logs/` 分目录保存，总 report 为输出目录根部 `report.json`。
- JustRL rollout 详细诊断见 `docs/analysis/justrl_rollout_diagnostics.md`：主要问题是 `<think>` 格式不稳定、`double_boxed` 普遍、简单题过度推理、长组合/几何题容易截断，不建议 raw rollout 直接作为 SFT 数据。
- 当前主环境中 `unsloth 2025.9.9 + trl 0.26.2` 会在 import 阶段生成非法 `UnslothGRPOTrainer.py`，SFT 小实验短期已切到 `backend=trl_peft`，避免被 Unsloth 的 GRPO patch 兼容问题阻塞；后续若继续使用 Unsloth，应单独整理兼容环境。
- 当前评测结果维护已统一到 `scripts/report_eval_results.py`：自动扫描 `outputs/eval/**/result.json` 并全量写入 registry；`docs/analysis/eval_metadata.yaml` 只作为人工展示白名单，维护重要模型、关键 baseline、效果好的 checkpoint 与明确归档的坏例。
- 当前主结果口径以 `docs/analysis/eval_leaderboard.md` 为准；`math500_dev200` 这类 dev-only 结果只进入 Task Details，不进入 Main Results。
- 当前已记录的主线结果：`outputs/grpo_3090_dapo_server/checkpoint-660` 在 MATH500 为 `0.7240`，`outputs/stage1_mix_long_sft/checkpoint-300` 在 MATH500 为 `0.7020`，`outputs/lightning_opd_nemotron_from_stage1_ckpt300_ep1` 在 MATH500 为 `0.6900`。
- 当前 Lightning OPD gap 诊断见 `docs/analysis/lightning_opd_gap_diagnostics.md`：Nemotron 和 SFT student 的 top16 overlap 平均为 `9.84/16`，但 `teacher_logprob < student_logprob` 的 token 占 `60.67%`，说明 topK support 有重叠但 sampled-token OPD 信号整体偏负；继续放大 topK=1 OPD 风险较高。

## 当前最重要的问题

1. `GRPO / DAPO-lite` 目前尚未在 `math500_dev200` 上稳定超过 `stage1_mix_long_sft/checkpoint-300`。
2. `math500_dev200` 只有 `200` 题，单次评测标准误差约在 `0.034` 档，当前小幅波动不足以支持强结论。
3. `select_sft_checkpoint.py` 当前逐 checkpoint 重新创建 `LLM(...)`，导致反复冷启动 `vLLM`，checkpoint 选择耗时过高。
4. `GRPO` 早期训练仍存在格式习惯容易被冲坏的风险，需要继续观察：
   - `boxed_rate`
   - `parse_success_rate`
   - `completions/clipped_ratio`
   - `entropy`
5. JustRL 采样轨迹用于 SFT 前必须加强长度与完整性过滤；当前 parser 会读取第一个 `\boxed{...}`，不能单独证明生成轨迹完整。
6. Lightning OPD 当前 topK=1 sampled-token 信号偏负，下一步若继续 OPD，应优先实现或测试 topK support loss，而不是直接增加 epoch。
7. 旧路线文档仍在仓库中保留，必须继续和当前 `GRPO` 主线区分，不自动视为当前推荐配置。

## 当前优先级

1. 把 `GRPO / DAPO-lite` 的 checkpoint 选择与 benchmark 评测跑顺，先确认单 seed 下是否至少不差于当前 SFT 基线。
2. 优化 checkpoint 选择链路，避免每个 checkpoint 反复重启 `vLLM`。
3. 继续盯 `boxed_rate / parse_success_rate / clipped_ratio / entropy`，确认当前 reward 与长度设置不会在早期破坏格式输出。
4. 只有当单 seed 结果达到“至少不差于基线”时，再考虑做 `3 seed` 复验，而不是现在就对每组配置做多 seed 全覆盖。

## 当前 GRPO 口径

- 当前默认双卡路线：`1 训 1 推`
  - trainer 优先 `GPU 6`
  - `vLLM server` 优先 `GPU 7`
  - 若 `6/7` 空余显存不足，再回退到其他空闲卡对
  - 指定 `--vllm-port` 时，wrapper 会同步覆盖 trainer 使用的 `vLLM server` 地址
- 当前 DAPO-lite 配置：
  - `vllm_mode=server`
  - `loss_type=dapo`
  - `epsilon=0.2`
  - `epsilon_high=0.28`
  - `mask_truncated_completions=true`
  - `max_completion_length=768`
  - `soft_overlong.weight=0.05`
  - `soft_overlong.cache_tokens=192`
- 当前 `GRPO` 判读优先级：
  - 先看 `math500_dev200`
  - 再看 `math500_test` 与 `gsm8k_test`
  - 不允许用 benchmark test 集反向挑 checkpoint

## 文档口径

- 稳定规则看 `AGENTS.md`
- 当前阶段设计看 `docs/SPEC.md`
- 结构分层看 `docs/ARCHITECTURE.md`
- 当前状态看 `docs/STATE.md`
- 评测结果总表看 `docs/analysis/eval_leaderboard.md`
- 评测结果维护命令看 `docs/runbooks/eval.md`
- `GRPO` 运行与操作说明看 `docs/runbooks/grpo.md`
- 旧路线实验记录看 `docs/experiments/README.md` 与 `docs/experiments/two-stage-sft-simpo.md`
