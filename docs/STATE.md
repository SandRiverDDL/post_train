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
- 当前 MLflow 索引：`mlruns/`，默认 backend 为 `sqlite:///mlruns/mlflow.db`
- 当前 checkpoint 选择输出口径：`<train_output_dir>/dev_eval/dev_ranking.json` 与 `best_checkpoint.json`
- 当前多 GPU checkpoint 选择 wrapper：`scripts/select_sft_checkpoint_parallel.py`，输出到 `<train_output_dir>/dev_eval_parallel/`；它按 checkpoint shard 启动多个现有串行 selector 子进程，适合多卡空闲时并行筛选。
- 当前 SFT/DFT tokenization 已在 completion 末尾追加 tokenizer `eos_token_id` 并参与 label 监督，用于修复 `Final answer / boxed` 后不会停止的问题；原始 JSONL 仍保持不写特殊 token。
- 当前 `eval_model.py` / `select_sft_checkpoint.py` 的 vLLM 评测链路已支持 `prompt_style/use_chat_template/system_prompt/assistant_prefill`，Qwen3 chat 训练模型必须用 chat-template eval，避免 train/eval prompt 格式错配。
- 当前 ConPress Qwen3-1.7B SFT/DFT 配置与 `configs/eval/qwen3_1p7b_dev.yaml` 已统一写入 `assistant_prefill="<think>\n\n</think>\n\n"`，用于禁用 Qwen3 thinking；这只影响后续训练/评测，不能 retroactively 改变已经训练完成的 checkpoint。
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
- JustRL / Nemotron 这类 Qwen chat 模型做 rollout 时必须显式使用 tokenizer chat template；裸 prompt 结果只作为弱参考或无效 probe。此前 JustRL-Nemotron probe50 使用 `scripts/rollout_math_sft.py` 裸 prompt，导致 `<think>` / `</think>` 边界不稳定，不能作为该模型真实格式能力结论。
- 已重跑 JustRL-Nemotron chat-template 50 条 probe：`data/rollout/math_sft/justrl_nemotron_chat_probe50_len8192/raw/raw_samples.merged.jsonl`。`<think>` 从裸 prompt 的 `8/50` 提升到 `50/50`，`</think>` 为 `42/50`；输出平均 `2805.5` tokens，`p95=5071.5`，max `6099`，未命中 `8192` 上限。结论是 chat template 修复了起始格式，但输出仍偏长，不能直接 raw SFT。
- JustRL rollout 详细诊断见 `docs/analysis/justrl_rollout_diagnostics.md`：主要问题是长推理偏置、`double_boxed` 普遍、简单题过度推理、长组合/几何题容易截断；其中 `<think>` 格式不稳定的结论必须区分是否使用了 chat template，不建议 raw rollout 直接作为 SFT 数据。
- 当前 JustRL 对齐的 Mix-Long SFT/DFT 小样本数据为 `data/stage1/mix_long_justrl_chat_lt2048_random2000/train.jsonl`：Mix-Long 归一化保留原有 boxed，只有缺 boxed 时追加裸 `\boxed{...}`；随后按 `justrl_math` prompt、`use_chat_template=true`、`system_prompt=""` 计算 `prompt + solution <= 2048` 后 seed=42 随机抽 2000 条；对应配置为 `configs/stage1/mix_long_justrl_chat_lt2048_random2000_sft.yaml` 与 `configs/stage1/mix_long_justrl_chat_lt2048_random2000_dft.yaml`。
- 当前 JustRL 对齐的 Mix-Long 全量 clean 数据为 `data/stage1/mix_long_justrl_chat_lt2048_all_consistent/train.jsonl`：同样按 `justrl_math + chat template + prompt + solution <= 2048` 过滤，从 5383 条中保留 5169 条长度合格样本，再剔除 50 条 `final_answer` 占位或与最后 boxed 不一致样本，最终 5119 条；对应配置为 `configs/stage1/mix_long_justrl_chat_lt2048_all_consistent_sft.yaml`、`configs/stage1/mix_long_justrl_chat_lt2048_all_consistent_dft.yaml`，Qwen3-1.7B 主线配置另有 `qwen3_1p7b_*_sft/dft/asft_topk.yaml`。
- 当前 SFT 的 `backend=trl_peft` 支持 `torchrun` 多卡 DP/DDP，入口为 `runs/sft/train_sft_ddp.sh`。三卡示例：`GPUS=0,1,2 CONFIG=configs/stage1/mix_long_justrl_chat_lt2048_random2000_sft.yaml runs/sft/train_sft_ddp.sh`。DDP 下有效 batch 为 `batch_size * gradient_accumulation_steps * GPU 数量`；若要保持单卡等效 batch，需要同步调小 `gradient_accumulation_steps` 或 `batch_size`。
- Qwen3-1.7B BF16 LoRA 全量 clean<2048 对照已完成，模型基座为本地 `Qwen/Qwen3-1.7B`，数据为 `data/stage1/mix_long_justrl_chat_lt2048_all_consistent/train.jsonl`，训练三卡 DDP，`batch_size=4`、`gradient_accumulation_steps=3`、`save_steps=25`、`max_seq_length=2048`。SFT dev200 最优为 `checkpoint-100`，DFT dev200 最优为 `checkpoint-25`。当前已实现 `loss_mode=asft_topk`，使用 LoRA `disable_adapter()` 取 base topK reference，默认 `asft_top_k=32`、`asft_kl_weight=0.03`，配置为 `configs/stage1/qwen3_1p7b_mix_long_justrl_chat_lt2048_all_consistent_asft_topk.yaml`。
- Qwen3-1.7B BF16 LoRA 全量 clean<2048 benchmark：SFT `checkpoint-100` 在 MATH500/GSM8K 为 `0.644 / 0.7885`，平均输出 `555.4 / 618.9` tokens；DFT `checkpoint-25` 为 `0.656 / 0.7923`，平均输出 `329.5 / 208.7` tokens；ASFT-topK dev200 最优 `checkpoint-50` 为 `0.680 / 0.8120`，平均输出 `565.3 / 610.6` tokens。当前同口径下 ASFT-topK 明显优于 SFT/DFT，但输出长度接近 SFT。
- 该 Qwen3 SFT/ASFT 评测暴露明显尾部重复风险：SFT GSM8K 首题生成 `Final Answer: \boxed{26}` 大量重复直到接近截断；ASFT `checkpoint-50` 也会出现 `Final Answer: \boxed{...}` 或整段推理重复，启发式统计 MATH500 约 `144/500`、GSM8K 约 `599/1319` 有多 boxed/Final Answer 重复嫌疑。训练集 `5119` 条中 `Final Answer` 多次出现约 `297` 条、尾部相同 boxed 重复约 `9` 条，说明问题不是数据全集崩坏，而是格式噪声放大后触发生成循环。后续应对训练数据尾部去重，并在评测中同时关注准确率、长度和重复率。
- 当前 ConPress 压缩策略见 `docs/analysis/conpress_compression_policy.md`，实验细节见 `docs/analysis/conpress_probe_diagnostics.md`。主文本桶仍用 `exclude_visual + balanced_level + default prompt`，但 visual/asy 不再一刀切丢弃；Qwen3-4B visual 消融为 `format_ok=12/12`、`parse=36/36`、`boxed=36/36`、`correct=25/36`，visual 子集 `8/12` 正确。当前建议 visual/asy 单独成桶，强过滤后按 `5%-10%` 混入。
- Qwen3-4B-2507 Thinking 5k 数据 `data/rsr/qwen3_4b_2507/short-sys_5k_gen1.json` 中可精确匹配到 `1667` 条 Hendrycks MATH train 题，且只覆盖 Level 4/5；完整 answer 平均约 `9560` tokenizer tokens，`p95` 约 `23343`，max 约 `31003`，当前不适合直接 SFT。`</think>` 后 summary 平均约 `772` tokens，可作为未来 clean-solution 候选，但该路线暂时搁置。
- 当前主环境中 `unsloth 2025.9.9 + trl 0.26.2` 会在 import 阶段生成非法 `UnslothGRPOTrainer.py`，SFT 小实验短期已切到 `backend=trl_peft`，避免被 Unsloth 的 GRPO patch 兼容问题阻塞；后续若继续使用 Unsloth，应单独整理兼容环境。
- 当前评测结果维护已统一到 `scripts/report_eval_results.py`：自动扫描 `outputs/eval/**/result.json` 并全量写入 registry；`docs/analysis/eval_metadata.yaml` 只作为人工展示白名单，维护重要模型、关键 baseline、效果好的 checkpoint 与明确归档的坏例。
- 当前 MLflow 已接入 SFT、GRPO、eval、Lightning-OPD 数据准备与 MATH rollout；只记录参数、指标、小报告和路径引用，不复制 checkpoint、raw rollout、teacher logits 等大文件。
- 一次性小样本 Stage1 YAML 已归档到 `configs/archive/stage1_oneoff_20260503/`；`configs/stage1/` 只保留稳定数据源或仍可复用的训练入口，后续不再为每个样本数/临时对照新建顶层 YAML。
- 当前主结果口径以 `docs/analysis/eval_leaderboard.md` 为准；`math500_dev200` 这类 dev-only 结果只进入 Task Details，不进入 Main Results。
- 当前已记录的主线结果：`outputs/grpo_3090_dapo_server/checkpoint-660` 在 MATH500 为 `0.7240`，`outputs/stage1_mix_long_sft/checkpoint-300` 在 MATH500 为 `0.7020`，`outputs/lightning_opd_nemotron_from_stage1_ckpt300_ep1` 在 MATH500 为 `0.6900`。
- 当前 Lightning OPD gap 诊断见 `docs/analysis/lightning_opd_gap_diagnostics.md`：Nemotron 和 SFT student 的 top16 overlap 平均为 `9.84/16`，但 `teacher_logprob < student_logprob` 的 token 占 `60.67%`，说明 topK support 有重叠但 sampled-token OPD 信号整体偏负；继续放大 topK=1 OPD 风险较高。
- 当前 verl 标准 OPD 数据准备已落地到 `scripts/prepare_verl_opd_data.py` 与 `src/post_train/verl_opd/`：smoke 数据为 `../verl/data/opd_dapo17k/smoke128/train.parquet`，共 128 条纯 DAPO；正式 1K 混合数据为 `../verl/data/opd_mix/candidate1_dapo4_1k/train.parquet`，共 1000 条，其中 `extra_info.source=opd_candidate` 为 200 条、`extra_info.source=opd_dapo` 为 800 条。`data_source` 统一写为 `math_dapo`，避免 verl 默认 reward manager 报未知 source；candidate 默认源为 `data/on_policy_loop/query_strategy/candidate_pool.jsonl`。
- 当前 verl 标准 OPD 启动脚本为 `../verl/examples/on_policy_distillation_trainer/run_qwen25_math_opd.sh`，默认使用 GPU 可见集合内的 `1` 张学生卡与 `1` 张教师卡；若要用物理 GPU 2/3，入口是 `CUDA_VISIBLE_DEVICES=2,3 bash ...`。默认 OPD 口径是 `loss_mode=k1`、`use_policy_gradient=True`、`use_task_rewards=False`，即 student rollout + teacher logprob，不是 DAPO 算法。
- 当前 verl 标准 OPD 的主要吞吐瓶颈应按 student rollout 优先处理；teacher 只对 student sampled tokens 计算 logprob，不应默认占用多数 GPU。4090 无 NVLink 时，多卡 student 会引入 FSDP/权重同步通信，必须用 smoke timing 比较 `timing_s/gen` 与 `timing_s/update_actor` 后再扩大 `STUDENT_WORLD_SIZE`。当前两卡 OPD 资源建议见 `docs/analysis/verl_opd_runtime_diagnostics.md`：先采用 `actor + student vLLM` 同卡、`teacher vLLM` 独占另一张卡，`STUDENT_GPU_MEMORY_UTILIZATION=0.55` 起步，`TEACHER_GPU_MEMORY_UTILIZATION=0.70` 起步，actor/student/teacher 均先用 BF16；`teacher=0.85` 已在 teacher `prompt_logprobs` 的 full-vocab `log_softmax` 阶段 OOM。
- 当前教师 SFT rollout 已从 Lightning-OPD 中拆出，入口为 `scripts/rollout_teacher_sft.py`，核心逻辑为 `src/post_train/rollout/teacher_sft.py`。该入口只生成 raw SFT rollout，不生成 `teacher_topk`；默认输出目录为 `data/rollout/teacher_sft/...`，支持 `prompts -> shard -> merge` 与 `launch` tmux 后台分片。
- Qwen3 Non-Thinking 教师 rollout 默认必须使用 Qwen3 chat template，并显式设置 `chat_template_enable_thinking=false`；此时空 `<think>\n\n</think>\n\n` 位于 rendered prompt 中，不应期待 generated response 再包含 `<think>`。
- 当前 ConPress 全量压缩 rollout 已完成两轮：首轮 7497 题重判后正确 `5966/7497=79.58%`；对剩余 1531 题做 ConPress `N=3, spp=4` 失败池重采样后，新增任一正确 694 题，合并预计覆盖 `6660/7497=88.84%`，仍失败 837 题。剩余失败按 level 为：L1 29/564、L2 70/1348、L3 105/1592、L4 131/1690、L5 502/2303；失败高度集中在 Level 5。
- 剩余 837 题已改用单题正常 eval prompt、Qwen3 chat template、`max_new_tokens=8192`、每题 1 条采样试水：平均 `5777.7` tokens，p50 `7869`，约 `49%` 打满长度上限，boxed `57.11%`，parse `56.87%`，correct `14.22%`。该 hard-tail batch 不适合作为 raw SFT/ASFT/OPD 数据，只能作为失败分析或严格过滤后的候选池。
- 当前 ConPress Qwen3-1.7B no-think SFT/DFT 已完成重训与并行选 ckpt，诊断见 `docs/analysis/conpress_qwen3_sft_dft_diagnostics.md`。SFT best `checkpoint-175` 在 MATH500/GSM8K 为 `0.652 / 0.7741`，平均输出 `327.7 / 108.8` tokens；DFT best `checkpoint-50` 为 `0.632 / 0.7665`，平均输出 `239.5 / 91.3` tokens。结论是 ConPress SFT 可用，但 ConPress DFT 未复现 UWLS 上 DFT 优于 SFT 的收益。
- 当前 ConPress 成功压缩训练集 `data/stage1/conpress_qwen3_4b_nt_correct_compressed/train.jsonl` 使用 Qwen3 chat/no-think tokenization 后，`prompt + solution` 共 6660 条，p50 `369`、p90 `1665`、p95 `2455`、p99 `4182`、max `6313`；超过 `2048` 的样本为 `487/6660=7.31%`，超过 `4096` 的样本为 `71/6660=1.07%`。
- 当前 ConPress ASFT-topK QLoRA 已完成，配置为 `configs/stage1/conpress_qwen3_1p7b_asft_topk.yaml`。四卡 `batch_size=2` 曾 OOM，最终使用 `batch_size=1`、`gradient_accumulation_steps=8`，四卡有效 batch 为 `32`。并行 dev200 选择 best 为 `checkpoint-175`：dev200 `0.645`，boxed `0.830`，avg tokens `368.4`；benchmark MATH500/GSM8K 为 `0.654 / 0.8006`，boxed `0.850 / 0.9848`，平均输出 `333.8 / 109.7` tokens。结论是 ASFT 相比 SFT 在 GSM8K 明显提升，MATH500 基本持平，且没有 DFT 退化。
- 当前 ConPress ASFT -> Qwen3-4B teacher 的不过滤 Lightning-OPD 数据已完成：配置 `configs/lightning_opd/conpress_asft_qwen3_4b_teacher_math2000.yaml`，输出目录 `data/lightning_opd/conpress_asft_qwen3_4b_teacher_math2000_20260504`。任务使用 2000 条 Hendrycks MATH train query、4 shard、GPU `0,1,2,4`；student 是 `outputs/stage1_conpress_qwen3_1p7b_asft_topk/checkpoint-175`，teacher 是 `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`。四个 shard 的 `raw_rollouts.jsonl` 与 sampled-token teacher forward 均已完成，每个 `500` 条；`train.jsonl` 共 `2000` 行，`shape_errors=0`，`top_k_values=[0]`。teacher scoring 使用 BF16 HF forward、`teacher_load_in_4bit=false`、`top_k=0`，只保存 sampled token logprob；实现已改为 `target_logit - logsumexp(logits)`，不再 materialize 完整 `[B, L, V]` log-softmax。
- 当前不过滤 Lightning-OPD 已完成训练、四卡并行 dev best 选择和 benchmark：训练配置 `configs/lightning_opd/train_conpress_asft_qwen3_4b_teacher_math2000.yaml`，输出目录 `outputs/lightning_opd_conpress_asft_qwen3_4b_teacher_math2000_unfiltered`，训练起点为 ConPress ASFT `checkpoint-175`，四卡 `0,1,2,4`，共 `63` optimizer steps。dev200 best 为 `checkpoint-30`：acc `0.630`，boxed `0.785`，avg tokens `459.3`；benchmark MATH500/GSM8K 为 `0.678 / 0.8135`，boxed `0.824 / 0.9750`，平均输出 `394.2 / 138.5` tokens。相比 ConPress ASFT `checkpoint-175` 的 `0.654 / 0.8006`，不过滤 OPD 准确率有正收益，但 MATH500 boxed rate 下降、输出变长。
- Qwen3 student rollout 做 OPD 数据准备时必须使用训练一致 prompt：`justrl_math` + Qwen3 chat template + `assistant_prefill="<think>\n\n</think>\n\n"`。本轮已生成显式渲染后的 query source：`data/lightning_opd/query_sources/conpress_asft_qwen3_chat_math2000_seed42.jsonl`，避免默认 Lightning-OPD Qwen2.5 裸 prompt 污染格式。
- Lightning-OPD vLLM rollout 对 LoRA adapter 已补齐自动读取 `adapter_config.json` 中的 rank，并传入 `max_lora_rank`；本轮 ASFT LoRA rank 为 `32`，否则 vLLM 默认 `16` 会报 `LoRA rank 32 is greater than max_lora_rank 16`。student vLLM 生成后也会显式 shutdown 并释放 CUDA cache，再加载 teacher 做 sampled-token logprob，降低同卡 OOM 风险。
- 若用 Qwen3-4B Non-Thinking 作为标准 OPD teacher，主要风险不是 tokenizer/template 不一致，而是 teacher 在 hard-tail 上严重长输出和低正确率，可能惩罚 ConPress student 的短解风格。本轮不过滤 OPD 的最终结果说明 sampled-token 信号不是完全无效：MATH500/GSM8K 均高于 ConPress ASFT 起点；但训练日志中 teacher logprob 多数低于 student logprob，且 MATH500 boxed rate 下降，后续仍应优先做格式过滤或 topK support，而不是直接扩大 epoch。

## 当前最重要的问题

1. `GRPO / DAPO-lite` 目前尚未在 `math500_dev200` 上稳定超过 `stage1_mix_long_sft/checkpoint-300`。
2. `math500_dev200` 只有 `200` 题，单次评测标准误差约在 `0.034` 档，当前小幅波动不足以支持强结论。
3. `select_sft_checkpoint.py` 当前单进程串行评测 checkpoint；虽然可复用单个 `vLLM` 底模实例切 LoRA，但多卡空闲时仍应优先用 parallel wrapper 分片筛选。
4. `GRPO` 早期训练仍存在格式习惯容易被冲坏的风险，需要继续观察：
   - `boxed_rate`
   - `parse_success_rate`
   - `completions/clipped_ratio`
   - `entropy`
5. JustRL 采样轨迹用于 SFT 前必须加强长度与完整性过滤；当前正式答案抽取会读取最后一个 `\boxed{...}`，但多 boxed 与截断仍不能单独证明生成轨迹完整。
6. 当前 MATH rollout 入口若未走 chat template，会系统性污染 `<think>`、summary 与 boxed 格式统计；后续 JustRL / Nemotron / Qwen chat 模型 probe 必须先确认 prompt 已套 chat template。
7. Lightning OPD 当前 topK=1 sampled-token 信号偏负，下一步若继续 OPD，应优先实现或测试 topK support loss，而不是直接增加 epoch。
8. verl OPD 当前只完成数据与启动脚本准备；截断样本整条 mask 仍是正式长跑前建议补丁。
9. verl OPD 弹性卡数调度不能简单把多数 GPU 分给 teacher；7/4/3 卡场景下应优先比较 `student rollout` 吞吐与 FSDP 通信开销。若需要严格 resume optimizer/FSDP 状态，`STUDENT_WORLD_SIZE` 必须在同一实验中保持不变；若允许只从 HF/LoRA 权重重启，则可改变 student 卡数但不再是完整训练状态 resume。
10. ConPress 剩余 hard-tail 单题正常 prompt 试水已经证明 raw rollout 质量很差，后续不能把这批长输出直接混入训练；若继续挖 hard-tail，只能多 attempt 后严格抽取 `boxed && parse_ok && correct` 且长度合理的轨迹。
11. ConPress Qwen3-1.7B no-think SFT/DFT/ASFT 已完成一轮同口径对照；当前结论是 SFT 可用、DFT 相对收益不足、ASFT 在 GSM8K 上有收益但 MATH500 只持平。
12. Qwen3-4B-2507 Thinking 完整思维链过长，暂不进入当前主线；如后续恢复，只考虑 summary-only 数据。
13. 教师 SFT rollout 长任务后续默认用 `tmux` 后台分 shard 运行，并写入 `logs/` 与 `jobs.json`；避免直接在当前终端前台阻塞。
14. 旧路线文档仍在仓库中保留，必须继续和当前 `GRPO` 主线区分，不自动视为当前推荐配置。
15. Qwen3-1.7B 全量 clean<2048 的 DFT 在 dev200 上 `ckpt25` 最好，后续 ckpt 整体变弱；这支持“DFT 易分布漂移”的判断。当前已实现轻量 `asft_topk` anchor，下一步应先做 smoke 与同口径 checkpoint 选择，而不是继续加 DFT epoch。
16. 标准 OPD 如果使用 Qwen3-4B Non-Thinking teacher 与 ConPress ASFT student，必须关注 teacher/student sampled-token gap；不能假设更大 teacher 自动提供正向训练信号。当前不过滤 2000 条 Lightning-OPD 最优 `checkpoint-30` 在 MATH500/GSM8K 达到 `0.678 / 0.8135`，说明 OPD 有收益；但 MATH500 boxed rate 低于 ASFT，下一步不应只扩大同一数据，而应先处理格式/重复污染或测试 topK support。

## 当前优先级

1. 把 `GRPO / DAPO-lite` 的 checkpoint 选择与 benchmark 评测跑顺，先确认单 seed 下是否至少不差于当前 SFT 基线。
2. 优化 checkpoint 选择链路，避免每个 checkpoint 反复重启 `vLLM`。
3. 继续盯 `boxed_rate / parse_success_rate / clipped_ratio / entropy`，确认当前 reward 与长度设置不会在早期破坏格式输出。
4. 只有当单 seed 结果达到“至少不差于基线”时，再考虑做 `3 seed` 复验，而不是现在就对每组配置做多 seed 全覆盖。
5. ConPress SFT/DFT/ASFT 暂停继续扩展，先用 ASFT best ckpt 做 OPD 数据诊断；如果后续再训练 ASFT，优先考虑 `<=2048` 长度过滤数据降低显存和墙钟成本。
6. 当前不过滤 Lightning-OPD 已完成并取得小幅正收益；短期优先复核输出污染、boxed 下降和 sampled-token gap，再决定是否做过滤版、topK support 或切换 teacher。

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
- 文档入口看 `docs/README.md`
- 配置入口看 `configs/README.md`
- 运行脚本入口看 `runs/README.md`
- 当前阶段设计看 `docs/SPEC.md`
- 结构分层看 `docs/ARCHITECTURE.md`
- 当前状态看 `docs/STATE.md`
- 评测结果总表看 `docs/analysis/eval_leaderboard.md`
- 评测结果维护命令看 `docs/runbooks/eval.md`
- SFT 运行说明看 `docs/runbooks/sft.md`
- MLflow 使用说明看 `docs/runbooks/mlflow.md`
- `GRPO` 运行与操作说明看 `docs/runbooks/grpo.md`
- 旧路线实验记录看 `docs/experiments/README.md` 与 `docs/experiments/two-stage-sft-simpo.md`

## 2026-05-04 增量状态

### 94 服务器迁移新增
- 已迁入：
  - `outputs/stage1_mix_long_justrl_chat_lt2048_random2000_dft/checkpoint-63`
  - `data/lightning_opd/nemotron_candidate`
  - `data/rollout`

### 94 服务器下载状态
- 已启动下载：
  - `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`
  - `Qwen/Qwen3-1.7B`
- 如需确认是否完成，以 `~/data/post_train/tmp/*.log` 和 `~/data/hf_cache/hub/` 为准。

### 代理守护
- 当前 `90/94` 的反向代理由本地 `scripts/watch_proxy_tunnels.py` 守护。
- 如需项目内启用代理，仍使用：
  - `source .runtime/env.sh`
  - `enable_local_proxy 17897`

### 90 服务器缓存新增
- 已下载模型：
  - `nvidia/OpenMath-Nemotron-1.5B`
  - `Qwen/Qwen3-1.7B`
  - `Qwen/Qwen3-4B-Thinking-2507`
  - `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`
  - `hbx/JustRL-Nemotron-1.5B`
- 已下载数据集：
  - `UWNSL/Mix-Long_long_0.2_short_0.8`
  - `open-r1/DAPO-Math-17k-Processed`
- 项目数据新增：
  - `data/rsr/qwen3_4b_2507/short-sys_5k_gen1.json`

### 代理守护
- 本地现在使用 `scripts/watch_proxy_tunnels.py` 守护 `90/94` 的 `ssh -R` 隧道。
- 健康检查以远端真实 `curl https://www.google.com` 为准，不再只看端口或进程。
