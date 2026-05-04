# 架构说明

## 目标

当前仓库当前主线实验重心转向 `GRPO / DAPO-lite`，同时保留旧的 `on-policy SFT`、两阶段 `SFT + SIMPO` 代码与文档作为历史路线或候选路线。

核心设计目标：

1. `scripts/` 只做编排
2. 训练、评测、数据处理逻辑全部下沉到 `src` 包
3. 数据格式与评测口径统一
4. `GRPO` 的训练、checkpoint 选择与 benchmark 评测形成稳定闭环
5. 关键链路必须能显式证明“当前到底在用 base 还是 base + adapter”

## 分层

### scripts 层

职责：

- 参数解析
- 调用库层 API
- 打印摘要日志
- 结果落盘
- 按稳定工作流提供少量专名入口
- 实验 workflow 入口只做单轮编排，不直接实现训练或评测细节

不负责：

- 数据清洗规则实现
- prompt 构造
- 训练细节实现
- 评测判定逻辑
- 长驻 server 生命周期管理细节

### src 包层

建议模块职责：

- `answers.py`
  - `boxed` 提取
  - 宽松答案提取
  - `math-verify` 等价判定
- `io.py`
  - JSONL 读写
- `prompts.py`
  - 训练 prompt
  - 评测 prompt
- `config/`
  - 按工作流拆分配置 schema 与 loader
  - 对外统一导出 `load_*_config(...)`
- `eval/`
  - `EvalRunner` 负责评测主编排与多任务执行
  - `tasks.py` 负责任务选择与临时 dataset 任务名推导
  - `model.py` 负责 base / LoRA adapter 解析与 `model_resolution`
  - `vllm.py` 只负责 vLLM backend runner 与 raw generation
  - `metrics.py` 负责 raw 结果到 metrics / predictions 的转换
  - `reporting.py` 负责扫描评测产物、合并历史手动记录、应用 metadata，并生成 leaderboard / registry
  - 输出按模型路径与任务名分目录落盘
  - 当前已补齐 `model_resolution` 诊断，显式记录：
    - `pretrained`
    - `enable_lora`
    - `lora_local_path`
  - `service.py` 保留旧 `run_eval_task(...)` facade，供历史调用方兼容
- `sft_selection.py`
  - checkpoint 扫描
  - dev 批量评测
  - best checkpoint 选择
  - 当前会把 `model_resolution` 一起写入 ranking / best summary
- `grpo/`
  - `train.py` 封装 TRL `GRPOTrainer` 主流程
  - `model_loading.py` 负责 PEFT 加载、量化参数与 dtype 对齐
  - `training_args.py` 负责 `GRPOConfig`
  - `diagnostics.py` 负责 adapter / dtype 诊断
  - `callbacks.py` 负责 W&B callback
  - `server.py` 负责双卡 `vLLM server + trainer` 编排
  - `data.py` 与 `rewards.py` 分别负责 GRPO 数据与 reward
  - 当前已补齐 `grpo.adapter.*` 诊断，显式记录：
    - 是否从 adapter checkpoint 继续训练
    - LoRA 参数数量
    - adapter 对应底模路径
  - 当前已支持：
    - 优先 `6/7`
    - 空余显存阈值判空
    - 已有 server 探活与复用
    - metadata 记录与安全清理
  - 旧的顶层 `grpo_*` 兼容导出已删除，项目内入口直接依赖 `post_train.grpo.*`
- `datasets/`
  - 按数据准备工作流收口 `stage1_math220k`、`stage2`、`stage2_hendrycks_long`、`stage2_mix_long`、`math220k_dev`、`aime`、`simpo`
  - 旧的顶层 `*_data.py` / `aime_eval.py` / `math220k_dev.py` 兼容导出已删除
- `on_policy/`
  - on-policy loop、query strategy、数据构造与 trainset 构造统一收口到子包
  - 旧的顶层 `on_policy_*` 兼容导出已删除，项目内入口直接依赖 `post_train.on_policy.*`
- `rollout/`
  - `common.py` 收口 shard 切分、长度统计与 chat template 渲染等通用逻辑
  - `teacher_sft.py` 只负责教师模型生成 SFT raw rollout，输出 `prompts.jsonl`、shard raw、merge raw 与统计 report
  - Lightning-OPD 不再承载教师 SFT rollout 生成；它只复用 raw rollout 做 teacher forward / topK logprob 与 OPD 数据构造
- `workflow.py`
  - 单轮实验编排层
  - 训练 -> dev 选 best checkpoint -> benchmark -> baseline 对比
  - workflow summary / failure summary 落盘
- `on_policy/`、`simpo.py`、`datasets/simpo.py`、`datasets/stage2_*`
  - 保留为旧路线或候选路线模块，不是当前主线
- `verl_opd/`
  - 负责把本项目 candidate 池与本地 DAPO-17K parquet 转成 verl RLHF parquet schema
  - `scripts/prepare_verl_opd_data.py` 只做 CLI 编排
  - 当前支持纯 DAPO smoke 与 candidate:DAPO = 1:4 的 1K 混合数据
  - OPD 训练本体在兄弟仓库 `../verl` 中运行
  - 当前不在本仓库中实现 verl 训练调度；本仓库只维护数据准备、启动口径与结果记录

## 主链路

### 1. Stage1 SFT 基线

输入：

- 原始数学数据集

输出：

- `stage1` 训练 JSONL
- 冻结 `dev` JSONL
- `stage1` 模型目录
- `stage1` 评测结果

说明：

- 当前最重要的对照基线是 `stage1_mix_long_sft/checkpoint-300`
- 这条线当前主要提供：
  - seed adapter
  - 对照评测分数
- `backend=trl_peft` 可通过 `runs/sft/train_sft_ddp.sh` 使用 `torchrun` 做多卡 DP/DDP；`scripts/train_sft.py` 仍只负责单次训练入口与主进程结果登记。

### 2. GRPO / DAPO-lite（当前主线）

输入：

- `rd211` 过滤后的题目
- 本地 anchor 题池
- `stage1` base / adapter 模型

输出：

- `GRPO` 训练 JSONL
- `GRPO` 模型目录
- `wandb` 训练指标
- dev checkpoint 排名
- benchmark 结果与对比摘要

流程：

rd211 + anchor
-> solve-rate 过滤
-> exact dedup
-> 按比例混合
-> reward 组装
-> 双卡 `vllm server + trainer`
-> `TRL GRPOTrainer`
-> checkpoint 评测
-> benchmark 对比

说明：

- 当前默认路线是双卡 `server mode`
- 当前默认不是完整 DAPO 论文复现，而是工程可跑的 `DAPO-lite`
- 当前关键观测项包括：
  - `boxed_rate`
  - `parse_success_rate`
  - `reward`
  - `reward_std`
  - `completions/clipped_ratio`
  - `entropy`

### 3. 单轮实验 Workflow

输入：

- 一份训练配置
- 一份评测配置
- 一组 benchmark 任务

输出：

- 训练输出目录
- best checkpoint
- benchmark 结果
- baseline 对比摘要

说明：

- `scripts/run_experiment_workflow.py` 当前提供一轮实验闭环
- 当前 v1 重点服务 `GRPO`
- 训练后编排与训练实现本身解耦

### 4. 评测结果维护

输入：

- `outputs/eval/**/result.json`
- `docs/analysis/eval_metadata.yaml` 中人工选中的模型
- 历史手动结果文件，例如 `tmp/results.md` 与 `tmp/new_results.md`

输出：

- `docs/analysis/eval_registry.jsonl`
- `docs/analysis/eval_leaderboard.md`

流程：

result.json
-> 历史手动结果解析
-> 同模型同任务去重
-> metadata 白名单标注与过滤
-> 生成机器索引与主表

说明：

- `scripts/report_eval_results.py` 只负责 CLI 编排
- 核心合并和渲染逻辑在 `post_train.eval.reporting`
- 正式 `outputs/eval/**/result.json` 优先于历史手动记录
- registry 保留扫描到的全量结果
- leaderboard 只展示 metadata 显式跟踪的模型
- dev-only 结果保留在 Task Details，不进入 Main Results

### 5. MLflow 实验索引

输入：

- 训练入口的配置、参数与 Trainer 日志
- 数据准备入口的 report 与输出路径
- 评测入口的 `result.json` 与指标

输出：

- `mlruns/` 本地 MLflow store
- `outputs/<run>/mlflow_run.json`

说明：

- MLflow 只保存参数、指标、小型 JSON artifact 和大文件路径
- checkpoint、raw eval、rollout JSONL 与 teacher logits 仍以 `outputs/`、`data/` 为真实存储
- 评测时若模型目录存在 `mlflow_run.json`，指标写回同一个训练 run
- 详细操作看 `docs/runbooks/mlflow.md`

### 6. verl OPD（候选路线）

输入：

- 本项目准备的 verl parquet 数据
- `outputs/stage1_mix_long_sft/checkpoint-300` adapter
- student base 与 teacher 模型

输出：

- verl 训练 checkpoint
- OPD 训练指标
- 后续通过本项目 `scripts/eval_model.py` 统一评测的结果

流程：

parquet prompt batch
-> student vLLM rollout
-> teacher vLLM 计算 teacher logprob
-> replay buffer
-> sleep student rollout
-> actor/FSDP 前向与更新
-> actor weights/adapter 同步回 student vLLM

说明：

- 当前标准 verl OPD 使用 `actor + student rollout` hybrid 资源池，teacher 由单独 `teacher_pool` 常驻管理。
- `actor` 与 `student vLLM` 语义上是同一个 base model，但显存中不是同一份 CUDA tensor；actor 用 HF/FSDP 训练权重，student vLLM 用推理引擎内部权重与 KV cache，二者通过 `update_weights` 同步。
- LoRA 训练只减少可训练参数与同步负担，不代表 actor 只占 adapter 显存；base 权重仍参与 actor 前向/反向，也在 student vLLM 中有推理副本。
- 当前两卡推荐调度是 `actor + student vLLM` 同卡、`teacher vLLM` 单独一张卡。teacher 可较高 utilization，student 需要给 actor/FSDP、weight sync 与临时 buffer 留余量。
- vLLM sleep 只能在对应阶段释放显存，不能降低 vLLM 初始化时的 `gpu_memory_utilization` 预算要求；把 student vLLM 与 teacher vLLM 同卡需要改 OPD 数据流，把 teacher scoring 延后到 batch 级并显式 sleep/wake。
- 当前 OPD baseline 统一使用 BF16；vLLM 量化和 FSDP 4bit/QLoRA 暂不作为默认架构假设。

### 7. on-policy SFT（历史路线）

输入：

- `stage1` 基础模型或 checkpoint
- on-policy 生成出的候选数据

输出：

- on-policy SFT 数据
- 新的模型目录
- 评测结果

说明：

- 该路线当前保留，但不再是主线
- 文档仍保留，供后续对照或回退时参考

### 8. 两阶段 SFT / SIMPO（候选路线）

输入：

- stage2 SFT JSONL / preference JSONL
- 基础模型或上游阶段模型

输出：

- `stage2` / `SIMPO` 模型目录
- 相应评测结果

说明：

- 当前属于暂停中的候选路线
- 不自动视为当前推荐实现

## 关键约束

- 训练与评测链路必须能显式证明当前在评：
  - 纯 base
  - 还是 `base + adapter`
- checkpoint 选择不能使用 benchmark test 集反向挑模型
- 若评测长度设置导致大量截断，必须优先修正评测口径，再比较模型优劣
