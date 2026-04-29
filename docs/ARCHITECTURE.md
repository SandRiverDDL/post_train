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
  - 顶层 `grpo_server.py`、`grpo_data.py`、`grpo_rewards.py`、`grpo_runtime.py` 只保留兼容导出
- `datasets/`
  - 按数据准备工作流收口 `stage1_math220k`、`stage2`、`stage2_hendrycks_long`、`stage2_mix_long`、`math220k_dev`、`aime`、`simpo`
  - 顶层旧 `*_data.py` / `aime_eval.py` / `math220k_dev.py` 只保留兼容导出
- `on_policy/`
  - on-policy loop、query strategy、数据构造与 trainset 构造统一收口到子包
  - 顶层 `on_policy_data.py`、`on_policy_query_strategy.py`、`on_policy_trainset.py` 只保留兼容导出
- `workflow.py`
  - 单轮实验编排层
  - 训练 -> dev 选 best checkpoint -> benchmark -> baseline 对比
  - workflow summary / failure summary 落盘
- `on_policy/`、`simpo.py`、`datasets/simpo.py`、`datasets/stage2_*`
  - 保留为旧路线或候选路线模块，不是当前主线

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

### 4. on-policy SFT（历史路线）

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

### 5. 两阶段 SFT / SIMPO（候选路线）

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
