# 架构说明

## 目标

当前仓库当前主线转向 `on-policy SFT`，同时保留旧的两阶段 `SFT + SIMPO` 代码与文档作为暂停中的候选路线。

核心设计目标：

1. `scripts/` 只做编排
2. 训练、评测、数据处理逻辑全部下沉到 `src` 包
3. 数据格式与评测口径统一
4. 后续可替换数据来源、轨迹选择与偏好构造方式，而不推翻主干结构

## 分层

### scripts 层

职责：

- 参数解析
- 调用库层 API
- 打印摘要日志
- 结果落盘
- 按稳定工作流提供少量专名入口；评测数据准备统一收口到 `prepare_eval_data.py`

不负责：

- 数据清洗规则实现
- prompt 构造
- 训练细节实现
- 评测判定逻辑

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
- `schemas.py`
  - SFT 数据行
  - preference 数据行
  - 评测结果行
- `data.py`
  - 数据映射
  - 数据清洗
  - 数据导出
  - 数据校验
- `on_policy_data.py`
  - on-policy query 池构造
  - 单轮多 response 采样
  - retained SFT 数据导出
  - retained 报告统计
- `on_policy/`
  - `loop_service`：多轮 on-policy 编排主入口
  - `loop_state`：resume / overwrite / history / final summary
  - `loop_round`：单轮 prepare / train / holdout / teacher 推进
  - `loop_paths`：round 路径与 round config 构造
  - `strategy_*`：uniform / mixed / candidate-random 三类 query strategy
- `stage2_data.py`
  - 暂停中的候选路线模块
  - source-specific filters
  - generic filters
  - 尾部标准化为单个 canonical boxed
  - token 长度分桶
  - 按配额混采 stage2 数据
- `math220k_dev.py`
  - 暂停中的候选路线辅助模块
  - 复用 `math220k` 清洗与校验
  - 固定 profile 分桶抽样
  - 导出 stage2 target dev
- `sft.py`
  - 当前主线与候选路线共用 SFT 训练逻辑
- `sft_selection.py`
  - checkpoint 扫描
  - dev 批量评测
  - best checkpoint 选择
- `simpo.py`
  - 暂停中的候选路线模块
  - TRL `CPOTrainer(loss_type=simpo)` 封装
  - 4bit + PEFT 加载
  - checkpoint / resume 编排
- `grpo.py`
  - 独立实验路线模块
  - TRL `GRPOTrainer` 封装
  - preflight 检查、4bit + PEFT 加载
  - 关键 wandb 指标白名单与平滑上报
- `grpo_data.py`
  - `rd211` 数学 RL 数据过滤
  - 与 anchor 题池按比例混合
  - 统一 exact dedup
- `grpo_rewards.py`
  - 正确性 reward
  - parse 失败惩罚
  - reward 组件装配
- `simpo_data.py`
  - 暂停中的候选路线模块
  - query 池构造
  - 多 response 采样
  - pair 构造与报告
  - pilot 子集导出
- `config/`
  - 按工作流拆分配置 schema 与 loader
  - 对外继续统一导出 `load_*_config(...)`
- `eval/`
  - `vllm_raw` 评测编排
  - sampled AIME task 的多采样聚合
  - 模型输出后处理
  - `math-verify` 判定与指标汇总
  - 结果按模型路径与任务名分目录落盘

## 主链路

### 1. Stage1 SFT

输入：

- 原始数学数据集

输出：

- `stage1` 训练 JSONL
- 冻结 `dev` JSONL
- `stage1` 模型目录
- `stage1` 评测结果

流程：

原始数据
-> 字段映射与清洗
-> SFT JSONL
-> `check_sft_data`
-> `train_sft`
-> `eval_model`

说明：

- `prepare_stage1_data.py` 当前只负责生成 `stage1_train` 与冻结 `dev`
- `gsm8k`、`math500`、`math500_dev200`、`AIME`、`math220k_dev` 统一通过 `prepare_eval_data.py` 准备

### 2. on-policy SFT（当前主线）

输入：

- `stage1` 基础模型或 checkpoint
- on-policy 生成出的候选数据

输出：

- on-policy SFT 数据
- 新的模型目录
- 评测结果

说明：

- 当前首版实现为“单轮数据准备 + 复用既有训练与评测入口手动串多轮”
- 当前也支持单脚本自动循环：
  - prepare -> train -> holdout eval
  - 每轮直接用该轮最终模型继续
  - 轮间用 `math500_dev200` 做 early stop
- 当前默认 seed 模型为 `outputs/stage1_sft_5000/checkpoint-200`
- query registry 默认来自 `data/stage1/train_5000.jsonl`
- 当前默认语义是已见 5000 题上的 self-refine，不是未见题 pure on-policy
- query 默认排除 `stage1_dev200_5000` 与 eval/dev
- 当前 on-policy 主线不混入 `OpenR1-Math-220k-Cleaned` / `math220k` 数据
- 自动循环按 epoch 洗牌消费 query 池，避免前几轮反复命中同一小闭集
- 生成使用 `SFT prompt`
- 每题默认采样 `4` 条，并始终保留全量轨迹归档
- 每轮默认同时导出 `any_correct_shortest` 与 `mixed_only_shortest` 两套 retained
- 每轮训练结束后，直接复用该轮输出模型进入下一轮

### 3. 两阶段 SFT（暂停候选路线）

输入：

- 清洗与筛选后的 stage2 SFT JSONL
- `stage1` 模型或 adapter

输出：

- `stage2` 模型目录
- `stage2` 评测结果

说明：

- 当前数据构造流程：
  - source-specific filter
  - generic validation
  - token 长度分桶
  - 多 pool 按配额混采
- 当前主数据源固定为 `OpenR1-Math-220k-Cleaned` 与 `stage1_train`
- 这部分属于暂停中的 stage2 路线，不是当前 on-policy 主线

Stage2 target dev：

Math220K
-> 复用标准化与过滤
-> 固定 profile 分桶抽样
-> eval JSONL

### 4. SIMPO（暂停候选路线）

输入：

- preference JSONL
- 基础模型或上游阶段模型

输出：

- `SIMPO` 模型目录
- `SIMPO` 评测结果

流程：

query pool
-> 多 response 采样
-> pair 构造
-> preference JSONL
-> pilot preference JSONL
-> schema 校验
-> `TRL CPOTrainer(loss_type=simpo)`
-> 模型导出
-> `eval_model`

### 5. GRPO（独立实验路线）

输入：

- `rd211` 过滤后的题目
- 本地 anchor 题池
- stage1 base / adapter 模型

输出：

- `GRPO` 训练 JSONL
- `GRPO` 模型目录
- `wandb` 关键训练指标

流程：

rd211 + anchor
-> solve-rate 过滤
-> exact dedup
-> 按比例混合
-> reward 组装
-> `TRL GRPOTrainer`
-> 模型导出

## 评测设计

正式评测入口统一为 `scripts/eval_model.py`。

职责划分：

1. 生成 runner
   - 当前固定通过本地 `vllm_raw` 链路直接调用 `vLLM`
   - sampled pass@1 也走同一条链路
2. 本地后处理
   - 提取最终答案
   - 先做规范化字符串精确判等，再调用 `math-verify`
   - 聚合单样本 accuracy 或多样本 `pass@1`
   - 写出可比较结果文件，默认落到 `outputs/eval/<模型路径>/<task>/`

首版核心指标：

- `boxed_rate`
- `parse_success_rate`
- `normalized_accuracy`
- `pass_at_1`
- `avg_output_tokens`
- `all_correct / mixed / all_wrong`
- `correct_k_of_4`

## 扩展原则

后续允许增强：

- 更大规模 SFT 数据
- 更复杂题目选择
- 更复杂轨迹筛选
- 新的 stage2 数据生产方式
- 新的 preference 构造方式
- 新的偏好训练方法

但这些增强必须建立在当前最小主干不变的前提下：

- 数据 contract 稳定
- 训练入口稳定
- 评测入口稳定
- `scripts` 仍然只做编排
