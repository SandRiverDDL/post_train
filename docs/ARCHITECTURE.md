# 架构说明

## 目标

当前仓库只为两阶段 `SFT + SIMPO` 建立最小可运行闭环，不追求一次性覆盖所有增强方向。

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
- `stage2_data.py`
  - source-specific filters
  - generic filters
  - 尾部标准化为单个 canonical boxed
  - token 长度分桶
  - 按配额混采 stage2 数据
- `sft.py`
  - stage1 / stage2 SFT 共用训练逻辑
- `sft_selection.py`
  - checkpoint 扫描
  - dev 批量评测
  - best checkpoint 选择
- `simpo.py`
  - TRL `CPOTrainer(loss_type=simpo)` 封装
  - 4bit + PEFT 加载
  - checkpoint / resume 编排
- `simpo_data.py`
  - query 池构造
  - 多 response 采样
  - pair 构造与报告
  - pilot 子集导出
- `eval.py`
  - `vllm_raw` / `lm_eval` 多 runner 编排
  - 模型输出后处理
  - `math-verify` 判定与指标汇总

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

### 2. Stage2 SFT

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

### 3. SIMPO

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

## 评测设计

正式评测入口统一为 `scripts/eval_model.py`。

职责划分：

1. 生成 runner
   - `vllm_raw` 直接调用 `vLLM`
   - `lm_eval` 通过 `lm-eval-harness` 编排
2. 本地后处理
   - 提取最终答案
   - 调用 `math-verify`
   - 聚合指标
   - 写出可比较结果文件

首版核心指标：

- `boxed_rate`
- `parse_success_rate`
- `normalized_accuracy`
- `avg_output_tokens`

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
