# Roadmap

## Phase 1：SFT 基线

目标：

- 用 NuminaMath 建立稳定 SFT 基线
- 在 `GSM8K dev200` 上完成日常对比
- 在 `MATH500 test` 上完成正式评测

当前状态：

- SFT MVP 基本完成
- 当前最佳 1.7B 路线是：
  - `NuminaMath`
  - 分层抽样 `2000`
  - `response token length = 64~256`
- 已确认该路线同时带来：
  - 更稳定的 boxed 协议学习
  - `GSM8K dev200` 相对 base 的明确提升

交付物：

- 数据准备与验收脚本
- SFT 训练脚本
- strict 评测脚本
- 第一版可汇报结果

## Phase 2：GRPO 最小闭环

目标：

- 在 Phase 1 基线之上接入 GRPO
- 使用 verifiable reward 和统一 `Final answer: \boxed{...}` 解析协议
- 证明 RL 闭环可以稳定运行

Phase 2 起点约束：

- 不再把“如何学会 boxed 协议”当成主问题
- 默认使用 Phase 1 的短 response SFT 作为 cold start
- 重点转向：
  - reward 设计
  - rollout 稳定性
  - SFT 与 GRPO 的增益对比

交付物：

- GRPO 配置与训练入口
- reward 解析模块
- 与 SFT 基线的对比结果

## Phase 3：扩展实验

目标：

- 增加训练规模和评测覆盖
- 做必要的 ablation 和失败案例分析

交付物：

- 实验记录
- 最终项目总结
