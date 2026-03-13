# Tiny GRPO 调试方案（短响应版）

  ## Summary

  当前 tiny 训练集直接复用 `prepare_grpo_data.py` 从 `GSM8K train(main)`
  重新生成，而不是从完整工件二次抽样。
  主因不是 `response` 不够短，而是 `NuminaMath` 对 `1.7B/4B`
  冷启动 policy 来说 reward 太稀疏。

  整体策略仍然保持最小改动：

  - 不改主训练逻辑
  - 不改 reward
  - 不新增 tiny dev 文件
  - 只新增 tiny 配置和 tiny 训练数据工件
  - dev 验证继续用现有评测入口的 --limit

  ## Key Changes

  - 保留现有训练入口：
      - train_grpo.py
      - grpo.py
  - 新增 tiny 配置：
      - configs/grpo_tiny.yaml
  - 新增 tiny 数据工件：
      - `data/grpo/train_grpo_gsm8k_tiny_short.jsonl`
  - tiny 数据通过 `prepare_grpo_data.py` 直接从 `GSM8K train(main)`
    重新生成
  - 推荐 tiny 数据生成参数：
      - --num-samples 64 或 128
      - --min-response-tokens 1
      - --max-response-tokens 128
  - tiny 训练配置推荐值：
      - train_dataset: data/grpo/train_grpo_gsm8k_tiny_short.jsonl
      - output_dir: outputs/grpo-qwen3-1.7b-tiny
      - max_prompt_length: 256
      - max_completion_length: 128
      - num_generations: 2
      - gradient_accumulation_steps: 4
      - logging_steps: 1
  - 不新增 tiny dev 工件：
      - 继续使用现有评测入口
      - 通过 --limit 50 或 --limit 100 做 SFT vs tiny GRPO 对照

  ## Test Plan

  - 数据准备：
      - 运行 tiny 数据生成命令
      - 确认样本数、response_tokens 和 prompt 预览符合预期
  - 训练验证：
      - 用 configs/grpo_tiny.yaml 跑通至少 30~50 个 step
      - 检查 train_log.jsonl，重点看：
          - reward
          - rewards/correctness_reward/mean
          - rewards/parse_reward/mean
          - rewards/format_reward/mean
          - completions/clipped_ratio
  - 效果验证：
      - 用现有 eval_dataset.py 对同一 checkpoint 做小规模
        benchmark
      - 例如：
          - SFT baseline --limit 50
          - tiny GRPO output --limit 50
  - tiny 阶段验收标准：
      - reward 指标方向正确
      - boxed 协议没有明显崩
      - benchmark 小样本上不要求显著提升，但至少不应明显退化

  ## Assumptions

  - 你的目标是先把训练有效性和速度问题验证清楚，而不是产出正式实
    验结果
  - prepare_grpo_data.py 当前的长度过滤接口足够承担 `gsm8k` tiny 数据生成
  - tiny 最重要的目标是先降低 reward 稀疏度，而不是单纯压短 completion
