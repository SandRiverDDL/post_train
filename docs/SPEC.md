# SPEC

## 目标

当前阶段的目标是围绕 `GRPO / DAPO-lite` 建立一条稳定、可比较、可复用的实验闭环，并用统一评测口径判断它是否真正优于当前最强的 `stage1 SFT` 基线。

本阶段关注点：

1. 保持 `stage1_mix_long_sft/checkpoint-300` 作为当前对照基线
2. 用 `GRPO / DAPO-lite` 在同一底模、同一评测口径下做可比较实验
3. 通过 `math500_dev200 -> math500_test + gsm8k_test` 的顺序做 checkpoint 选择与最终验证
4. 在保证显存可行的前提下，优先解决：
   - 长度截断
   - 格式退化
   - checkpoint 评测过慢

本阶段不是为了同时推进所有候选路线，而是为了先回答一个明确问题：

`GRPO / DAPO-lite` 在当前 1.5B 数学模型上，是否能稳定带来超过当前 SFT 基线的收益。

## 当前阶段定义

### 当前对照基线

- 对照模型：`outputs/stage1_mix_long_sft/checkpoint-300`
- 当前已知 dev 口径：`math500_dev200`
- 当前实际比较时必须保证：
  - 相同 prompt 口径
  - 相同 `max_new_tokens`
  - 相同 parser 与 `math-verify` 口径
- 若评测长度上限过小导致大量截断，不允许据此直接下基线优劣结论

### GRPO / DAPO-lite

- 当前默认起点：`outputs/stage1_mix_long_sft/checkpoint-300`
- 当前训练数据：`data/grpo/train.jsonl`
- 当前默认路线：双卡 `server mode`
  - trainer 与 `vLLM server` 分卡运行
  - 不再使用 `colocate` 作为默认主路线
- 当前默认核心配置：
  - `loss_type=dapo`
  - `epsilon=0.2`
  - `epsilon_high=0.28`
  - `beta=0.0`
  - `mask_truncated_completions=true`
  - `max_completion_length=768`
  - `soft_overlong.weight=0.05`
  - `soft_overlong.cache_tokens=192`
- 当前优先目标不是完整复现 DAPO 论文，而是先做一版工程上可跑、可比、可解释的 `DAPO-lite`

### 当前评测与选择口径

- checkpoint 粗筛：`math500_dev200`
- 最终 benchmark：`math500_test`、`gsm8k_test`
- checkpoint 选择指标：`normalized_accuracy`
- 当前默认不做 test-set 反向挑 checkpoint
- 当前默认不对每组配置直接做 `3 seed` 全覆盖；只有单 seed 结果至少不差于基线，才升级为多 seed 复验

### 当前边界

本阶段当前不做：

- 同时把 `on-policy SFT` 与两阶段 `SFT + SIMPO` 当作现行主线
- 直接把 `math500_test` 当成 checkpoint selection 数据集
- 在 checkpoint 尚未达到基线前，对每组配置做完整多 seed 扫描
- 一次性同时修改过多变量，例如在同一轮里同时改：
  - loss
  - reward 结构
  - 长度上限
  - 多 seed 策略
  - benchmark 口径

## 当前模型与评测口径

- 开发模型：`Qwen/Qwen2.5-Math-1.5B`
- 当前默认评测后端：`vllm_raw`
- 正式 benchmark：`GSM8K`、`MATH-500`
- 当前核心指标：
  - `boxed_rate`
  - `parse_success_rate`
  - `normalized_accuracy`
  - `pass_at_1`
  - `avg_output_tokens`
  - `completions/clipped_ratio`
  - `entropy`
- 当前结果输出口径：`outputs/eval/<模型路径>/<task>/result.json` 与 `raw.json`
- 当前 checkpoint 排名输出口径：`<train_output_dir>/dev_eval/dev_ranking.json` 与 `best_checkpoint.json`

## Prompt 与格式原则

- 统一要求答案中必须包含 `oxed{...}`
- 当前阶段若 `boxed_rate / parse_success_rate` 显著下降，优先视为格式退化，而不是直接下结论说推理能力退化
- 若评测长度设置导致截断比例过高，必须先修正长度口径，再比较模型优劣

## 参数口径

- `configs/<workflow>/*.yaml` 仍是可执行参数真源
- `SPEC.md` 只保留当前推荐实验口径与边界，不逐项同步所有实现细节
