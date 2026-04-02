# SPEC

## 目标

当前阶段的目标是把项目主线切换为 `on-policy SFT`，并围绕这条主线建立最小可运行闭环。

本阶段关注点：

1. 保持 `stage1 SFT` 作为稳定起点
2. 设计并验证 `on-policy` 数据生成与筛选链路
3. 用统一评测链路比较 `stage1` 与 `on-policy SFT` 的结果
4. 判断 `on-policy SFT` 是否比已有离线增量路线更值得继续投入

本阶段不是为了同时推进所有候选路线，而是为了给当前主线提供清晰、可执行、可比较的实验口径。

## 当前阶段定义

### Stage1 SFT

- 继续作为当前主线的基础模型来源。
- 当前数据口径：`UWNSL/MATH_training_split_short_cot`
- 当前首版训练规模：`2000`
- `dev`：从同源数据中切出 `200` 条并冻结
- 当前 checkpoint 选择口径：`2 epoch`、`save_steps=25`、训练结束后统一在 `dev200` 上按 `normalized_accuracy` 选 best checkpoint

### on-policy SFT

- 定义为“从当前模型出发生成新样本，再经过筛选后继续做 SFT”。
- 当前首版口径：
  - seed 模型固定为 `outputs/stage1_sft_5000/checkpoint-200`
  - query registry 固定为 `data/stage1/train_5000.jsonl`
  - 当前语义是“seed 模型已见 5000 题上的 self-refine / instability mining”，不是未见题 pure on-policy
  - 当前不使用 `OpenR1-Math-220k-Cleaned` 或其他 `math220k` 数据做 on-policy query 或 retained 训练数据
  - 默认排除 `stage1_dev200_5000` 与各类 eval/dev 集
  - 生成 prompt 复用当前 `SFT prompt`
  - 单轮数据准备默认对整个 query registry 做 full-harvest
  - 自动循环时每轮默认抽 `256` 题、每题采样 `4` 条
  - 样本筛选统一使用本地 parser + `math-verify`
  - 每轮必须保留全量轨迹归档
  - 默认同时导出两套 retained：
    - `any_correct_shortest`
    - `mixed_only_shortest`
  - 两套 retained 都要求：`parse_success=True`、`is_correct=True`、`completion_len<=512`
  - 每题最多保留 `1` 条，若多条满足则保留最短正确回答
  - 继续复用现有 `train_sft.py` / `eval_model.py`
  - 支持自动循环脚本：每轮直接使用该轮最终模型进入下一轮，轮间用 `math500_dev200` 上的 `normalized_accuracy` 做早停
  - 自动循环的 query 采样按 epoch 洗牌消费：先无放回扫完整个 query 池，耗尽后重洗继续
  - 当前默认早停口径：`patience=3`、`min_delta=0.0`
- 当前首版边界：
  - 只实现 `pure_onpolicy`
  - 不实现复杂 prompt 去重或模糊去重
  - 不实现每题保留多条正确轨迹
  - retained 样本过少时直接失败，不做自动补采样

### 暂停路线

- 两阶段 `SFT + SIMPO` 当前不是主线。
- `OpenR1-Math-220k-Cleaned` 当前只出现在暂停中的 stage2 / mixed 路线，不属于 on-policy 主线数据源。
- 相关设计、观察和保留原因迁到：
  - `docs/experiments/two-stage-sft-simpo.md`

## 当前模型与评测口径

- 开发模型：`Qwen/Qwen2.5-Math-1.5B`
- 正式 benchmark：`GSM8K`、`MATH-500`
- 额外支持 opt-in 的 sampled benchmark 组：`AIME24`、`AIME25`
- 当前默认评测链路：`vllm_raw`
- 当前执行口径默认只使用 `vllm_raw`
- `eval` 结果默认落到 `outputs/eval/<模型路径>/<task>/result.json` 与 `raw.json`
- `AIME24/AIME25` 当前口径：
  - 不进入默认 `configs/eval/default.yaml`
  - 通过单独 `configs/eval/aime.yaml` opt-in
  - 每题默认采样 `4` 次
  - 主指标为严格 `pass@1 = mean(correct_count / 4)`
  - sampled pass@1 当前只支持 `vllm_raw`

当前核心指标：

- `boxed_rate`
- `parse_success_rate`
- `normalized_accuracy`
- `pass_at_1`
- `avg_output_tokens`
- `all_correct / mixed / all_wrong`
- `correct_k_of_4`

## Prompt 原则

- SFT 使用更强约束的 prompt，优先学稳格式。
- 评测使用更短、接近公开基线的 prompt。
- 统一要求答案中必须包含 `\boxed{...}`。

## 当前边界

本阶段当前不做：

- 同时把 `on-policy SFT` 与两阶段 `SFT + SIMPO` 都当作现行主线
- 更复杂题目选择策略
- 超出 `any_correct_shortest` / `mixed_only_shortest` 的复杂轨迹筛选策略
- `mixed_onpolicy`
- 为未来阶段预先设计复杂抽象

## 参数口径

- 当前推荐 prompt、LoRA、训练与评测超参会在配置中维护。
- `configs/<workflow>/*.yaml` 是可执行参数真源。
- `SPEC.md` 只保留当前主线推荐口径与原则，不逐项同步所有细参数。
