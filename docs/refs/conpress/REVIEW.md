# ConPress Review

## 一句话结论

ConPress 最值得吸收的是：不要直接让模型“少想一点”，而是把多个独立题目打包进同一个 prompt，利用上下文压力让模型自然生成更短的 per-question CoT；再把其中正确且可解析的短轨迹拆回单题 SFT 数据。它特别适合本仓库当前“JustRL/Nemotron 轨迹太长、直接 SFT 可学性差”的问题。

注意：公开可靠来源目前是 arXiv 预印本 `2602.01472`，未确认 ICML 2026 正式接收页；本文档按预印本调研记录。

## 核心机制

ConPress 观察到一个 self-compression 现象：同一个 reasoning model 在单题 prompt 下容易长篇推理，但在一个 prompt 中同时回答多个独立可解问题时，会为每道题自发压缩推理链。

关键点：

- 压缩不是来自显式长度惩罚，也不是来自 “be concise” 指令。
- 第二道题即使是 `1+1=?` 这种 toy question，也能明显压缩目标题 CoT。
- 题目数 `N` 越大，压缩越强，但准确率风险也上升。
- 压缩主要减少 exploration、verification、reflection 这类过度推理行为，planning 相对保留。

论文默认把这个现象转成训练数据：

1. 从题库随机抽 `N` 道独立题。
2. 用固定格式打包：

```text
Question 1: ...

Question 2: ...

Question 3: ...
```

3. 用当前模型 rollout 多题回答。
4. 先抽取 `<think>` 内 reasoning trace 与对应 final answers，再以显式题号 `Question i` 作为主锚点切分每题轨迹。
5. 对每个已切分 question block 抽取 boxed answer，并用 verifier 判断正确性。
6. 只保留正确、格式正常的压缩轨迹。
7. 拆回单题 `(prompt, compressed_response)` 做普通 SFT。

这本质是 self-distillation：teacher 不是另一个模型，而是同一个模型在多题上下文压力下的更短行为。

## 实验要点

论文主设置：

- 数据：MATH、AIME 2024 前题目、LIMO，约 `8k` 单题。
- 默认多题数：`N=3`。
- rollout：vLLM，`temperature=0.6`，`top_p=0.95`，`32k` context。
- 每个 multi-question prompt 采样 `8` 次。
- 训练：普通 next-token NLL / SFT，无 RL、无 DPO、无显式长度正则。
- 训练超参：`3 epoch`，`lr=2e-5`，`warmup_ratio=0.05`，batch size `32`。

主结果：

- Qwen3-4B-Thinking：平均 token 降 `48.7%`，平均准确率降 `0.6` 点。
- R1-Distill-Qwen-7B：平均 token 降 `33.9%`，平均准确率基本持平。
- R1-Distill-Qwen-1.5B：平均 token 降 `40.2%`，平均准确率降 `0.2` 点。

消融：

- `N=2` 已有明显压缩。
- `N=3` 是论文推荐默认，压缩和准确率折中最好。
- `N=4/6/8` 继续变短，但收益递减且准确率更不稳定。
- 采样位置不是硬约束；第 1 个位置更短，但各位置都能用。

## 和本仓库问题的关系

当前本仓库遇到的问题：

- JustRL / Nemotron 轨迹长，很多题存在重复验证、重复 boxed、正确后继续输出。
- 直接把长 CoT 当 SFT 数据，可能让 1.5B student 学到冗余和格式噪声。
- 纯 bad words 或 stop 很难区分有效验证和无效废话。

ConPress 给出的思路更干净：

- 不需要手工定义“废话词”。
- 不需要外部 teacher 重写 CoT。
- 不需要 RL 长度惩罚。
- 先让同一个强 reasoning 模型在多题上下文里自然变短，再只保留正确样本。

这比“直接过滤 token < 2000 的 JustRL 轨迹”更主动，因为它改变了生成分布，而不是只在长轨迹里挑短样本。

## 本仓库复现建议

### V0：只做数据诊断

目标：验证多题打包是否真的能压缩你当前 teacher。

默认建议：

```yaml
teacher_model: nvidia/OpenMath-Nemotron-1.5B 或 JustRL-DeepSeek-1.5B
N: 3
sample_size: 300
samples_per_pack: 4
temperature: 0.6
top_p: 0.95
max_model_len: 4096 或 8192
max_new_tokens: 3072
```

统计：

- per-question output tokens
- p50 / p90 / p95 / max
- boxed rate
- correctness
- parse success
- double boxed rate
- first correct answer 后的剩余 token 数

判据：

- 如果 per-question p95 明显低于单题 rollout，说明 ConPress 可用。
- 如果 correctness 掉太多，先降低 `N` 到 2。
- 如果解析失败多，先固定输出模板，不急着训练。

当前 Nemo 9 题 smoke 说明：必须使用模型 chat template；裸 prompt 会诱导续写题目列表。Nemo 在 chat template 下会自然使用 `Problem A/B/C` reasoning heading，而不稳定遵守 `Answer A/B/C` 硬格式。因此工程解析应记录 `reasoning_split_ok` 与 `answer_pairing_ok`，不能只看严格格式。

### V1：生成压缩 SFT 数据

默认数据构造：

- 从当前 candidate prompt 池或 MATH 中等难度样本抽题。
- 每个 pack 随机选 `N=3` 道题。
- 每个 pack 采样 `4-8` 次。
- 对每个 question slot 拆出独立 response。
- 用本地 parser + `math-verify` 过滤正确样本。
- 只保留：
  - 能用 `Question i` / `Answer i` / 模型特定分隔符稳定切出对应题段
  - 对应题段内 boxed answer 可解析
  - response token 在 `[128, 2048]` 或 `[128, 3072]`
  - 没有明显截断
  - 没有多次重复同一 final answer 段

训练集格式仍然走现有 SFT JSONL：

```json
{"prompt": "...", "solution": "<compressed cot> \\boxed{...}"}
```

### V2：快速训练对照

建议对照：

- baseline：当前 `outputs/stage1_mix_long_sft/checkpoint-300`
- long teacher SFT：单题 JustRL/Nemotron 正确轨迹
- shortest rejection：同题多采样中最短正确轨迹
- ConPress SFT：多题上下文压力生成的正确轨迹

训练默认：

```yaml
backend: trl_peft
model_name: outputs/stage1_mix_long_sft/checkpoint-300
adapter_base_model: Qwen/Qwen2.5-Math-1.5B 本地 snapshot
epochs: 1
learning_rate: 1e-5 或 2e-5
batch_size: 1-2
gradient_accumulation_steps: 16-32
max_seq_length: 4096
loss_mode: sft
```

评测必须同时看：

- MATH500 accuracy
- GSM8K accuracy
- avg output tokens
- boxed rate
- truncation / max token hit rate

ConPress 的目标不是单纯涨准确率，而是在准确率接近的前提下降 token。

## 实现注意点

- 多题 pack 必须保证题目独立，不要把同一道题重复放进一个 pack。
- 题目难度最好分层采样；简单题过多会让模型学到过短模板，难题过多会导致 correctness 掉。
- `N=3` 是默认首选；3090 上如果 `max_model_len=4096` 不够，可以先 `N=2`。
- 解析不能按 boxed 切分；boxed 只用于分段后的答案抽取与正确性校验。
- 主切分锚点应是 `Question i` / `Answer i` / `Problem A-B-C` 这类显式题号；必要时再用模型 discourse marker 辅助。
- 对 Nemo 这类输出，优先接受 `<think>` 内第一次按顺序出现的 `Problem A/B/C` 作为 reasoning block，再从尾部 boxed summary 或 block 内最后 boxed 配对答案。
- 图形/asy 题会显著增加多题输出漂移，早期 probe 应先过滤。
- 对没有 `<think>` 的模型，直接把每个 question 对应的 reasoning+answer 当 response span。
- 不要把所有 pack 的完整 multi-question output 直接训练；必须拆成单题样本。
- 如果 teacher 是 Nemotron 而 student 是 Qwen2.5-Math，格式迁移风险仍然存在；最好先用同 base 的 JustRL/Qwen 系模型验证。

## 和 OPD / Lightning OPD 的关系

ConPress 和 OPD 解决的问题不同：

| 方法 | 目标 | 数据来源 | 训练方式 |
|---|---|---|---|
| Lightning OPD | 模仿 teacher token 分布 | student rollout + teacher logprob | distillation loss |
| Revisiting OPD | 修 sampled-token OPD 噪声 | on-policy rollout + teacher topK | topK reverse-KL |
| ConPress | 压缩 CoT 长度 | 同模型 multi-question rollout | 普通 SFT |

推荐顺序：

1. 先用 ConPress 生成短且正确的 SFT 数据，解决 CoT 过长。
2. 再用 OPD/topK loss 做分布吸收，不要直接在很长 teacher 轨迹上蒸馏。
3. 如果 ConPress 数据有效，后续可以把它作为 OPD 的 candidate rollout 来源。

## 风险

- 多题上下文会降低单题准确率，必须 rejection filtering。
- 解析失败会污染训练数据，尤其是模型没有稳定 `Question i` 输出格式时。
- 过强压缩会砍掉必要推理，难题上更明显。
- 论文使用 16k/32k 上下文；本仓库 4096 上下文下需要先做小规模诊断。
- ConPress 不是“无损压缩”；它是准确率-长度 trade-off，需要用 avg tokens 和 accuracy 一起判断。

## 最小可执行路线

最建议本仓库下一步做：

1. 基于现有 MATH/candidate prompt 池实现 `multi_question_pack`。
2. 用 `N=3`、`temperature=0.6`、`top_p=0.95` 生成 300 pack。
3. 先按题号锚点拆分 per-question block，再在每个 block 内抽取 boxed answer 做正确性过滤。
4. 和单题 rollout 对比长度分布。
5. 如果 p95 明显下降且 correctness 可接受，再扩大到 `1000-3000` 单题样本。
6. 训练一个 1 epoch LoRA，对比 MATH500/GSM8K 的 accuracy 与 avg tokens。
