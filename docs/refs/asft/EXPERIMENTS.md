# ASFT 实验摘记与本仓库建议

## 论文实验口径

论文训练域：

- 数学推理：NuminaMath CoT，训练规模 `10k / 30k / 100k`。
- 医学知识：MedMCQA，训练规模 `10k / 30k / 100k`。
- 代码生成也作为评估域之一。

数学设置：

- 模型：`Qwen2.5-7B`。
- max length：`2048`。
- global batch size：`256`。
- learning rate：`5e-5`。
- epoch：`1`。
- ASFT `lambda=0.05`。
- 评测：Math500、Minerva Math、OlympiadBench、AIME 2024、AMC 2023；CoT prompting，`temperature=1.0`，max length `4096`，16 runs 平均。

医学设置：

- max length：`512`。
- global batch size：`64`。
- learning rate：`2e-5`。
- epoch：`3`。
- ASFT `lambda=0.05`。

官方 README 另给工程建议：

- bf16/fp16 下推荐 `kl_weight=0.03`，更大的 KL 权重可能放大精度噪声。
- 支持 DeepSpeed 和 LoRA，但 README 备注 native run 更稳定。
- LoRA 医学任务推荐 `rank=8, alpha=16, dropout=0.05, lr=5e-4`。

## 结果要点

论文主表显示：

- DFT 在数学上比 SFT 强很多，但在医学/知识任务上可能明显不稳定。
- ASFT 在数学、医学、代码上整体更稳。
- `SFT w/ KL` 不是 ASFT 的替代，很多设置下不如 ASFT。
- forward KL 比 reverse KL 更稳，作者解释为 forward KL 更 mode-covering。

对本仓库最重要的读法：

- ASFT 适合解决“DFT/重加权导致分布漂移”的训练问题。
- ASFT 不负责清洗 teacher 轨迹，也不能把长 CoT 自动变短。
- 如果数据本身是错误、截断、串题、多 boxed，ASFT 仍然会学到这些噪声。

## 本仓库建议实验

建议先用 `outputs/stage1_mix_long_sft/checkpoint-300` 作为起点，而不是 base model。

第一轮只做小规模对照：

| 实验 | 数据 | 目标 | 建议 |
|---|---|---|---|
| SFT baseline | clean summary / mix-long matched | 复现基线 | 1 epoch |
| DFT | 同数据 | 看重加权是否提升 | 同学习率 |
| ASFT | 同数据 | 看 KL anchor 是否稳定 | `lambda=0.03/0.05` |

数据优先级：

1. 已有短 clean solution / mix-long matched 数据，作为稳定基线。
2. ConPress 正确且可切分的短轨迹。
3. Qwen3-4B-2507 Thinking 的 `</think>` 后 summary，要求正确、闭合、未截断；完整 thinking 暂不使用。

不要第一轮使用：

- JustRL/Nemotron raw 长 `<think>`。
- Qwen3-4B-2507 完整 thinking 轨迹。
- 命中 max token 的样本。
- `asy/diagram` 样本。
- parser 只能靠 tail boxed 伪配对的样本。

## 需要观察的指标

训练中：

- `dft_weight_mean / min / max`。
- `asft_kl_mean`。
- `loss_sft / loss_dft / loss_kl` 分项。
- response token 长度分布。

评测中：

- Math500 pass@1。
- GSM8K pass@1。
- avg output tokens。
- boxed rate / parse rate。
- max token hit rate。

如果 ASFT 提升准确率但输出变长，需要和当前“短 CoT 可学性”目标分开判断。

## 风险

- KL reference 选错会把模型拉回不合适的分布。继续训练 adapter 时，reference 应该是训练起点而不是原始 base。
- LoRA 下 `disable_adapter()` 作为 reference 最省显存；全量训练加载两份模型会明显增加显存。
- DFT 权重会降低低概率 token 的学习强度，可能不适合包含大量新格式、新符号、新领域知识的数据。
- ASFT 对学习率和 KL 权重仍敏感；不要直接照搬 7B 全局 batch 到 1.5B/3090。
