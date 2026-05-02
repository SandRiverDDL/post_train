# JustRL Rollout 诊断

## 数据范围

本诊断只看本地 JustRL rollout 产物：

- `data/rollout/math_sft/justrl_deepseek_1500/raw_samples.jsonl`
- `data/rollout/math_sft/justrl_deepseek_math_all_len3650/raw/raw_samples.merged.jsonl`
- `data/rollout/math_sft/justrl_deepseek_1500/train.rollout_rejection_lt2000_matched321.jsonl`

## 长度与格式统计

`justrl_deepseek_1500/raw_samples.jsonl`：

- response 数：`1500`
- 平均输出长度：`1013.6` tokens
- `p50=1040`，`p75=1177`，`p90=1304`，`p95=1387`，`max=1727`
- `<think>` 出现率：`46.3%`
- `</think>` 闭合率：`36.1%`
- `double_boxed`：`31.8%`
- 无 boxed：`59.4%`

`justrl_deepseek_math_all_len3650/raw/raw_samples.merged.jsonl`：

- response 数：`7498`
- 平均输出长度：`1442.1` tokens
- `p50=1465`，`p75=1840`，`p90=2134`，`p95=2318`，`p99=2619`，`max=3649`
- `<think>` 出现率：`70.6%`
- `</think>` 闭合率：`64.5%`
- `double_boxed`：`62.7%`
- 无 boxed：`31.9%`

`train.rollout_rejection_lt2000_matched321.jsonl`：

- 样本数：`321`
- 平均长度约 `777.7` words
- `<think>`：`100%`
- `boxed`：`100%`
- `double_boxed`：`90.3%`

## 主要推理模式

- 自问自答和反复确认非常强。常见句式包括 `Wait`、`let me double-check`、`verify`、`that seems consistent`。
- 简单题也会长链路推理。例如“最小能被 2 和 3 整除的平方数”这类题，正确答案是 `36`，但轨迹会列平方数、讲 LCM、讲质因数、再验证多次，达到约 `1000+` tokens。
- 常见结构是 `<think>...\\boxed{...}</think>` 后再写一段正式答案，并再次给出 `\\boxed{...}`。
- 枚举和分类讨论很频繁。数论、组合题常出现 `Case 1/2`、`Checking p=...`、`First/Second/Third` 等结构。
- verification 尾巴很长。很多轨迹在已经得到答案后，会再代入或举例验证一到多轮。

## 噪声与冗余模式

- `double_boxed` 很普遍：`<think>` 内已经有 `\\boxed{}`，`</think>` 后又重复一次。
- “Wait” 式自我纠错过多，容易把简单推理拉长。
- 简单题过度解释，可能把学生蒸馏成废话更多的风格。
- 长题容易截断，尤其组合、几何、图形题。截断轨迹常没有 boxed 或推到一半停住。
- 部分样本有读题漂移。例如开头出现与原题无关的短句，随后才进入原题。
- 几何/Asymptote 图题容易发散，进入坐标或角度推导后无法收束。
- 组合题常能识别成排列、循环、容斥，但计数公式未完成就被截断。

## 具体例子

### 简单题过度推理

题目：`What is the smallest positive perfect square that is divisible by both 2 and 3?`

现象：

- level 2，正确答案 `36`
- JustRL 输出约 `1092` tokens
- 先在 `<think>` 内列举平方数、LCM、质因数、验证多遍，并给出 `\\boxed{36}`
- `</think>` 后又写一段正式答案，再次给出 `\\boxed{36}`

### 组合题截断

题目：`Nine delegates, three each from three different countries...`

现象：

- level 5，输出约 `1654` tokens
- 能识别成圆桌排列与容斥问题
- 推理停在定义集合和 inclusion-exclusion 阶段，未给出 boxed
- `boxed=false`、`parse_ok=false`、`correct=false`

### 几何题发散

题目：`A cube has edges of length 1 cm and has a dot marked in the centre of the top face...`

现象：

- level 5，输出约 `1727` tokens
- 进入圆弧、旋转半径、路径长度等推导
- 多次改口半径和路径解释，最后未收束
- 无 boxed，不能作为 SFT 正样本

## 对训练使用的建议

- 不建议把 raw JustRL rollout 直接用于 SFT。
- 若用于 SFT，至少做长度、完整性和重复答案过滤。
- 对照实验建议保留三种版本：
  - 原始 rejection 正确轨迹
  - 只保留 `</think>` 后正式答案
  - 保留 `<think>`，但去掉 `</think>` 后重复答案和多余 verification
- 对 Lightning-OPD，JustRL 轨迹更适合作为固定 rollout 或 teacher 信号，而不是无清洗的 imitation target。
- 当前 parser 读到第一个 `\\boxed{...}` 不能证明轨迹完整；应同时检查是否截断、是否有 `</think>`、是否存在多个 boxed。
