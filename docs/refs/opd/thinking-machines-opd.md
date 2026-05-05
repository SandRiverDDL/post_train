# Thinking Machines / 标准 OPD

## 来源

- Blog：`https://thinkingmachines.ai/blog/on-policy-distillation/`
- 本地对应：`../verl` 的 distillation / teacher loop 实现。

## 方法

核心形式：

```text
student rollout y
teacher 在 prompt + y 的每个 response prefix 上给 logprob
用 teacher/student sampled-token logprob 构造 reverse-KL / k1 信号
```

verl 当前可表达为：

```text
distillation.distillation_loss.loss_mode=k1
distillation.distillation_loss.use_policy_gradient=True
distillation.distillation_loss.use_task_rewards=False
```

它的优点是工程路径最短：student rollout、teacher scoring、actor update 都在在线链路内。

## 本项目当前 baseline

当前先做小测试，而不是直接上 1000+ 混合数据：

```text
student: Qwen3-1.7B SFT checkpoint-150
teacher: Qwen3-4B-Instruct-2507
dataset: EleutherAI/hendrycks_math train Level 4
prompt filter: Qwen3 chat-template prompt tokens <= 256
sample_size: 500
max_response_length: 2048
n: 1
student_gpus: 2
teacher_gpus: 1
```

数据已生成到：

```text
../verl/data/opd_math/hendrycks_l4_prompt256_500_seed42/train.parquet
```

启动脚本：

```bash
CUDA_VISIBLE_DEVICES=3,5,6 bash runs/opd/run_tm_qwen3_math_l34_1k.sh
```

## 局限

它仍是 sampled-token OPD：每个位置只监督 student 实际采样出来的 token。

本项目已经在 Lightning OPD top1 中观察到类似风险：

- teacher 对 student sampled token 的 logprob 经常低于 student。
- 训练后输出变长。
- no-box 和重复推理上升。
- 放宽 generation cap 也不能完全救回。

因此它适合做 smoke，不适合在没有 topK/长度防线前长跑。

另一个风险是 special token 监督：如果 response 中出现 `<|im_end|>`、EOS、`<think>`、`</think>` 等 token，原生 k1 会把它们当普通 response token 参与 loss。本项目当前小测试先不 patch verl；若出现提前结束、过早 `</think>` 或格式坍缩，再加 response mask。

## 原实验要点

Thinking Machines 文中把 OPD 描述为 SFT/RL 的折中：student 自己采样 trajectory，teacher 给 dense token-level 评价。文中报告了数学推理任务中 OPD 相比 SFT/RL 的 compute efficiency，但具体实现是其 Tinker cookbook，而不是本仓库可直接复用的脚本。
