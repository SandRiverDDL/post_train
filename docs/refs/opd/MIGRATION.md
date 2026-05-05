# OPD 迁移到本项目的建议

## 起点选择

student：

- 首选 SFT：`outputs/stage1_qwen3_1p7b_bf16_lora_qwen3_4b_instruct2507_math_correct_len4096_sft_ddp3_b2_acc6_bs36/checkpoint-150`
- 备选 ASFT：`outputs/stage1_qwen3_1p7b_bf16_lora_qwen3_4b_instruct2507_math_correct_len4096_asft_topk_ddp3_bs33/checkpoint-100`

teacher：

- 首选：`Qwen/Qwen3-4B-Instruct-2507`
- 原因：同 tokenizer / chat template，teacher probe 正确率和长度更可控。

不建议当前使用：

- hard-tail 上输出极长且正确率低的 Qwen3-4B Non-Thinking teacher。
- tokenizer 不一致的 teacher，除非先解决 topK token id 对齐。

## 最小迁移路线

第一阶段先做 Thinking Machines / verl k1 小测试：

```text
loss_mode=k1
use_policy_gradient=True
use_task_rewards=False
student=SFT checkpoint-150
teacher=Qwen3-4B-Instruct-2507
dataset=MATH train Level 4
prompt_token <= 256
max_response_length=2048
train prompts=500
n=1
```

当前入口：

```bash
CUDA_VISIBLE_DEVICES=3,5,6 bash runs/opd/run_tm_qwen3_math_l34_1k.sh
```

当前数据：

```text
../verl/data/opd_math/hendrycks_l4_prompt256_500_seed42/train.parquet
```

这批数据从 Level 4 原始 `1690` 条中过滤 prompt token `>256` 的 `82` 条，剩 `1608` 条后抽 `500` 条；抽样 prompt token mean `84.5`、p95 `154.05`、max `251`。

成功标准不是分数提升，而是：

- 链路能跑通。
- boxed/parse 不快速崩。
- 截断率不上升。
- 输出长度不持续膨胀。

第二阶段做 THUNLP-style topK reward：

```text
K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
N_RESPONSES=2 或 4
```

第三阶段才做 topK reverse-KL：

```text
teacher_top_k=16
support 内重新归一化
special-token mask
truncation sample-level mask
```

## 必须补的工程防线

- 达到 `max_response_length` 且没有 EOS 的样本整条 mask。
- 如果 response 内出现 `<|im_start|>`、`<|im_end|>`、`<think>`、`</think>`、EOS/PAD 等 special token，后续应考虑把对应 token 的 OPD loss mask 掉，避免把“提前结束/结束思考”训练成高 advantage。
- 记录 `truncated_sample_rate`。
- 记录 response length mean/p90/p95/max。
- 记录多 boxed、重复 `Final Answer`、高频 `wait/check`。
- 如果 batch 全部被 mask，跳过 actor update。

当前小测试暂不改 `../verl` 做 special-token mask，先用原生 k1 口径确认链路、吞吐和长度行为。

## 显存判断

topK 相比 sampled-token 主要多存：

```text
teacher_topk_ids: [B, L, K]
teacher_topk_logprobs: [B, L, K]
```

K=16、response 2048、batch 8 时，这部分通常只是 MB 到十几 MB 级别。真正的显存风险仍在：

- actor/student full logits
- rollout vLLM 与 actor 同卡
- 长 response
- batch / micro batch
- 是否 materialize full log_probs

teacher 独占 GPU 时，teacher topK 本身不是主要瓶颈。
