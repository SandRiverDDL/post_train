# SFT 运行说明

## 单卡

```bash
.venv/bin/python scripts/train_sft.py --config configs/stage1/mix_long_justrl_chat_lt2048_random2000_sft.yaml
```

## 多卡 DP / DDP

当前 `backend=trl_peft` 支持通过 `torchrun` 做多卡数据并行。封装入口：

```bash
GPUS=0,1,2 CONFIG=configs/stage1/mix_long_justrl_chat_lt2048_random2000_sft.yaml \
  runs/sft/train_sft_ddp.sh
```

也可以直接传配置路径：

```bash
GPUS=0,1,2 runs/sft/train_sft_ddp.sh configs/stage1/mix_long_justrl_chat_lt2048_random2000_sft.yaml
```

说明：

- `GPUS` 控制 `CUDA_VISIBLE_DEVICES`。
- `NPROC_PER_NODE` 默认等于 `GPUS` 数量。
- DDP 下每张卡都会加载一份 QLoRA 模型副本。
- 有效 batch 为 `batch_size * gradient_accumulation_steps * GPU 数量`。
- 若想保持单卡等效 batch，需要按 GPU 数量反向调小 `gradient_accumulation_steps` 或 `batch_size`。
- 当前不建议用 `backend=unsloth` 做多卡；主线多卡 SFT 只按 `backend=trl_peft` 维护。

## Qwen3-1.7B 全量 clean<2048 对照

当前可复用配置：

```bash
GPUS=0,2,7 CONFIG=configs/stage1/qwen3_1p7b_mix_long_justrl_chat_lt2048_all_consistent_sft.yaml \
  runs/sft/train_sft_ddp.sh

GPUS=0,2,7 CONFIG=configs/stage1/qwen3_1p7b_mix_long_justrl_chat_lt2048_all_consistent_dft.yaml \
  runs/sft/train_sft_ddp.sh

GPUS=0,2,7 CONFIG=configs/stage1/qwen3_1p7b_mix_long_justrl_chat_lt2048_all_consistent_asft_topk.yaml \
  runs/sft/train_sft_ddp.sh
```

评测口径：

- dev 选 checkpoint：`data/eval/math500_dev200.jsonl`
- benchmark：`math500`、`gsm8k`
- eval 配置：`configs/eval/qwen3_1p7b.yaml`
- 生成上限：`max_new_tokens=2048`

已确认结果：

| loss | best ckpt | MATH500 | GSM8K | MATH tokens | GSM8K tokens |
|---|---:|---:|---:|---:|---:|
| SFT | 100 | 0.644 | 0.7885 | 555.4 | 618.9 |
| DFT | 25 | 0.656 | 0.7923 | 329.5 | 208.7 |
| ASFT-topK | 50 | 0.680 | 0.8120 | 565.3 | 610.6 |

DFT 分数只略高，未达到强显著；但输出明显更短。ASFT-topK 在当前同口径下明显优于 SFT/DFT，不过输出长度接近 SFT，且仍有尾部重复风险。后续不建议只看 parse/boxed rate 判断质量。

## ASFT topK

`loss_mode=asft_topk` 是当前 DFT 的轻量 anchor 版本：

- loss 为 `DFT + asft_kl_weight * topK truncated KL(base || current)`。
- LoRA 下通过 `disable_adapter()` 获取 base/reference topK，不加载第二份模型。
- 默认 `asft_top_k=32`、`asft_kl_weight=0.03`。
- 当前只维护 `backend=trl_peft`；`backend=unsloth` 会直接报错。
- 实现会先 no-grad base forward 取 topK 并释放，再 current forward 反传；仍会额外产生一次 full logits，若 `batch_size=4` OOM，优先改为 `batch_size=2`。
- 三卡 3090 实测 `batch_size=4, gradient_accumulation_steps=3` OOM；当前 ASFT YAML 使用 `batch_size=2, gradient_accumulation_steps=6` 保持有效 batch 约 36。
- 当前 best checkpoint 按 `math500_dev200` 选择为 `checkpoint-50`；最终 checkpoint-143 的 MATH500/GSM8K 为 `0.678 / 0.8014`，不如 ckpt50。
