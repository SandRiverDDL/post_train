# verl OPD 运行时与显存诊断

## 背景

当前 verl OPD 目标是先用两张 24GB 卡跑通 smoke / 小规模实验，不把两卡效率优化作为第一目标。

当前标准路径使用：

- `actor + student vLLM` hybrid 资源池
- 独立 `teacher_pool` 常驻 teacher vLLM
- `loss_mode=k1`
- `use_policy_gradient=True`
- `use_task_rewards=False`

## 两卡调度判断

当前可行基线：

```text
GPU A: actor/FSDP + student vLLM
GPU B: teacher vLLM
```

建议起步参数：

```text
STUDENT_GPU_MEMORY_UTILIZATION=0.55
TEACHER_GPU_MEMORY_UTILIZATION=0.70
```

如果 `0.55` 稳定，再单独尝试：

```text
STUDENT_GPU_MEMORY_UTILIZATION=0.60
```

不建议一开始使用 `student=0.65` 或 `0.85`。此前 `student=0.85` 启动失败的直接原因是 actor 初始化后 GPU 已占约 `8GB`，student vLLM 仍按 `0.85 * 23.56GB` 预留预算，启动阶段即超过剩余显存。vLLM sleep 发生在 rollout 之后，不能解决初始化预算冲突。

`teacher=0.85` 也不建议作为默认值。GPU0/7 smoke 中，teacher vLLM 可以完成初始化，但在 teacher `prompt_logprobs` 的 full-vocab `log_softmax` 临时张量处 OOM；降低到 `teacher=0.70` 后，`TOTAL_TRAINING_STEPS=3` 跑完。这里降低 utilization 不是降低 softmax 本身的张量大小，而是减少 vLLM KV/cache 预留，给 `log_softmax` 运行期临时显存留余量。

## actor 与 student vLLM 的显存关系

`actor` 与 `student vLLM` 语义上是同一个 base model，但显存中不是同一份 CUDA tensor。

```text
actor/HF/FSDP:
  负责训练
  需要 forward/backward
  持有 base、LoRA、梯度、优化器/offload 状态、FSDP/通信状态

student vLLM:
  负责 rollout
  持有 vLLM 内部推理权重、KV cache、prefix cache、paged attention 内存池
```

LoRA 训练只减少可训练参数与同步负担，不代表 actor 只占 adapter 显存。base 权重仍参与 actor 前向/反向，也在 student vLLM 中有推理副本。

verl 当前采用：

```text
actor weights/adapter -> update_weights -> student vLLM
```

而不是 actor 与 vLLM 共享同一份 base 权重显存。

## vLLM sleep 语义

verl 中需要区分 sleep level：

```text
level=1: 主要释放 KV cache，保留 weights
level=2: 释放 weights + KV cache
```

本地 vLLM 版本满足 verl 默认 `VLLM_SLEEP_LEVEL=2`，但 LoRA adapter 模式下 hybrid rollout 会偏向 `level=1`，以避免每步搬运 base weights。

`gpu_memory_utilization` 仍然约束 vLLM 初始化和 wake 后的显存预算。sleep 可以释放阶段性显存，但不能让 vLLM 启动时忽略原始预算。

## student vLLM 与 teacher vLLM 同卡

当前不建议在标准 verl OPD 路径下把 student vLLM 与 teacher vLLM 放同一张卡。

原因是 teacher scoring 当前在 agent loop 内联发生：

```text
student vLLM rollout
-> 立刻请求 teacher vLLM 计算 teacher_logprobs
-> batch 进入 replay buffer
-> sleep student rollout
-> actor 训练
```

因此 teacher 计算时 student vLLM 尚未 sleep。若两者同卡，会在 rollout 阶段同时 awake。

理想的同卡串行调度需要改数据流：

```text
student rollout
-> sleep student vLLM
-> wake teacher vLLM
-> batch 级 teacher scoring
-> sleep teacher vLLM
-> actor train/update
-> wake student vLLM
```

这不是当前 shell 参数能表达的能力，需要把 teacher scoring 从 agent loop 内联阶段移到 trainer batch 阶段。

## 精度与量化建议

当前 OPD baseline 统一使用 BF16：

```text
actor: BF16 + LoRA + FSDP/offload
student vLLM: BF16
teacher vLLM: BF16
```

暂不把 vLLM FP8/torchao 量化或 FSDP 4bit 训练作为 baseline：

- teacher 量化会改变 token-level logprob，直接影响 OPD 训练信号。
- student rollout 量化会改变采样分布，影响 on-policy 数据。
- 当前 FSDP actor 路径没有明确的标准 bitsandbytes QLoRA `load_in_4bit` 配置入口。
- 本地 QAT/NVFP4 配置不等同于可直接启用的 QLoRA。

后续若 baseline 跑通，可以单独开分支测试 student vLLM 量化；teacher 量化应更谨慎。

## 当前建议

先跑通标准 verl OPD：

```text
actor + student vLLM 同卡
teacher vLLM 独占另一张卡
student utilization 从 0.55 起步
teacher utilization 使用 0.70
三者均使用 BF16
```

如果两卡效率成为主要瓶颈，再考虑小改 verl 调度，把 teacher scoring 延后到 batch 级，并显式管理 student/teacher vLLM 的 sleep/wake。
