# OPD 方法横向总结

## 方法对比

| 方法 | 核心信号 | 主要优点 | 主要风险 | 本项目定位 |
|---|---|---|---|---|
| Thinking Machines / k1 | sampled token 的 teacher/student logprob | 最容易跑通在线链路 | sampled-token 噪声、长度膨胀 | smoke |
| THUNLP topK reward | topK token set 上的 3D weighted advantage | 比 top1 稳，仍贴近 RL/verl | reward 设计敏感 | 优先尝试 |
| topK truncated reverse-KL | teacher topK support 内局部分布 KL | 分布语义干净，针对 top1 失败 | 需要改 loss / gather | 第二优先 |
| Lightning OPD | 离线 fixed rollout + teacher logprob | 省 teacher server | teacher consistency 强，top1 已失稳 | 只保留离线 topK 数据价值 |
| StableOPD | reference + rollout mixture | 针对长度膨胀 | 不是独立主算法 | 稳定性补丁 |
| G-OPD / ExOPD | reward scaling + reference | 可能 beyond teacher | 参数和稳定性更难 | 暂缓 |
| TIP / entropy-aware | 选重要 token | 降低 token 计算量 | 需要额外统计 | 后续优化 |

## 是否互补

这些方法可以互补，但不应该一次全混。

推荐组合顺序：

1. 用 Thinking Machines k1 只验证在线链路。
2. 将 sampled-token 信号替换为 THUNLP topK reward，优先复刻 `only_stu + student_p + K=16`。
3. 如果 topK reward 仍不稳，再转 topK truncated reverse-KL。
4. 一旦出现长度膨胀，加入 StableOPD 风格的 reference / rollout mixture。
5. 只有稳定后，再考虑 TIP token selection 或 G-OPD reward extrapolation。

## 当前最推荐路线

本项目当前最实际的路线是：

```text
online student rollout
Qwen3-4B-Instruct-2507 teacher
K=16
先 THUNLP only_stu + student_p
监控长度/boxed/parse/截断
稳定后再试 teacher topK reverse-KL
```

不推荐：

- 继续扩大 Lightning OPD top1。
- 一开始就做 G-OPD / ExOPD。
- 用极长输出 teacher 直接做 OPD 主 teacher。
