# OPD Sources

## 主要来源

- THUNLP OPD repo：`https://github.com/thunlp/OPD`
  - 标题：`Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe`
  - arXiv：`2604.13016`
  - 重点：OPD 成败条件、topK token reward、off-policy cold start、teacher-aligned prompt selection。

- Hugging Face paper page：`https://huggingface.co/papers/2604.13016`
  - 重点：摘要、作者、相似论文列表。

- Thinking Machines：`https://thinkingmachines.ai/blog/on-policy-distillation/`
  - 重点：student rollout + teacher dense token reward；reverse KL；OPD 与 SFT/RL 对比。

- Revisiting OPD：`https://arxiv.org/abs/2603.25562`
  - 标题：`Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes`
  - 重点：sampled-token OPD 的失败模式；teacher topK support；truncated reverse-KL；top-p rollout 和 special-token masking。

- Lightning OPD：`https://arxiv.org/abs/2604.13010`
  - 标题：`Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation`
  - 重点：offline OPD、teacher consistency、预计算 teacher logprob。

- StableOPD：`https://arxiv.org/abs/2604.08527`
  - 标题：`Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models`
  - 重点：length inflation、truncation collapse、reference divergence constraint、rollout mixture distillation。

- G-OPD / ExOPD：`https://arxiv.org/abs/2602.12125`
  - 标题：`Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation`
  - 重点：把 OPD 统一到 KL-constrained RL，引入 reference model 与 reward scaling。

- TIP：`https://huggingface.co/papers/2604.14084`
  - 标题：`TIP: Token Importance in On-Policy Distillation`
  - 重点：student entropy 与 teacher-student divergence 作为 token importance。

- slime OPD docs：`https://www.mintlify.com/THUDM/slime/advanced/on-policy-distillation`
  - 重点：OPD 可作为 RL advantage estimator 上的 additive KL penalty，支持 SGLang/Megatron teacher。

## 本地相关文档

- `docs/refs/lightning-opd/`
- `docs/refs/revisiting-opd/`
- `docs/analysis/lightning_opd_gap_diagnostics.md`
- `../verl/cdocs/ARCHITECTURE.md`
- `../verl/cdocs/STATE.md`

## 当前阅读结论

- THUNLP OPD 与 Revisiting OPD 对本项目最有直接价值。
- Thinking Machines / verl 标准 k1 OPD 可做 smoke，但不宜作为最终主力。
- Lightning OPD 的 teacher consistency 理论重要，但本项目当前 top1 离线实验已暴露明显失稳，不建议继续扩大。
- StableOPD/TIP/G-OPD 是后续补丁或扩展，不应抢在 topK OPD 稳定性验证之前。

