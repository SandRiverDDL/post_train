# ASFT Sources

## Primary Sources

- OpenReview: `Anchored Supervised Fine-Tuning`
  - https://openreview.net/forum?id=PORko7QT64
  - ICLR 2026 Poster，Published `2026-01-26`，Last Modified `2026-04-11`。
- PDF:
  - https://openreview.net/pdf?id=PORko7QT64
- arXiv:
  - https://arxiv.org/abs/2509.23753
- Official implementation:
  - https://github.com/zhuchichi56/ASFT

## Related Source

- DFT paper: `On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification`
  - https://arxiv.org/abs/2508.05629
  - DFT 官方仓库入口见论文页和 HF paper card：`https://github.com/yongliang-wu/DFT`

## Key Facts Used

- OpenReview abstract states that DFT reweights SFT with token probabilities, but lacks distributional anchoring; ASFT adds lightweight KL regularization.
- PDF Section 3.3 gives DFT objective with `stop-gradient` target-token probability weighting.
- PDF Section 4.1-4.2 frames DFT in RWR and attributes instability to distributional drift.
- PDF Section 4.2 defines ASFT with `KL(pi_base || pi_theta)`.
- PDF Section 5.1 lists math and medical training settings, including `Qwen2.5-7B`, NuminaMath CoT, max length `2048`, global batch `256`, learning rate `5e-5`, 1 epoch, and `lambda=0.05`.
- PDF Table 1 shows ASFT generally outperforming SFT/DFT across math and medical benchmarks, while DFT is stronger but less stable than SFT.
- PDF Section 5.3 reports forward KL outperforming reverse KL in their ablation.
- Official README says ASFT was accepted to ICLR 2026, code was released, and ASFT support was merged into LLaMA-Factory main; it recommends `kl_weight=0.03` for mixed precision and documents LoRA/DeepSpeed usage.

## Notes

- 用户提到 `AST`，但公开论文、OpenReview、arXiv 和官方仓库均使用 `ASFT`。
- 本文档没有把 ASFT 当作数据生成/CoT 压缩方法；它是 SFT loss 的替代或增强。
