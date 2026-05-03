# ConPress Sources

## Paper

- Title: ConPress: Learning Efficient Reasoning from Multi-Question Contextual Pressure
- Authors: Jie Deng, Shining Liang, Jun Li, Hongzhi Li, Yutao Xie
- arXiv: https://arxiv.org/abs/2602.01472
- PDF: https://arxiv.org/pdf/2602.01472
- Submitted: 2026-02-01
- Current version checked: v1

## Key Source Passages Used

- Abstract and method summary: arXiv abstract, lines 40-42.
- Self-compression phenomenon and multi-question pressure: PDF Section 2, especially Figure 2 / Figure 3 and lines 180-240.
- Method pipeline: PDF Section 3, lines 294-371.
- Main experimental setup and hyperparameters: PDF Section 4.1, lines 397-423.
- Main results: PDF Table 2 and Section 4.2, lines 373-459.
- N ablation and position ablation: PDF Section 4.3, lines 468-579.
- Behavior analysis: PDF Section 5.2, lines 609-661.
- Sampling/parsing/training details: PDF Appendix B, lines 1105-1151.

## Notes

- The user referred to this as an ICML 2026 paper. In public search results checked on 2026-05-02, I only confirmed the arXiv preprint and did not confirm an official ICML 2026 proceedings or accepted-paper page.
- No official code repository was confirmed during this review.
- The implementation recommendations in `REVIEW.md` are adapted to this repository's current Qwen math post-training workflow and 3090-style resource constraints.
