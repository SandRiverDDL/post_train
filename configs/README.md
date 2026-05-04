# 配置入口

本目录保存训练、评测和数据准备 YAML。当前阶段不引入 Hydra；优先用清晰目录、README 和少量稳定模板降低配置查找成本。

## 当前有效配置

- `configs/eval/`：正式评测配置。Qwen3-1.7B no-think 模型优先看 `qwen3_1p7b_dev.yaml` 和 `qwen3_1p7b_benchmark.yaml`。
- `configs/stage1/`：SFT/DFT/ASFT 训练配置。当前 ConPress Qwen3 对照为：
  - `conpress_qwen3_1p7b_sft.yaml`
  - `conpress_qwen3_1p7b_dft.yaml`
  - `conpress_qwen3_1p7b_asft_topk.yaml`
- `configs/lightning_opd/`：Lightning-OPD 数据准备和离线 OPD 训练配置。当前 ConPress OPD 对照为：
  - `conpress_asft_qwen3_4b_teacher_math2000.yaml`
  - `train_conpress_asft_qwen3_4b_teacher_math2000.yaml`
- `configs/grpo/` 和 `configs/workflow/`：GRPO / DAPO-lite 当前主线配置。

## 历史或降权路线

- `configs/on_policy/`：旧 on-policy SFT / OPSFT 路线，当前不作为主线。
- `configs/simpo/`：旧 SIMPO 路线，当前不作为主线。
- `configs/stage2/`：旧两阶段路线配置，保留用于复现。
- `configs/archive/`：一次性小样本或已经归档的实验配置。

## 维护规则

- 同族配置超过 3 个时，优先在本 README 写索引，不继续依赖超长文件名记忆。
- 新 YAML 必须能从文件名看出路线、模型、数据和 loss；不能只写日期或临时编号。
- 核心业务逻辑不得写进 YAML 或 shell；YAML 只承载参数。
- 如果未来继续新增大量同族配置，再考虑引入 OmegaConf `extends` 或 Hydra；当前不做配置系统迁移。
