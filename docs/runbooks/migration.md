# Migration Runbook

这份文档只记录换服务器继续实验时，除 `git pull` 之外必须迁移的本地产物。

## 先同步 Git

先在源服务器提交并 push 代码、配置、脚本和文档；目标服务器再 `git pull`。

注意：以下目录默认被 `.gitignore` 忽略，不会随 Git 同步：

- `data/`
- `outputs/`
- `wandb/`
- `mlruns/`

## 当前必须迁移的实验产物

继续 Lightning-OPD / Nemo 实验至少需要：

- `data/lightning_opd/nemotron_candidate/`
- `outputs/stage1_mix_long_sft/checkpoint-300`
- `outputs/lightning_opd_nemotron_from_stage1_ckpt300_ep1`

如需复现 JustRL / Nemo rollout 诊断，再迁移：

- `data/rollout/`

## 模型缓存

确认目标服务器已有对应 Hugging Face 模型缓存，或修改 YAML 中的本地模型路径：

- `Qwen/Qwen2.5-Math-1.5B`
- `nvidia/OpenMath-Nemotron-1.5B`

## 推荐方式

使用 `rsync -avh --progress` 同步被 Git 忽略的目录。

若目标服务器需要继续查看或追加 MLflow 实验记录，也要同步 `mlruns/`。
