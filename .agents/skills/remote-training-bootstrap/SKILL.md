---
name: remote-training-bootstrap
description: 将当前 LLM 训练项目迁移到远程 GPU 服务器时使用。适用于远程开发、环境恢复、目录规划、模型/数据/cache 布局与训练 smoke。
---

# Remote Training Bootstrap：远程训练环境落地

## 目标

将当前项目迁移到远程 GPU 服务器，并形成可重建的开发与训练环境。

优先级：

1. 持久化目录正确
2. 环境可重建
3. 模型、数据、输出分离
4. 能完成最小训练 smoke

# 适用场景

- 新建远程 GPU 训练机
- 从本地迁移到远程服务器开发
- 容器重建后恢复训练环境

# 输入信息

开始前先确认：

- SSH 连接方式与认证方式
- GPU 与驱动信息：`nvidia-smi`
- 持久化目录与临时目录
- 代码仓库地址与目标分支
- 底模来源路径
- 数据集来源路径
- 是否需要 `vllm`
- 是否需要 `wandb`
- 服务器是否需要 `push` 权限

# 主流程

1. 检查环境边界
- 先验证 SSH 能稳定非交互登录，并确认 `scp` 可用
- 确认 GPU、驱动、CUDA 兼容性
- 确认哪些目录会持久化，哪些会在容器销毁后丢失
- 先做网络 smoke：分别检查 GitHub、Hugging Face、PyPI 是否可达

2. 规划目录
- 代码目录只放仓库
- 数据、模型、输出、cache 分开放
- 不把 checkpoint、wandb、HF cache 写进临时目录
- 若服务器实际只有一个持久化根目录，则临时收口到仓库内 `.runtime/`

3. 恢复代码与 git
- 拉取仓库到持久化代码目录
- 明确远程开发以服务器仓库为真源
- 默认只在一台主开发环境上 `push`
- 只迁移必要产物，不默认整包复制 `outputs/` 或历史 archive

4. 恢复 Python 环境
- 先确认依赖真源，再决定是否使用当前目录的 `pyproject.toml` / `uv.lock`
- 使用项目内 `.venv`
- 不依赖容器里预装的临时 Python 状态
- 若 `uv.lock` 已失效，直接按确认后的 `pyproject.toml` 重建，不强求复用旧锁文件

5. 配运行时目录
- 默认把项目级环境变量写入 `.runtime/env.sh`，不写进 `.bashrc`
- 配 `HF_HOME`、`WANDB_DIR`、`TMPDIR`、`UV_CACHE_DIR`
- 远端首次 `uv sync` 前先配置镜像源与下载参数

6. 跑最小验证
- import 关键训练依赖
- 检查模型路径可读
- 检查数据路径可读
- 跑一个最小 smoke，而不是直接上正式训练

7. 记录恢复命令
- 记录环境变量
- 记录环境重建命令
- 记录训练启动命令

# `env.sh` 模板

```bash
mkdir -p .runtime/{hf_cache,wandb,tmp,uv_cache}

cat > .runtime/env.sh <<'EOF'
export HF_HOME="$(pwd)/.runtime/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENDPOINT="https://hf-mirror.com"
export WANDB_DIR="$(pwd)/.runtime/wandb"
export TMPDIR="$(pwd)/.runtime/tmp"
export UV_CACHE_DIR="$(pwd)/.runtime/uv_cache"
export HF_TOKEN="<your-huggingface-token>"
EOF

chmod 600 .runtime/env.sh
source .runtime/env.sh
```

说明：

- 只在项目内保存项目级环境变量
- 凭证只在执行时注入，不写进仓库
- 不把 token、镜像源、cache 路径写进 `.bashrc`

# 输出格式

Bootstrap Result

Environment Findings

- GPU / driver / CUDA
- 持久化目录
- 风险点

Proposed Layout

- 代码目录
- 模型目录
- 数据目录
- 输出与 cache 目录

Commands

- 环境变量
- 环境恢复命令
- 训练 smoke 命令

Open Risks

- 版本兼容
- 权限
- 网络 / 代理
