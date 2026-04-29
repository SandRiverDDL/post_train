#!/usr/bin/env bash

# 兼容旧文件名：这里只配置代理，不启动训练。
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/proxy_env.sh" "$@"
