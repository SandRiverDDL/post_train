# 远端代理与 Codex 运行

这份文档只说明一件事：当服务器本身不能直接出海时，如何临时借本地机场访问 GitHub、Hugging Face、OpenAI/Codex。

## 原则

- 不修改系统全局代理。
- 不修改用户 shell 全局默认代理。
- 只在当前项目 shell 中启用代理。
- 默认通过本地机器维持一条 SSH 反向隧道。

## 本地机器操作

本地机器已有 HTTP 代理：127.0.0.1:7897。

推荐在本地单独开一个 tmux 窗口，保持这条隧道：

```bash
ssh -N -R 17897:127.0.0.1:7897 fsw@10.103.12.90
```

如果密码登录太烦，优先给服务器配置免密登录，再考虑 autossh 保活。

## 服务器项目内启用

进入项目目录后：

```bash
cd ~/dataY/post_train
source .runtime/env.sh
enable_local_proxy
project_env_summary
```

或直接使用项目脚本：

```bash
cd ~/dataY/post_train
source scripts/proxy_env.sh 17897
```

默认 enable_local_proxy 使用 17897 端口；如果你换了反向端口，也可以显式传参：

```bash
enable_local_proxy 17898
```

关闭代理：

```bash
disable_local_proxy
```

## 连通性测试

启用代理后，先测：

```bash
curl -I -m 15 https://www.google.com
curl -I -m 15 https://github.com
curl -I -m 15 https://huggingface.co
curl -I -m 15 https://api.openai.com/v1/models
```

预期：

- Google 返回 200
- GitHub 返回 200
- Hugging Face 返回 200
- OpenAI API 常见返回 401，这说明网络已打通，只是没有携带 API key

## 关键陷阱：隧道可能“假活”

ssh -R 会出现一种常见故障：

- 本地 ssh -N -R 进程还在
- 服务器 127.0.0.1:17897 端口也还能 connect
- 但真实 HTTP 请求已经转不出去，只会一直卡住直到超时

所以不要只用下面两种信号判断：

- ps -ef | grep ssh -N -R
- Python socket.connect(("127.0.0.1", 17897))

真正可靠的检查只有：

```bash
curl -I -m 15 https://www.google.com
```

如果出现：

- Connection refused
- Operation timed out
- unexpected eof

就说明这条隧道已经不可用，即使 ssh 进程表面上还活着。

此时的处理方式是：

1. 杀掉本地旧的 ssh -R 进程
2. 重新建立一条新的隧道
3. 再次用 curl -I -m 15 https://www.google.com 验证

## Codex CLI

确认代理已启用后，再在同一个 shell 中运行：

```bash
codex
```

如果你只想让某个命令走代理，也可以不全局启用当前 shell，而是临时前缀：

```bash
HTTP_PROXY=http://127.0.0.1:17897 HTTPS_PROXY=http://127.0.0.1:17897 codex
```

验证 Codex 主后端是否可达：

```bash
curl -I -m 15 "https://chatgpt.com/backend-api/codex/models?client_version=0.128.0"
```

常见返回：

- 200 / 401 / 405 都说明后端可达
- 只要不是 Connection refused、timeout 或 TLS 错误，就不是纯网络故障

## 注意事项

- 这套方案依赖你的本地机器在线。
- 本地关机、休眠、断网、SSH 断开后，服务器代理会立刻失效。
- 不建议把代理写进 ~/.bashrc 或 ~/.zshrc，否则这个账号下的其它终端也会被你的本地机器绑住。
- 若服务器访问异常，先执行 disable_local_proxy，再重试本地直连或重新建立隧道。
