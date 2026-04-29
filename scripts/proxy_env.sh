#!/usr/bin/env bash

# 仅配置网络代理。用法：
# source scripts/proxy_env.sh
# source scripts/proxy_env.sh 17898

proxy_port="${1:-17897}"
proxy_url="http://127.0.0.1:${proxy_port}"
no_proxy_value="${NO_PROXY:-127.0.0.1,localhost,10.*,192.168.*,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,172.21.*,172.22.*,172.23.*,172.24.*,172.25.*,172.26.*,172.27.*,172.28.*,172.29.*,172.30.*,172.31.*}"

export HTTP_PROXY="$proxy_url"
export HTTPS_PROXY="$proxy_url"
export http_proxy="$proxy_url"
export https_proxy="$proxy_url"
export ALL_PROXY="$proxy_url"
export all_proxy="$proxy_url"
export NO_PROXY="$no_proxy_value"
export no_proxy="$NO_PROXY"

echo "HTTP_PROXY=$HTTP_PROXY"
echo "HTTPS_PROXY=$HTTPS_PROXY"
echo "ALL_PROXY=$ALL_PROXY"
echo "NO_PROXY=$NO_PROXY"
