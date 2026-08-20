#!/usr/bin/env bash
# EMind 公开子数据集下载环境（国内加速 + 断点续传）
# 用法：source scripts/download_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "请使用 source 加载本脚本，例如：source scripts/download_env.sh" >&2
  exit 1
fi

_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 激活 won 环境（hf / aria2 等工具在此环境中）
# shellcheck disable=SC1091
source "${_PROJECT_ROOT}/scripts/env.sh"

# 外部原始数据集根目录（与项目 RFData H5 分离）
export DATASET_EXTERNAL_ROOT="${DATASET_EXTERNAL_ROOT:-${_PROJECT_ROOT}/dataset/external}"
export EMIND_RAW_ROOT="${EMIND_RAW_ROOT:-${DATASET_EXTERNAL_ROOT}}"

# GitHub 加速代理（留空则直连）；示例：export GITHUB_PROXY=https://ghproxy.com
export GITHUB_PROXY="${GITHUB_PROXY:-}"

# Hugging Face 国内镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 大文件下载：优先 aria2 多连接（国内跨境更稳）；未安装则回退 wget/curl
export ARIA2_MAX_CONNECTION="${ARIA2_MAX_CONNECTION:-16}"
export ARIA2_SPLIT="${ARIA2_SPLIT:-16}"
export ARIA2C_OPTS="${ARIA2C_OPTS:--x ${ARIA2_MAX_CONNECTION} -s ${ARIA2_SPLIT} -k 1M --file-allocation=none --auto-file-renaming=false --continue=true}"
export CURL_OPTS="${CURL_OPTS:--L --retry 10 --retry-delay 5 -C -}"
export WGET_OPTS="${WGET_OPTS:---continue --tries=10 --timeout=60 --read-timeout=60 --waitretry=5}"

# 可选 HTTP/HTTPS 代理（按需取消注释）
# export https_proxy=http://127.0.0.1:7890
# export http_proxy=http://127.0.0.1:7890

mkdir -p "${DATASET_EXTERNAL_ROOT}"/{hisarmod,panoradio_hf,radarcomm,transmitter_classification}

unset _PROJECT_ROOT
