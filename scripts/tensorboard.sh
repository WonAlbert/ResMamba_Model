#!/usr/bin/env bash
# 只加载 CURRENT_RUN 对应目录；换 run 后需重启本脚本（否则 TB 会缓存旧 tag）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_ROOT="/root/tf-logs"
if [[ -n "${1:-}" ]]; then
  LOGDIR="$1"
elif [[ -f "${TF_ROOT}/CURRENT_RUN" ]]; then
  RUN="$(tr -d '[:space:]' < "${TF_ROOT}/CURRENT_RUN")"
  LOGDIR="${TF_ROOT}/${RUN}"
else
  LOGDIR="${TF_ROOT}/resmamba_current"
fi
if [[ ! -d "$LOGDIR" ]]; then
  echo "logdir 不存在: $LOGDIR" >&2
  echo "请先启动 train.py，或查看 ${TF_ROOT}/LOGDIR" >&2
  exit 1
fi
PORT="${PORT:-6006}"
echo "TensorBoard logdir: $LOGDIR (port ${PORT})"
echo "换 run 后请 Ctrl+C 重启本脚本；AutoDL 面板 6007 也需重启 TensorBoard。"
exec tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT" --reload_interval 5
