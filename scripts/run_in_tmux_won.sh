#!/usr/bin/env bash
# 在用户实际使用的 tmux won 会话（多为 won-0）里前台跑完整流水线
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRETRAIN_RUN="${PRETRAIN_RUN:-pretrain_20260826_pipeline}"
WINDOW_NAME="${WINDOW_NAME:-pipeline}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux 未安装" >&2
  exit 1
fi

# AutoDL 常见：attach 的是 won-0（group won），不是新建的 won 会话
pick_tmux_session() {
  if tmux has-session -t won-0 2>/dev/null; then
    echo won-0
    return
  fi
  if tmux has-session -t won1 2>/dev/null; then
    echo won1
    return
  fi
  if tmux has-session -t won 2>/dev/null; then
    echo won
    return
  fi
  tmux new-session -d -s won -n shell
  echo won
}

TMUX_SESSION="$(pick_tmux_session)"

pkill -f "bash scripts/run_full_pipeline.sh" 2>/dev/null || true
pkill -f "python scripts/train.py --stage pretrain" 2>/dev/null || true
sleep 2

tmux list-windows -t "$TMUX_SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$WINDOW_NAME" \
  && tmux kill-window -t "${TMUX_SESSION}:${WINDOW_NAME}" 2>/dev/null || true

mkdir -p "${ROOT}/runs/pipeline_logs"

RUN_CMD="cd ${ROOT} && source scripts/env.sh && export PYTHONPATH=\$(pwd):\$PYTHONPATH && export RFDATA_ROOT=\${RFDATA_ROOT:-${ROOT}/dataset} && export PRETRAIN_RUN=${PRETRAIN_RUN} && export RUN_PRETRAIN=1 && export WAIT_PRETRAIN=0 && bash scripts/run_full_pipeline.sh"

tmux new-window -t "$TMUX_SESSION" -n "$WINDOW_NAME" bash -lc "
  set -o pipefail
  exec > >(tee -a ${ROOT}/runs/pipeline_logs/pipeline.log) 2>&1
  echo '=== pipeline @ tmux ${TMUX_SESSION}:${WINDOW_NAME} ==='
  ${RUN_CMD}
  echo '=== pipeline finished; Enter to close ==='
  read
"

tmux select-window -t "${TMUX_SESSION}:${WINDOW_NAME}"

echo "已在 tmux 会话 '${TMUX_SESSION}' 窗口 '${WINDOW_NAME}' 前台启动。"
echo "查看: tmux attach -t ${TMUX_SESSION}"
echo "切窗口: tmux select-window -t ${TMUX_SESSION}:${WINDOW_NAME}"
echo "PRETRAIN_RUN=${PRETRAIN_RUN}"
