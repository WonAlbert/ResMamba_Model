#!/usr/bin/env bash
# 等待指定 Stage2 跑完后，用其 best.ckpt 串行启动 Stage3（low_rank_prototype=false）。
#
# 用法：
#   STAGE2_RUN=stage2_nolr_20260831_181047 bash scripts/wait_stage2_then_stage3.sh
# tmux：
#   STAGE2_RUN=... bash scripts/run_in_tmux_won.sh   # 或本脚本自带 --tmux
#
# 环境变量：
#   STAGE2_RUN       必填（或传参 $1）：stage2 run 目录名
#   STAGE3_TAG       默认 nolr_<timestamp>
#   STAGE3_PROFILE   分类任务 profile，默认 disc_fast
#   POLL_SEC         轮询间隔秒，默认 60
#   SKIP_WAIT       =1 则不等待，直接用已有 best 启 Stage3

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export RFDATA_ROOT="${RFDATA_ROOT:-${ROOT}/dataset}"

STAGE2_RUN="${STAGE2_RUN:-${1:-}}"
if [[ -z "$STAGE2_RUN" ]]; then
  echo "ERROR: 请设置 STAGE2_RUN 或传入 stage2 run 名" >&2
  exit 1
fi

STAGE2_DIR="${ROOT}/runs/experiments/${STAGE2_RUN}"
STAGE2_CKPT="${STAGE2_DIR}/ckpts/best.ckpt"
STAGE2_LOG="${ROOT}/runs/pipeline_logs/${STAGE2_RUN}.log"
STAGE3_TAG="${STAGE3_TAG:-nolr_$(date +%Y%m%d_%H%M%S)}"
STAGE3_PROFILE="${STAGE3_PROFILE:-disc_fast}"
POLL_SEC="${POLL_SEC:-60}"
SKIP_WAIT="${SKIP_WAIT:-0}"

# 与近期 Stage2/3 实验一致：最差优先顺序；prediction 单独用 prediction profile
STAGE3_TASKS_CLASS=(ld_model ld_clustering ld_intrapulse tx_modulation tx_clustering)
PIPELINE_LOG="${ROOT}/runs/pipeline_logs/stage3_after_${STAGE2_RUN}_${STAGE3_TAG}.log"

mkdir -p "${ROOT}/runs/pipeline_logs"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$PIPELINE_LOG"
}

stage2_train_running() {
  pgrep -f "python scripts/train.py --stage stage2 .*--run-name ${STAGE2_RUN}" >/dev/null 2>&1
}

wait_stage2() {
  log "waiting Stage2 run=${STAGE2_RUN} poll=${POLL_SEC}s ckpt=${STAGE2_CKPT}"
  while true; do
    if stage2_train_running; then
      if [[ -f "${STAGE2_DIR}/train_state.json" ]]; then
        local ep mon
        ep="$(python3 -c "import json; j=json.load(open('${STAGE2_DIR}/train_state.json')); print(j.get('epoch'))" 2>/dev/null || echo "?")"
        mon="$(python3 -c "import json; j=json.load(open('${STAGE2_DIR}/train_state.json')); print(j.get('monitor_value'))" 2>/dev/null || echo "?")"
        log "Stage2 still running epoch=${ep} monitor_value=${mon}"
      else
        log "Stage2 still running (no train_state yet)"
      fi
      sleep "$POLL_SEC"
      continue
    fi
    if [[ -f "$STAGE2_CKPT" ]]; then
      # 进程已退出且有 best：视为完成（含 early-stop / 正常结束）
      if [[ -f "${STAGE2_DIR}/train_state.json" ]]; then
        python3 - <<PY | tee -a "$PIPELINE_LOG"
import json
from pathlib import Path
p = Path("${STAGE2_DIR}/train_state.json")
j = json.loads(p.read_text())
exc = j.get("exception")
print(f"[stage2] finished epoch={j.get('epoch')} best_score={j.get('best_model_score')} exception={exc!r}")
if exc and not Path("${STAGE2_CKPT}").is_file():
    raise SystemExit(2)
PY
      fi
      log "Stage2 done; using ${STAGE2_CKPT}"
      return 0
    fi
    log "Stage2 process gone but missing best.ckpt; wait ${POLL_SEC}s more..."
    sleep "$POLL_SEC"
  done
}

run_stage3() {
  log "Stage3 start tag=${STAGE3_TAG} init-from=${STAGE2_CKPT} profile=${STAGE3_PROFILE} low_rank_prototype=false (configs/stage3.yaml)"
  local task
  for task in "${STAGE3_TASKS_CLASS[@]}"; do
    log "===== STAGE3 ${task} ====="
    python scripts/train.py --stage stage3 --config configs/stage3.yaml \
      --profile "${STAGE3_PROFILE}" --task "${task}" \
      --init-from "${STAGE2_CKPT}" \
      --run-name "stage3_${task}_${STAGE3_TAG}" \
      2>&1 | tee -a "$PIPELINE_LOG"
  done
  log "===== STAGE3 prediction ====="
  python scripts/train.py --stage stage3 --config configs/stage3.yaml \
    --profile prediction --task prediction \
    --init-from "${STAGE2_CKPT}" \
    --run-name "stage3_prediction_${STAGE3_TAG}" \
    2>&1 | tee -a "$PIPELINE_LOG"
  log "Stage3 queue DONE tag=${STAGE3_TAG}"
}

log "=== wait_stage2_then_stage3 ==="
log "STAGE2_RUN=${STAGE2_RUN} STAGE3_TAG=${STAGE3_TAG}"

if [[ "$SKIP_WAIT" == "1" ]]; then
  [[ -f "$STAGE2_CKPT" ]] || { log "ERROR: missing ${STAGE2_CKPT}"; exit 1; }
  log "SKIP_WAIT=1; jump to Stage3"
else
  wait_stage2
fi

run_stage3
echo DONE | tee -a "$PIPELINE_LOG"
