#!/usr/bin/env bash
# 前台串行：预训练 → loss 分析 → infer → stage2 → 汇总 → stage3
# 在 tmux won 中运行（推荐）：
#   bash scripts/run_in_tmux_won.sh
# 或本机前台：
#   PRETRAIN_RUN=pretrain_20260826_pipeline bash scripts/run_full_pipeline.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source scripts/env.sh

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export RFDATA_ROOT="${RFDATA_ROOT:-${ROOT}/dataset}"

PRETRAIN_RUN="${PRETRAIN_RUN:-}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
WAIT_PRETRAIN="${WAIT_PRETRAIN:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_STAGE3="${SKIP_STAGE3:-0}"
STAGE3_TASKS="${STAGE3_TASKS:-ld_intrapulse,ld_model,tx_modulation,ld_clustering,tx_clustering,prediction}"
PIPELINE_LOG="${PIPELINE_LOG:-${ROOT}/runs/pipeline_logs/pipeline.log}"

log() {
  echo "[$(date -Iseconds)] $*"
  echo "[$(date -Iseconds)] $*" >> "$PIPELINE_LOG"
}

mkdir -p "${ROOT}/runs/pipeline_logs"

if [[ -z "$PRETRAIN_RUN" ]]; then
  PRETRAIN_RUN="pretrain_$(date +%Y%m%d_%H%M%S)"
fi

PRETRAIN_DIR="${ROOT}/runs/experiments/${PRETRAIN_RUN}"
PRETRAIN_CKPT="${PRETRAIN_DIR}/ckpts/best.ckpt"
PIPELINE_STATE="${PRETRAIN_DIR}/pipeline_state.json"

log "pipeline start PRETRAIN_RUN=${PRETRAIN_RUN} RUN_PRETRAIN=${RUN_PRETRAIN}"

if [[ "$RUN_PRETRAIN" == "1" ]]; then
  log "step 0/5 pretrain (foreground) run=${PRETRAIN_RUN}"
  resume_args=(--run-name "$PRETRAIN_RUN")
  if [[ -f "${PRETRAIN_DIR}/ckpts/best.ckpt" ]] || [[ -f "${PRETRAIN_DIR}/ckpts/last.ckpt" ]]; then
    resume_args+=(--resume auto)
    log "resuming pretrain from existing ckpt in ${PRETRAIN_DIR}"
  fi
  python scripts/train.py --stage pretrain --config configs/pretrain.yaml "${resume_args[@]}"
fi

if [[ "$WAIT_PRETRAIN" == "1" ]]; then
  log "waiting for external pretrain to finish..."
  while true; do
    if pgrep -f "python scripts/train.py --stage pretrain" >/dev/null 2>&1; then
      sleep 120
      continue
    fi
    if [[ -f "$PRETRAIN_CKPT" ]]; then
      log "pretrain process ended and best.ckpt exists"
      break
    fi
    if [[ -f "${PRETRAIN_DIR}/train_state.json" ]] && grep -q '"exception"' "${PRETRAIN_DIR}/train_state.json" 2>/dev/null; then
      if [[ -f "$PRETRAIN_CKPT" ]]; then
        break
      fi
    fi
    sleep 60
  done
fi

if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  log "ERROR: missing $PRETRAIN_CKPT"
  exit 1
fi

log "step 1/5 analyze_pretrain_loss"
python scripts/analyze_pretrain_loss.py "$PRETRAIN_DIR" || log "WARN: analyze_pretrain_loss failed"

if [[ "$SKIP_INFER" != "1" ]]; then
  log "step 2/5 infer pretrain-sources (prediction val)"
  OUT_PRED="${ROOT}/outputs/prediction_${PRETRAIN_RUN}_val"
  python scripts/infer.py \
    --task prediction \
    --checkpoint "$PRETRAIN_CKPT" \
    --config configs/pretrain.yaml \
    --pretrain-sources \
    --per-dataset \
    --split val \
    --output-dir "$OUT_PRED" \
    || log "WARN: infer failed"
fi

STAGE2_RUN=""
if [[ "$SKIP_STAGE2" != "1" ]]; then
  STAGE2_RUN="stage2_pipeline_${PRETRAIN_RUN}"
  log "step 3/5 stage2 -> ${STAGE2_RUN}"
  python scripts/train.py \
    --stage stage2 \
    --config configs/stage2.yaml \
    --init-from "$PRETRAIN_CKPT" \
    --run-name "$STAGE2_RUN" \
    || log "WARN: stage2 failed"
  if [[ -f "${ROOT}/runs/experiments/${STAGE2_RUN}/ckpts/best.ckpt" ]]; then
    log "step 4/5 summarize_stage2_metrics"
    python scripts/summarize_stage2_metrics.py "${ROOT}/runs/experiments/${STAGE2_RUN}" --json \
      > "${ROOT}/runs/experiments/${STAGE2_RUN}/stage2_summary.json" \
      || true
    python scripts/summarize_stage2_metrics.py "${ROOT}/runs/experiments/${STAGE2_RUN}" || true
  fi
else
  log "SKIP_STAGE2=1"
fi

STAGE2_CKPT="${ROOT}/runs/experiments/${STAGE2_RUN}/ckpts/best.ckpt"
if [[ "$SKIP_STAGE3" != "1" ]] && [[ -n "$STAGE2_RUN" ]] && [[ -f "$STAGE2_CKPT" ]]; then
  log "step 5/5 stage3 tasks: ${STAGE3_TASKS}"
  IFS=',' read -ra TASKS <<< "$STAGE3_TASKS"
  for task in "${TASKS[@]}"; do
    task="$(echo "$task" | xargs)"
    [[ -z "$task" ]] && continue
    log "stage3 task=${task}"
    python scripts/train.py \
      --stage stage3 \
      --config configs/stage3.yaml \
      --task "$task" \
      --init-from "$STAGE2_CKPT" \
      || log "WARN: stage3 ${task} failed"
  done
else
  log "SKIP_STAGE3 or no stage2 ckpt"
fi

python - <<'PY' "$PIPELINE_STATE" "$PRETRAIN_RUN" "${STAGE2_RUN:-}" "$PRETRAIN_CKPT"
import json, sys
from datetime import datetime, timezone
path, pretrain_run, stage2_run, ckpt = sys.argv[1:5]
payload = {
    "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "pretrain_run": pretrain_run,
    "pretrain_ckpt": ckpt,
    "stage2_run": stage2_run or None,
    "status": "done",
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY

log "pipeline complete pretrain=${PRETRAIN_RUN} stage2=${STAGE2_RUN:-none}"
