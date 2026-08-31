#!/usr/bin/env bash
# 依次跑 ld_clustering / tx_clustering / prediction 各 1 epoch（预训练 init）
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export PYTHONPATH="$(pwd):$PYTHONPATH"
INIT="${1:-runs/experiments/pretrain_20260824_112422/ckpts/best.ckpt}"
LOG_DIR="runs/experiments/probes_cluster_pred_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local cfg="$2"
  local log="$LOG_DIR/${name}.log"
  echo "=== START $name $(date -Is) ===" | tee -a "$log"
  python scripts/train.py --stage stage2 --config "$cfg" \
    --init-from "$INIT" \
    --run-name "probe_${name}_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee -a "$log"
  echo "=== DONE $name $(date -Is) ===" | tee -a "$log"
}

run_one ld_clustering configs/stage2_probe_ld_clustering.yaml
run_one tx_clustering configs/stage2_probe_tx_clustering.yaml
run_one prediction configs/stage2_probe_prediction.yaml

echo "All probes finished. Logs: $LOG_DIR"
