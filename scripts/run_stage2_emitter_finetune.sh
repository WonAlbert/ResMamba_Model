#!/usr/bin/env bash
# 个体识别微调：解冻 backbone，使用不含 communication_emitters 的下游 pool
#
# 用法：
#   STAGE2_EMITTER_CKPT=runs/checkpoints/stage2_emitter/best.pt scripts/run_stage2_emitter_finetune.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"

if [[ -z "${STAGE2_EMITTER_CKPT:-}" ]]; then
  mapfile -t _ckpts < <(find "${ROOT}/runs/checkpoints/stage2_emitter" -name "best.pt" 2>/dev/null | sort)
  if [[ ${#_ckpts[@]} -eq 0 ]]; then
    echo "未找到 stage2 emitter best.pt，请设置 STAGE2_EMITTER_CKPT=..." >&2
    exit 1
  fi
  STAGE2_EMITTER_CKPT="${_ckpts[-1]}"
fi
if [[ ! -f "${STAGE2_EMITTER_CKPT}" ]]; then
  echo "stage2 emitter 权重不存在: ${STAGE2_EMITTER_CKPT}" >&2
  exit 1
fi

python scripts/train_pipeline.py \
  --stage stage2 \
  --task emitter \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_emitter_finetune.yaml \
  --pool downstream_emitter_train \
  --val-pool downstream_emitter_val \
  --rfdata-root "${RFDATA_ROOT}" \
  --pretrained-checkpoint "${STAGE2_EMITTER_CKPT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --early-stopping-patience "${EARLY_STOP_PATIENCE}" \
  --lr "${LR}" \
  --num-workers "${NUM_WORKERS}" \
  --amp \
  --log-dir runs/tensorboard \
  --output-dir runs/checkpoints/stage2_emitter_finetune
