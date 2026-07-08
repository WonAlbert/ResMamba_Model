#!/usr/bin/env bash
# 一键 stage2 下游训练：依次训练 modulation / emitter / prediction / clustering
#
# 用法：
#   scripts/run_stage2_all.sh
#   PRETRAIN_CKPT=path/to/best.pt scripts/run_stage2_all.sh
#   scripts/run_stage2_all.sh modulation emitter   # 仅跑指定任务
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-3e-4}"

if [[ -z "${PRETRAIN_CKPT:-}" ]]; then
  mapfile -t _ckpts < <(find "${ROOT}/runs/checkpoints/pretrain_400m" -name "best.pt" 2>/dev/null | sort)
  if [[ ${#_ckpts[@]} -eq 0 ]]; then
    echo "未找到预训练 best.pt，请设置 PRETRAIN_CKPT=..." >&2
    exit 1
  fi
  PRETRAIN_CKPT="${_ckpts[-1]}"
fi
if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "预训练权重不存在: ${PRETRAIN_CKPT}" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=(modulation emitter prediction clustering)
fi

COMMON_ARGS=(
  --stage stage2
  --model-config configs/model_resmamba_400m.yaml
  --rfdata-root "${RFDATA_ROOT}"
  --pretrained-checkpoint "${PRETRAIN_CKPT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --early-stopping-patience "${EARLY_STOP_PATIENCE}"
  --lr "${LR}"
  --num-workers "${NUM_WORKERS}"
  --amp
  --log-dir runs/tensorboard
)

declare -A TASK_POOL=(
  [modulation]=downstream_modulation_train
  [emitter]=downstream_emitter_train
  [prediction]=downstream_prediction_train
  [clustering]=clustering_train
)
declare -A TASK_VAL_POOL=(
  [modulation]=downstream_modulation_val
  [emitter]=downstream_emitter_val
  [prediction]=downstream_prediction_val
  [clustering]=clustering_val
)
declare -A TASK_OUT=(
  [modulation]=runs/checkpoints/stage2_modulation
  [emitter]=runs/checkpoints/stage2_emitter
  [prediction]=runs/checkpoints/stage2_prediction
  [clustering]=runs/checkpoints/stage2_clustering
)

echo "========================================"
echo "Stage2 一键训练"
echo "  预训练权重 : ${PRETRAIN_CKPT}"
echo "  epochs       : ${EPOCHS}"
echo "  batch_size   : ${BATCH_SIZE}"
echo "  grad_accum   : ${GRAD_ACCUM}  (有效 batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  early_stop   : ${EARLY_STOP_PATIENCE}"
echo "  任务列表     : ${TASKS[*]}"
echo "========================================"

for task in "${TASKS[@]}"; do
  if [[ -z "${TASK_POOL[$task]+x}" ]]; then
    echo "未知任务: ${task}（可选: modulation emitter prediction clustering）" >&2
    exit 1
  fi

  TASK_CONFIG="configs/stage2_heads.yaml"
  if [[ "${task}" == "emitter" ]]; then
    TASK_CONFIG="configs/stage2_emitter.yaml"
  fi

  echo ""
  echo ">>> [${task}] 开始训练 (config=${TASK_CONFIG}) ..."
  python scripts/train_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --config "${TASK_CONFIG}" \
    --task "${task}" \
    --pool "${TASK_POOL[$task]}" \
    --val-pool "${TASK_VAL_POOL[$task]}" \
    --output-dir "${TASK_OUT[$task]}"
  echo ">>> [${task}] 完成，checkpoint -> ${TASK_OUT[$task]}"
done

echo ""
echo "全部 stage2 任务训练完成。"
