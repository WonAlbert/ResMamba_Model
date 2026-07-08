#!/usr/bin/env bash
# 项目统一 conda 环境：won
# 用法：source scripts/env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "请使用 source 加载本脚本，例如：source scripts/env.sh" >&2
  exit 1
fi

_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_CONDA_ENV_NAME="won"

if command -v conda >/dev/null 2>&1; then
  _CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
fi
_CONDA_ROOT="${_CONDA_ROOT:-${CONDA_ROOT:-$HOME/miniconda3}}"

if [[ ! -f "${_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "找不到 conda.sh：${_CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${_CONDA_ENV_NAME}"

cd "${_PROJECT_ROOT}"
export PYTHONPATH="${_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RFDATA_ROOT="${RFDATA_ROOT:-${_PROJECT_ROOT}/dataset}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-20}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-20}"
# 缓解 CUDA 显存碎片导致的 OOM 增长
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

unset _PROJECT_ROOT _CONDA_ENV_NAME _CONDA_ROOT
