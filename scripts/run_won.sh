#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

if [[ $# -eq 0 ]]; then
  echo "用法: scripts/run_won.sh <command> [args...]" >&2
  echo "示例: scripts/run_won.sh python scripts/train.py --stage pretrain --config configs/pretrain.yaml" >&2
  exit 1
fi

exec "$@"
