#!/usr/bin/env bash
# 下载 EMind 预训练语料中的 4 个公开子数据集
# 用法：
#   source scripts/env.sh
#   source scripts/download_env.sh
#   bash scripts/download_em_public_datasets.sh all
#   bash scripts/download_em_public_datasets.sh panoradio radarcomm transmitter hisarmod

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/download_env.sh"

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  echo ">>> 下载: ${url}"
  echo "    保存: ${out}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x "${ARIA2_MAX_CONNECTION}" -s "${ARIA2_SPLIT}" -k 1M \
      --file-allocation=none --auto-file-renaming=false \
      --continue=true -d "$(dirname "${out}")" -o "$(basename "${out}")" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    wget ${WGET_OPTS} -O "${out}" "${url}"
  else
    curl -L --retry 10 --retry-delay 5 -C - -o "${out}" "${url}"
  fi
}

download_hisarmod() {
  local dest="${DATASET_EXTERNAL_ROOT}/hisarmod"
  mkdir -p "${dest}"
  cat <<'EOF'
[HisarMod2019.1] 需 IEEE 账号登录后下载（约 5.13 GB）：
  1. 注册/登录 https://www.ieee.org/profile/public/createwebaccount/showRegister.html
  2. 打开 https://ieee-dataport.org/open-access/hisarmod-new-challenging-modulated-signals-dataset
  3. 点击 LOGIN TO ACCESS DATASET FILES -> 下载 HisarMod2019.1.zip
  4. 将 zip 放到: dataset/external/hisarmod/HisarMod2019.1.zip

若已登录浏览器，也可在已登录会话中复制直链后用 wget -c 下载。
EOF
  if [[ -f "${dest}/HisarMod2019.1.zip" ]]; then
    echo "已存在: ${dest}/HisarMod2019.1.zip"
  fi
}

download_panoradio() {
  local dest="${DATASET_EXTERNAL_ROOT}/panoradio_hf"
  mkdir -p "${dest}"
  local base="https://www.panoradio-sdr.de/wp-content/uploads"
  # 推荐单文件 npy（约 5 GB，比分卷 zip 更省事）
  download_file "${base}/dataset_panoradio_hf.npy" "${dest}/dataset_panoradio_hf.npy"
  download_file "${base}/dataset_panoradio_hf_tags.csv" "${dest}/dataset_panoradio_hf_tags.csv"
  download_file "${base}/dataset_panoradio_hf_readme.txt" "${dest}/dataset_panoradio_hf_readme.txt"
}

download_radarcomm() {
  local dest="${DATASET_EXTERNAL_ROOT}/radarcomm"
  mkdir -p "${dest}"
  download_file "https://www.androcs.com/uploads/RadComAWGN.zip" "${dest}/RadComAWGN.zip"
  download_file "https://www.androcs.com/uploads/RadComDynamic.zip" "${dest}/RadComDynamic.zip"
  download_file "https://www.androcs.com/uploads/RadComOta2.45GHz.zip" "${dest}/RadComOta2.45GHz.zip"
  if command -v unzip >/dev/null 2>&1; then
    for z in RadComAWGN.zip RadComDynamic.zip RadComOta2.45GHz.zip; do
      if [[ -f "${dest}/${z}" && ! -d "${dest}/${z%.zip}" ]]; then
        unzip -q "${dest}/${z}" -d "${dest}/${z%.zip}"
      fi
    done
  fi
}

download_transmitter() {
  local dest="${DATASET_EXTERNAL_ROOT}/transmitter_classification"
  mkdir -p "${dest}"
  # Morin et al. 2019 官方公开包（wiki: tx-id）
  download_file "https://xp.cortexlab.fr/storage/monoRxSet_012019.tgz" "${dest}/monoRxSet_012019.tgz"
  if [[ -f "${dest}/monoRxSet_012019.tgz" && ! -d "${dest}/monoRxSet_012019" ]]; then
    tar -xzf "${dest}/monoRxSet_012019.tgz" -C "${dest}"
  fi
  cat <<'EOF'

[说明] EMind 论文中 Transmitter Classification 约 1190 万样本，来自多轮 CorteXlab 实验汇总。
官方 wiki 目前仅公开 monoRxSet_012019.tgz 子集；完整复现需在 FIT/CorteXlab 上运行 gr-txid 生成。
EOF
}

usage() {
  cat <<'EOF'
用法: bash scripts/download_em_public_datasets.sh <target>

target:
  all              下载全部（HisarMod 仅打印 IEEE 指引）
  hisarmod         HisarMod2019.1（IEEE DataPort，需手动登录）
  panoradio        Panoradio HF（~5 GB npy）
  radarcomm        RadarCommDataset（3 个 zip，约 500MB+）
  transmitter      Transmitter Classification 官方子集（monoRxSet）

环境:
  source scripts/download_env.sh
  # 可选安装 aria2 加速: apt-get install -y aria2  或  conda install -c conda-forge aria2
EOF
}

main() {
  local target="${1:-all}"
  case "${target}" in
    all)
      download_hisarmod
      download_panoradio
      download_radarcomm
      download_transmitter
      ;;
    hisarmod) download_hisarmod ;;
    panoradio) download_panoradio ;;
    radarcomm) download_radarcomm ;;
    transmitter) download_transmitter ;;
    -h|--help) usage ;;
    *) echo "未知 target: ${target}" >&2; usage; exit 1 ;;
  esac
  echo "完成。数据目录: ${DATASET_EXTERNAL_ROOT}"
}

main "$@"
