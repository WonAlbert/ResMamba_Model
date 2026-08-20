#!/usr/bin/env bash
# 下载 EMind 预训练涉及的 4 个公开子数据集（参考命令，可按需单独执行各段）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/download_env.sh"

usage() {
  cat <<'EOF'
用法: bash scripts/download_emind_sources.sh [all|hisarmod|panoradio|radarcomm|txid]

  hisarmod   - HisarMod2019.1（IEEE DataPort 或 HF 镜像）
  panoradio  - Panoradio HF（~5.3 GB npy）
  radarcomm  - RadarCommDataset（3 个 zip）
  txid       - Transmitter Classification / tx-id（monoRxSet）
  all        - 依次下载除 hisarmod 外的可直接 wget/aria2 数据集

HisarMod 官方包需 IEEE 账号登录后手动下载，见脚本内注释。
EOF
}

TARGET="${1:-all}"

hf_download() {
  if command -v hf >/dev/null 2>&1; then
    hf download "$@"
    return
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$@"
    return
  fi
  echo "错误：未找到 hf / huggingface-cli。请先执行：source scripts/download_env.sh" >&2
  exit 1
}

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  echo ">>> ${url}"
  echo "    -> ${out}"
  if [[ -f "${out}" ]] && [[ "$(stat -c%s "${out}" 2>/dev/null || echo 0)" -gt 1048576 ]]; then
    echo "    已存在且 >1MB，跳过"
    return 0
  fi
  if command -v aria2c >/dev/null 2>&1; then
    if aria2c ${ARIA2C_OPTS} --timeout=120 --connect-timeout=60 \
      -d "$(dirname "${out}")" -o "$(basename "${out}")" "${url}"; then
      :
    else
      echo "    aria2 失败，尝试 curl ..."
      curl ${CURL_OPTS} --max-time 3600 -o "${out}.part" "${url}" && mv -f "${out}.part" "${out}"
    fi
  else
    curl ${CURL_OPTS} --max-time 3600 -o "${out}.part" "${url}" && mv -f "${out}.part" "${out}"
  fi
  local size
  size="$(stat -c%s "${out}" 2>/dev/null || echo 0)"
  if [[ "${size}" -lt 1048576 ]]; then
    echo "错误：下载未完成或文件过小（${size} 字节）: ${out}" >&2
    return 1
  fi
  echo "    完成: $(du -h "${out}" | awk '{print $1}')"
}

print_txid_unreachable_help() {
  cat <<'EOF'

[CorteXlab 服务器不可达] xp.cortexlab.fr 在国内/AutoDL 网络常出现 TLS 已连接但 0 字节无响应。

可选方案：
  1) 配置代理后重试（如有境外 HTTP 代理）：
       export https_proxy=http://127.0.0.1:7890
       export http_proxy=http://127.0.0.1:7890
       bash scripts/download_emind_sources.sh txid

  2) 在可访问该站的机器（本地+VPN/海外 VPS）下载后上传到服务器：
       # 本机浏览器或 wget 下载：
       https://xp.cortexlab.fr/storage/monoRxSet_012019.tgz
       # 上传到 AutoDL：
       scp -P <端口> monoRxSet_012019.tgz root@<host>:/root/autodl-tmp/ResMamba_Signal_Model/dataset/external/transmitter_classification/

  3) 暂时跳过 txid，先下载其他三个数据集：
       bash scripts/download_emind_sources.sh panoradio
       bash scripts/download_emind_sources.sh radarcomm

官方页面：https://wiki.cortexlab.fr/doku.php?id=tx-id
EOF
}

download_panoradio() {
  local d="${EMIND_RAW_ROOT}/panoradio_hf"
  mkdir -p "${d}"
  echo "[Panoradio HF] -> ${d}"
  aria2c ${ARIA2C_OPTS} -d "${d}" -o dataset_panoradio_hf.npy \
    "https://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf.npy"
  curl ${CURL_OPTS} -o "${d}/dataset_panoradio_hf_tags.csv" \
    "https://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf_tags.csv"
  curl ${CURL_OPTS} -o "${d}/dataset_panoradio_hf_readme.txt" \
    "https://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf_readme.txt"
}

download_radarcomm() {
  local d="${EMIND_RAW_ROOT}/radarcommdataset"
  mkdir -p "${d}"
  echo "[RadarCommDataset] -> ${d}"
  for name in RadComAWGN.zip RadComDynamic.zip RadComOta2.45GHz.zip; do
    aria2c ${ARIA2C_OPTS} -d "${d}" -o "${name}" "https://www.androcs.com/uploads/${name}"
  done
  if [[ ! -d "${d}/RadarCommDataset" ]]; then
    local repo_url="https://github.com/ANDROComputationalSolutions/RadarCommDataset.git"
    if [[ -n "${GITHUB_PROXY}" ]]; then
      repo_url="${GITHUB_PROXY%/}/${repo_url}"
    fi
    git clone "${repo_url}" "${d}/RadarCommDataset"
  fi
}

download_txid() {
  local d="${EMIND_RAW_ROOT}/transmitter_classification"
  mkdir -p "${d}"
  echo "[Transmitter Classification / tx-id] -> ${d}"
  if ! download_file "https://xp.cortexlab.fr/storage/monoRxSet_012019.tgz" "${d}/monoRxSet_012019.tgz"; then
    print_txid_unreachable_help
    return 1
  fi
  if [[ ! -d "${d}/monoRxSet_012019" ]]; then
    tar -xzf "${d}/monoRxSet_012019.tgz" -C "${d}"
  fi
}

download_hisarmod_hf_mirror() {
  local d="${EMIND_RAW_ROOT}/hisarmod2019.1"
  mkdir -p "${d}"
  echo "[HisarMod2019 HF 镜像] -> ${d}"
  if ! hf auth whoami >/dev/null 2>&1; then
    echo "请先登录: hf auth login" >&2
    return 1
  fi
  local user
  user="$(hf auth whoami 2>/dev/null | head -1 || true)"
  echo "当前 HF 账号: ${user}"
  cat <<'EOF'

[门禁数据集] SICEAI/HisarMod2019_* 需在 Hugging Face 官网（非镜像页）同意条款：
  1. 浏览器打开（务必登录同一账号）：
     https://huggingface.co/datasets/SICEAI/HisarMod2019_train
     https://huggingface.co/datasets/SICEAI/HisarMod2019_test
  2. 点击 “Agree and access repository” / 同意并访问
  3. 回到终端重试本脚本

若仍 Access denied，建议改用 IEEE 官方完整包（推荐，780k 样本）：
  https://ieee-dataport.org/open-access/hisarmod-new-challenging-modulated-signals-dataset
  下载 HisarMod2019.1.zip 放到: dataset/external/hisarmod/HisarMod2019.1.zip

EOF
  if ! hf_download SICEAI/HisarMod2019_train --repo-type dataset --local-dir "${d}/hf_train"; then
    echo "hf_train 下载失败（多为未在官网同意条款）。" >&2
    return 1
  fi
  hf_download SICEAI/HisarMod2019_test --repo-type dataset --local-dir "${d}/hf_test"
}

case "${TARGET}" in
  panoradio) download_panoradio ;;
  radarcomm) download_radarcomm ;;
  txid) download_txid ;;
  hisarmod) download_hisarmod_hf_mirror ;;
  all)
    download_panoradio
    download_radarcomm
    if ! download_txid; then
      echo "txid 下载失败，已跳过（见上方说明）。"
    fi
    echo
    echo "HisarMod2019.1 请按 README 中 IEEE DataPort 步骤手动下载，或执行："
    echo "  bash scripts/download_emind_sources.sh hisarmod"
    ;;
  -h|--help|help) usage ;;
  *) usage; exit 1 ;;
esac

echo "完成。数据目录: ${EMIND_RAW_ROOT}"
