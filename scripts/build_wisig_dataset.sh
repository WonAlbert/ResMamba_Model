#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "[1/2] ManyTx.pkl -> npy (equalized_data_0)"
python scripts/convert_wisig_manytx_genNPY.py

echo "[2/2] npy -> RFData H5 + label_maps"
python scripts/prepare_datasets.py --output "${RFDATA_ROOT:-dataset}" --datasets wisig

echo "[done] wisig dataset ready under ${RFDATA_ROOT:-dataset}/h5"
