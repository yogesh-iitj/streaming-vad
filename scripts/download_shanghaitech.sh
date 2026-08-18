#!/usr/bin/env bash
# Download the ShanghaiTech Campus Dataset.
# This dataset is distributed via Google Drive / Baidu links maintained by
# the community (there is no single stable direct-download URL, and file
# IDs on Google Drive change/expire). The current, actively maintained
# links are listed in:
#   https://github.com/StevenLiuWen/ano_pred_cvpr2018 (see its "Datasets" section)
#   https://svip-lab.github.io/dataset/campus_dataset.html
# Approx size: several GB (frames + per-frame test masks).
#
# NOT executed automatically by this codebase — Google Drive bulk
# downloads generally need `gdown` and the current file ID, which you
# should copy from the pages above rather than trust a hardcoded one here.
set -euo pipefail

RAW_DIR="${1:-data/raw/ShanghaiTech}"
mkdir -p "$RAW_DIR"

GDRIVE_ID="${SHANGHAITECH_GDRIVE_ID:-}"

if [ -z "$GDRIVE_ID" ]; then
  echo "Set SHANGHAITECH_GDRIVE_ID to the current Google Drive file ID, e.g.:"
  echo "  SHANGHAITECH_GDRIVE_ID=xxxxxxxx bash scripts/download_shanghaitech.sh"
  echo "Find the current ID at:"
  echo "  https://github.com/StevenLiuWen/ano_pred_cvpr2018"
  echo "  https://svip-lab.github.io/dataset/campus_dataset.html"
  echo ""
  echo "Requires: pip install gdown"
  exit 1
fi

cd "$RAW_DIR"
gdown --id "$GDRIVE_ID" -O shanghaitech.zip
unzip -q shanghaitech.zip

echo "Next step (after checking the extracted folder names match):"
echo "  python scripts/prepare_dataset.py --dataset shanghaitech \\"
echo "      --raw-root $RAW_DIR/<extracted-folder> \\"
echo "      --out data/ShanghaiTech --dry-run"
