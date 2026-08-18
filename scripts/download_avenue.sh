#!/usr/bin/env bash
# Download the CUHK Avenue Dataset.
# Home page (has the current, authoritative download links — hosting has
# moved between CUHK pages and third-party mirrors over the years, so
# check here first rather than trusting a hardcoded URL):
#   http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
# A commonly used, actively maintained mirror with a direct download link
# and a re-hosted copy is linked from the reference implementation:
#   https://github.com/StevenLiuWen/ano_pred_cvpr2018 (see its "Datasets" section)
# Approx size: ~1 GB (training + testing videos + ground-truth masks).
#
# NOT executed automatically by this codebase. This script only creates the
# target directory and prints instructions — CUHK's file has moved between
# direct-link and Google-Drive hosting in the past, and a wrong hardcoded
# link is worse than no link. Fill in DOWNLOAD_URL yourself after checking
# the page above, then re-run.
set -euo pipefail

RAW_DIR="${1:-data/raw/Avenue}"
mkdir -p "$RAW_DIR"

DOWNLOAD_URL="${AVENUE_URL:-}"

if [ -z "$DOWNLOAD_URL" ]; then
  echo "Set AVENUE_URL to the current dataset zip/tar link, e.g.:"
  echo "  AVENUE_URL=https://... bash scripts/download_avenue.sh"
  echo "Find the current link at:"
  echo "  http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html"
  echo "  or the 'Datasets' section of https://github.com/StevenLiuWen/ano_pred_cvpr2018"
  exit 1
fi

cd "$RAW_DIR"
curl -L -o avenue_dataset.zip "$DOWNLOAD_URL"
unzip -q avenue_dataset.zip

echo "Next step (after checking the extracted folder names match):"
echo "  python scripts/prepare_dataset.py --dataset avenue \\"
echo "      --raw-root $RAW_DIR/<extracted-folder> \\"
echo "      --out data/Avenue --dry-run"
