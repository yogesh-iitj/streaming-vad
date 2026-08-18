#!/usr/bin/env bash
# Download the UCSD Anomaly Detection Dataset (contains both Ped1 and Ped2).
# Source: UCSD Statistical Visual Computing Lab (official, stable for years).
#   http://www.svcl.ucsd.edu/projects/anomaly/dataset.html
# Approx size: ~290 MB compressed (tar.gz), both Ped1+Ped2.
#
# NOT executed automatically by this codebase — run manually after
# confirming you have the rights/bandwidth to do so.
set -euo pipefail

RAW_DIR="${1:-data/raw/UCSD}"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

URL="http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
echo "Downloading UCSD Anomaly Dataset from: $URL"
echo "If this URL 404s, check http://www.svcl.ucsd.edu/projects/anomaly/dataset.html for the current link."
curl -L -o UCSD_Anomaly_Dataset.tar.gz "$URL"

echo "Extracting..."
tar -xzf UCSD_Anomaly_Dataset.tar.gz

echo "Done. Ped2 raw data should now be under:"
echo "  $RAW_DIR/UCSD_Anomaly_Dataset.v1p2/UCSDped2/"
echo ""
echo "Next step:"
echo "  python scripts/prepare_dataset.py --dataset ped2 \\"
echo "      --raw-root $RAW_DIR/UCSD_Anomaly_Dataset.v1p2/UCSDped2 \\"
echo "      --out data/UCSDped2 --dry-run   # inspect first, then drop --dry-run"
