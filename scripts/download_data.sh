#!/usr/bin/env bash
# Pulls IEEE-CIS Fraud Detection (competition) and PaySim (open dataset) into data/raw/.
# Requires: ~/.kaggle/kaggle.json API token, and the IEEE-CIS competition rules
# accepted at https://www.kaggle.com/c/ieee-fraud-detection/rules (Kaggle blocks
# API downloads for competitions until you've clicked "I Understand and Accept" once).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
KAGGLE="$REPO_ROOT/.venv/Scripts/kaggle.exe"

mkdir -p "$RAW_DIR/ieee-cis" "$RAW_DIR/paysim"

echo "==> Downloading IEEE-CIS Fraud Detection..."
"$KAGGLE" competitions download -c ieee-fraud-detection -p "$RAW_DIR/ieee-cis"
unzip -o "$RAW_DIR/ieee-cis/ieee-fraud-detection.zip" -d "$RAW_DIR/ieee-cis"
rm "$RAW_DIR/ieee-cis/ieee-fraud-detection.zip"

echo "==> Downloading PaySim..."
"$KAGGLE" datasets download -d ealaxi/paysim1 -p "$RAW_DIR/paysim"
unzip -o "$RAW_DIR/paysim/paysim1.zip" -d "$RAW_DIR/paysim"
rm "$RAW_DIR/paysim/paysim1.zip"

echo "==> Done. Contents of data/raw:"
find "$RAW_DIR" -maxdepth 2 -type f
