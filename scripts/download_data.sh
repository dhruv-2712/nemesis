#!/usr/bin/env bash
# Pulls all NEMESIS datasets into data/raw/.
#
# Datasets:
#   Elliptic Bitcoin (primary)  — open, no rules acceptance needed
#   IEEE-CIS Fraud Detection    — requires accepting competition rules once at
#                                 https://www.kaggle.com/c/ieee-fraud-detection/rules
#   PaySim                      — open synthetic dataset
#
# Requires: ~/.kaggle/kaggle.json API token
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
KAGGLE="$REPO_ROOT/.venv/Scripts/kaggle.exe"

mkdir -p "$RAW_DIR/elliptic" "$RAW_DIR/ieee-cis" "$RAW_DIR/paysim"

echo "==> Downloading Elliptic Bitcoin Dataset (primary)..."
"$KAGGLE" datasets download -d ellipticco/elliptic-data-set -p "$RAW_DIR/elliptic" --unzip

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
