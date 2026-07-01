#!/usr/bin/env bash
# =============================================================================
#  Deep SLDA – Single-command execution script
#  Usage:   bash run.sh [config.yaml]
# =============================================================================
set -euo pipefail

CONFIG="${1:-config.yaml}"

echo "=========================================="
echo " Deep SLDA – CVPRW 2020"
echo " Config: ${CONFIG}"
echo "=========================================="

# --- Install dependencies (skip if already installed) ---
pip install -q -r requirements.txt

# --- Run training & evaluation ---
python train.py --config "${CONFIG}"

echo ""
echo "Done.  Results saved to the 'save_dir' specified in ${CONFIG}."
