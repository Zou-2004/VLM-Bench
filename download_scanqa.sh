#!/bin/bash
# Auto-download and setup ScanQA dataset

set -e

echo "=========================================="
echo "Downloading ScanQA Dataset..."
echo "=========================================="

# Download ScanQA folder from Google Drive
gdown --folder "https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2" -O data/scanqa --remaining-ok

echo ""
echo "=========================================="
echo "ScanQA Dataset Downloaded!"
echo "=========================================="
echo ""
ls -lh data/scanqa/
echo ""
echo "Ready to evaluate:"
echo "python eval_scanqa.py --model qwen --data_root data/scanqa --output results/scanqa"
echo ""
