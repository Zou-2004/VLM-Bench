#!/bin/bash
# Test all benchmarks with small samples

set -e

echo "Testing all benchmarks with 3 samples each..."
echo ""

# CV-Bench
echo "========================================="
echo "1. Testing CV-Bench..."
echo "========================================="
python eval_mcq.py --model qwen --benchmark cvbench --max_samples 3 --output results/test_cvbench.json
echo ""

# 3DSRBench
echo "========================================="
echo "2. Testing 3DSRBench..."
echo "========================================="
python eval_mcq.py --model qwen --benchmark 3dsr --max_samples 3 --output results/test_3dsr.json
echo ""

# MMSI-Bench
echo "========================================="
echo "3. Testing MMSI-Bench..."
echo "========================================="
python eval_mcq.py --model qwen --benchmark mmsi --max_samples 3 --output results/test_mmsi.json
echo ""

# BLINK (single subtask)
echo "========================================="
echo "4. Testing BLINK (Spatial_Relation)..."
echo "========================================="
python eval_mcq.py --model qwen --benchmark blink --subtask Spatial_Relation --max_samples 3 --output results/test_blink.json
echo ""

echo "========================================="
echo "All tests completed successfully!"
echo "========================================="
echo ""
echo "Results saved in results/ directory:"
ls -lh results/test_*.json
echo ""
echo "Statistics files:"
ls -lh results/test_*_stats.json
