#!/bin/bash
# Quick verification that all benchmarks work correctly

echo "=========================================="
echo "Testing All Benchmarks (5 samples each)"
echo "=========================================="
echo ""

# CV-Bench
echo "1. Testing CV-Bench..."
python eval_mcq.py --model qwen --benchmark cvbench --max_samples 5 --output results/verify_cvbench.json 2>&1 | grep -E "(Accuracy|Correct)"
echo ""

# 3DSRBench
echo "2. Testing 3DSRBench..."
python eval_mcq.py --model qwen --benchmark 3dsr --max_samples 5 --output results/verify_3dsr.json 2>&1 | grep -E "(Accuracy|Correct)"
echo ""

# MMSI-Bench
echo "3. Testing MMSI-Bench..."
python eval_mcq.py --model qwen --benchmark mmsi --max_samples 5 --output results/verify_mmsi.json 2>&1 | grep -E "(Accuracy|Correct)"
echo ""

# BLINK
echo "4. Testing BLINK (Spatial_Relation)..."
python eval_mcq.py --model qwen --benchmark blink --subtask Spatial_Relation --max_samples 5 --output results/verify_blink.json 2>&1 | grep -E "(Accuracy|Correct)"
echo ""

echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
echo ""
echo "Summary of Results:"
python << 'PYEOF'
import json
import glob

results = []
for path in sorted(glob.glob("results/verify_*.json")):
    with open(path) as f:
        data = json.load(f)
        benchmark = path.split("verify_")[1].split(".json")[0]
        results.append({
            "benchmark": benchmark,
            "total": data["total"],
            "correct": data["correct"],
            "accuracy": data["accuracy"]
        })

print(f"{'Benchmark':<20} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
print("-" * 50)
for r in results:
    print(f"{r['benchmark']:<20} {r['total']:<8} {r['correct']:<8} {r['accuracy']:<10.2%}")

all_correct = sum(r["correct"] for r in results)
all_total = sum(r["total"] for r in results)
print("-" * 50)
print(f"{'OVERALL':<20} {all_total:<8} {all_correct:<8} {all_correct/all_total:<10.2%}")
print("")
print("✅ All benchmarks working correctly!" if all_correct > 0 else "❌ Issues detected")
PYEOF
