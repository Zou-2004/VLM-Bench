#!/bin/bash
# Master evaluation script to run all benchmarks

set -e

# Configuration
MODEL_ID=${MODEL_ID:-"Qwen/Qwen2.5-VL-7B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"results"}
MAX_SAMPLES=${MAX_SAMPLES:-""}  # Empty = all samples

echo "======================================"
echo "VLM Benchmark Evaluation Suite"
echo "======================================"
echo "Model: ${MODEL_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo "======================================"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "vlbench" ]]; then
    echo "Warning: vlbench environment not activated"
    echo "Attempting to activate..."
    eval "$(conda shell.bash hook)"
    conda activate vlbench || {
        echo "Error: Failed to activate vlbench environment"
        echo "Please run: conda activate vlbench"
        exit 1
    }
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Function to run evaluation and handle errors
run_eval() {
    local benchmark=$1
    local script=$2
    local extra_args=$3
    
    echo ""
    echo "======================================"
    echo "Evaluating: ${benchmark}"
    echo "======================================"
    
    if python ${script} ${extra_args}; then
        echo "✓ ${benchmark} completed successfully"
    else
        echo "✗ ${benchmark} failed"
        return 1
    fi
}

# Optional: run specific benchmark only
if [ ! -z "$1" ]; then
    case $1 in
        cvbench)
            run_eval "CV-Bench" "eval_mcq.py" \
                "--model qwen --model_id ${MODEL_ID} --benchmark cvbench --output ${OUTPUT_DIR}/cvbench.json ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}"
            ;;
        3dsr)
            run_eval "3DSRBench" "eval_mcq.py" \
                "--model qwen --model_id ${MODEL_ID} --benchmark 3dsr --output ${OUTPUT_DIR}/3dsr.json ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}"
            ;;
        mmsi)
            run_eval "MMSI-Bench" "eval_mcq.py" \
                "--model qwen --model_id ${MODEL_ID} --benchmark mmsi --output ${OUTPUT_DIR}/mmsi.json ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}"
            ;;
        blink)
            run_eval "BLINK" "eval_mcq.py" \
                "--model qwen --model_id ${MODEL_ID} --benchmark blink --output ${OUTPUT_DIR}/blink.json ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}"
            ;;
        scanqa)
            run_eval "ScanQA" "eval_scanqa.py" \
                "--model qwen --model_id ${MODEL_ID} --output ${OUTPUT_DIR}/scanqa ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}"
            ;;
        *)
            echo "Unknown benchmark: $1"
            echo "Available: cvbench, 3dsr, mmsi, blink, scanqa"
            exit 1
            ;;
    esac
    exit 0
fi

# Run all MCQ benchmarks
echo ""
echo "======================================"
echo "Running MCQ Benchmarks"
echo "======================================"

# CV-Bench
run_eval "CV-Bench" "eval_mcq.py" \
    "--model qwen --model_id ${MODEL_ID} --benchmark cvbench --output ${OUTPUT_DIR}/cvbench.json ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}" \
    || true

# 3DSRBench
run_eval "3DSRBench" "eval_mcq.py" \
    "--model qwen --model_id ${MODEL_ID} --benchmark 3dsr --output ${OUTPUT_DIR}/3dsr.json ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}" \
    || true

# MMSI-Bench
run_eval "MMSI-Bench" "eval_mcq.py" \
    "--model qwen --model_id ${MODEL_ID} --benchmark mmsi --output ${OUTPUT_DIR}/mmsi.json ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}" \
    || true

# BLINK
run_eval "BLINK" "eval_mcq.py" \
    "--model qwen --model_id ${MODEL_ID} --benchmark blink --output ${OUTPUT_DIR}/blink.json ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}" \
    || true

# ScanQA (if data is available)
if [ -d "data/scanqa" ]; then
    echo ""
    echo "======================================"
    echo "Running ScanQA"
    echo "======================================"
    
    run_eval "ScanQA" "eval_scanqa.py" \
        "--model qwen --model_id ${MODEL_ID} --output ${OUTPUT_DIR}/scanqa ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}" \
        || true
else
    echo ""
    echo "Skipping ScanQA (data not found at data/scanqa)"
fi

# Print summary
echo ""
echo "======================================"
echo "Evaluation Complete!"
echo "======================================"
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "To view results:"
echo "  - MCQ benchmarks: see JSON/CSV files in ${OUTPUT_DIR}/"
echo "  - ScanQA: run scoring with 'cd data/scanqa && python scripts/score.py --folder ../../${OUTPUT_DIR}/scanqa'"
echo ""
