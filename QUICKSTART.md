# Quick Start Guide

Get up and running with VLM evaluation in 5 minutes!

## 1. Setup Environment (2 minutes)

```bash
cd MLLM_eval
bash setup_env.sh
conda activate vlbench
```

This installs all dependencies including PyTorch, Transformers, and evaluation metrics.

## 2. Test Installation (30 seconds)

```bash
python test_installation.py
```

You should see all tests pass. If not, check the error messages.

## 3. Run Your First Evaluation (2 minutes)

Let's evaluate Qwen2.5-VL on CV-Bench with just 10 samples to make sure everything works:

```bash
python eval_mcq.py \
    --model qwen \
    --benchmark cvbench \
    --max_samples 10 \
    --output results/test.json
```

**Expected output:**
```
Loading model: Qwen/Qwen2.5-VL-7B-Instruct
Model loaded successfully!
Loading CV-Bench (test split)...
Loaded 1000 samples
Starting evaluation
Evaluating: 100%|████████████| 10/10 [00:45<00:00]
Results Summary
Total samples: 10
Correct: 7
Accuracy: 0.7000 (70.00%)
```

## 4. Run Full Benchmark (varies)

Once the test works, run a complete benchmark:

```bash
# Single benchmark (5-15 minutes depending on size)
python eval_mcq.py --model qwen --benchmark cvbench --output results/cvbench.json

# All benchmarks (30-60 minutes)
bash run_all.sh
```

## 5. View Results

Results are saved as JSON and CSV files:

```bash
# View summary
cat results/cvbench.json | grep accuracy

# View per-sample results
head results/cvbench.csv
```

## Common Issues & Solutions

### Issue: "CUDA out of memory"

**Solution 1:** Use smaller model
```bash
python eval_mcq.py --model qwen --model_id "Qwen/Qwen2.5-VL-2B-Instruct" --benchmark cvbench
```

**Solution 2:** Run on CPU (slower)
```bash
export CUDA_VISIBLE_DEVICES=""
python eval_mcq.py --model qwen --benchmark cvbench
```

### Issue: "Dataset download too slow"

**Solution:** Set cache to a location with more space
```bash
export HF_HOME=/path/to/large/disk/.cache
python eval_mcq.py --model qwen --benchmark cvbench
```

### Issue: "conda: command not found"

**Solution:** Install Miniconda first
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## Next Steps

### Evaluate on More Benchmarks

```bash
# 3D Spatial Reasoning
python eval_mcq.py --model qwen --benchmark 3dsr --output results/3dsr.json

# Multi-Image Spatial Intelligence
python eval_mcq.py --model qwen --benchmark mmsi --output results/mmsi.json

# BLINK (Perception)
python eval_mcq.py --model qwen --benchmark blink --output results/blink.json
```

### Add Your Own Model

1. Create `models/your_model.py` following the template in `models/example_vlm.py`
2. Implement `ask_images()` and `ask_video()` methods
3. Run evaluation:
```bash
python eval_mcq.py --model your_model --benchmark cvbench
```

See `examples.py` for detailed code examples.

### Generate Tables

```bash
python tables/format_tables.py \
    --mode mcq \
    --input results/cvbench.json \
    --model_name "Qwen2.5-VL-7B"
```

Output:
```
| Model           | CV-Bench Accuracy (%) |
|-----------------|-----------------------|
| Qwen2.5-VL-7B   | 75.20                 |
```

## Benchmark-Specific Notes

### CV-Bench
- **Auto-downloads**: Yes
- **Size**: ~1000 samples
- **Time**: ~10-15 minutes
- **Note**: Tests both 2D and 3D visual reasoning

### 3DSRBench
- **Auto-downloads**: Yes
- **Size**: Varies by split
- **Time**: ~10-20 minutes
- **Note**: May require fetching images from URLs (slower)

### MMSI-Bench
- **Auto-downloads**: Yes
- **Size**: Varies
- **Time**: ~15-25 minutes
- **Note**: Multiple images per question (slower inference)

### BLINK
- **Auto-downloads**: Yes
- **Size**: Varies by subtask
- **Time**: ~10-20 minutes per subtask
- **Note**: Can evaluate specific subtasks or all at once

### ScanQA
- **Auto-downloads**: No (manual setup required)
- **Size**: 800 val samples
- **Time**: ~30-45 minutes
- **Note**: Requires scene images or videos

#### ScanQA Quick Setup
```bash
# Download data
git clone https://github.com/ATR-DBI/ScanQA.git data/scanqa
cd data/scanqa
# Follow their README to download ScanQA_v1.0 annotations

# Run evaluation (without visuals for now)
cd ../..
python eval_scanqa.py --data_root data/scanqa --output results/scanqa

# Compute metrics
cd data/scanqa
python scripts/score.py --folder ../../results/scanqa
```

## Useful Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage during evaluation
watch -n 1 nvidia-smi

# Check conda environment
conda info --envs
conda list | grep torch

# Test model loading
python -c "from models.qwen_vl import QwenVL; model = QwenVL(); print('OK')"

# View dataset info
python -c "from datasets.cvbench import CVBenchDataset; ds = CVBenchDataset(); print(f'Samples: {len(ds)}')"
```

## Performance Tips

1. **Use GPU**: 10-50x faster than CPU
2. **Batch evaluation**: Use `run_all.sh` to evaluate multiple benchmarks sequentially
3. **Sample limiting**: Use `--max_samples` for quick tests
4. **Model size**: Smaller models are faster but less accurate

## Getting Help

- Read the full `README.md` for comprehensive documentation
- Check `examples.py` for code samples
- Open an issue on GitHub for bugs
- See original benchmark papers for dataset-specific questions

## Benchmarking Best Practices

1. **Always report model size**: e.g., "Qwen2.5-VL-7B" not just "Qwen"
2. **Use deterministic settings**: We disable sampling by default
3. **Report all metrics**: Don't cherry-pick benchmarks
4. **Use official splits**: test/val as specified
5. **Compare fairly**: Same model size, same hardware when possible

## Ready to Start?

```bash
# Complete workflow
conda activate vlbench
python test_installation.py
bash run_all.sh

# Wait for results...
# View outputs in results/
```

Happy evaluating! 🚀
