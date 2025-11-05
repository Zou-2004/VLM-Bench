# VLM Evaluation Framework

**Evaluate ANY Vision-Language Model on 6 major benchmarks with minimal code.**

## ✨ Key Feature: Works with YOUR Custom Model

This framework is **model-agnostic**. You only need to implement **ONE method** to evaluate your custom VLM (like VLM3R):

```python
def ask_images(self, images: List[Image], prompt: str, max_new_tokens: int) -> str:
    # Your model inference code here
    return response
```

That's it! See **`HOW_TO_ADD_YOUR_MODEL.py`** for complete instructions.

---

## 📊 Supported Benchmarks

| Benchmark | Type | Focus | Samples | Auto-Download |
|-----------|------|-------|---------|---------------|
| **CV-Bench** | MCQ | 2D/3D vision | ~1000 | ✅ |
| **3DSRBench** | MCQ | 3D spatial reasoning | ~800 | ✅ |
| **MMSI-Bench** | MCQ | Multi-image spatial | ~600 | ✅ |
| **BLINK** | MCQ | Perception (14 tasks) | ~1400 | ✅ |
| **ScanQA** | Open-ended | 3D scene QA | 800 | ✅ (Q&A only) * |
| **SQA3D** | Open-ended | Situated 3D QA | 3200+ | ✅ (Q&A only) * |

\* ScanQA and SQA3D require scene images/videos from ScanNet (see [3D Scene Setup](#3d-scene-setup))

---

## 🚀 Quick Start

### 1. Setup (2 minutes)

```bash
bash setup_env.sh
conda activate vlbench
python test_installation.py
```

### 2. Evaluate Qwen (Example)

```bash
python eval_mcq.py --model qwen --benchmark cvbench --output results/cvbench.json
```

### 3. Add YOUR Custom Model

See **`HOW_TO_ADD_YOUR_MODEL.py`** - it's just ~50 lines of code!

**Quick summary:**
1. Create `models/your_model.py`
2. Inherit from `BaseVLM`
3. Implement `ask_images(images, prompt, max_new_tokens) -> str`
4. Run: `python eval_mcq.py --model your-model --benchmark cvbench`

---

## 📁 Project Structure

```
MLLM_eval/
├── README.md                    ← You are here
├── QUICKSTART.md                ← 5-minute tutorial
├── HOW_TO_ADD_YOUR_MODEL.py     ← ⭐ Template for YOUR model
│
├── eval_mcq.py                  ← Run MCQ benchmarks
├── eval_scanqa.py               ← Run ScanQA
├── eval_sqa3d.py                ← Run SQA3D
├── metrics.py                   ← Evaluation metrics
│
├── models/
│   ├── base_vlm.py              ← Abstract interface (what you inherit)
│   └── qwen_vl.py               ← Example: Qwen2.5-VL
│
├── benchmark_loaders/           ← Auto-loading for all benchmarks
│   ├── cvbench.py
│   ├── three_dsr.py
│   ├── mmsi.py
│   ├── blink.py
│   ├── scanqa.py
│   └── sqa3d.py
│
├── tables/
│   └── format_tables.py         ← Generate publication tables
│
├── setup_env.sh                 ← One-time setup
├── run_all.sh                   ← Evaluate all benchmarks
└── test_installation.py         ← Verify setup
```

---

## 🤖 How to Evaluate YOUR Model

### Step 1: Implement Your Model Wrapper

```python
# models/your_vlm.py
from models.base_vlm import BaseVLM
from PIL import Image
from typing import List
import torch

class YourVLM(BaseVLM):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        # Load YOUR model checkpoint
        self.model = YourModelClass.load(model_id)
        self.model.eval()
    
    def ask_images(self, images: List[Image.Image], prompt: str, 
                   max_new_tokens: int = 64) -> str:
        """
        This is ALL you need to implement!
        
        Args:
            images: List of PIL Images (single or multiple)
            prompt: Text question
            max_new_tokens: Max generation length
            
        Returns:
            Text response from your model
        """
        # 1. Preprocess images (your way)
        image_tensors = self.preprocess_images(images)
        
        # 2. Encode text (your way)
        text_tokens = self.tokenizer(prompt)
        
        # 3. Run inference (your way)
        with torch.no_grad():
            output = self.model(image_tensors, text_tokens, max_new_tokens)
        
        # 4. Decode and return
        response = self.tokenizer.decode(output)
        return response.strip()
    
    def ask_video(self, video_path: str, prompt: str, **kwargs) -> str:
        # Optional: sample frames and call ask_images
        frames = self.sample_video_frames(video_path)
        return self.ask_images(frames, prompt, **kwargs)
```

### Step 2: Register in Evaluation Script

```python
# Add to eval_mcq.py, line ~200
elif args.model == "your-model":
    from models.your_vlm import YourVLM
    model = YourVLM(model_id=args.model_id)
```

### Step 3: Evaluate!

```bash
# Single benchmark
python eval_mcq.py \
    --model your-model \
    --model_id path/to/your/checkpoint.pth \
    --benchmark cvbench \
    --output results/your_model.json

# All benchmarks
bash run_all.sh  # (after updating with your model)
```

---

## 📈 Example Results

After evaluation, you get:

```json
{
  "total": 1000,
  "correct": 752,
  "accuracy": 0.752,
  "results": [
    {
      "question": "What color is the car?",
      "prediction": "B",
      "ground_truth": "B",
      "correct": true
    },
    ...
  ]
}
```

Format as publication table:

```bash
python tables/format_tables.py --mode mcq --input results/your_model.json
```

Output:
```
| Model      | CV-Bench Accuracy |
|------------|-------------------|
| Your Model | 75.20%            |
```

---

## 🎯 Why This Framework?

### For YOUR Custom Model:
✅ **Minimal integration** - Just implement `ask_images()`  
✅ **No benchmark code needed** - All datasets auto-load  
✅ **Consistent evaluation** - Same protocol as published models  
✅ **Multiple modalities** - Single/multi-image, video support  

### Features:
✅ **6 major benchmarks** (CV-Bench, 3DSRBench, MMSI, BLINK, ScanQA, SQA3D)  
✅ **Deterministic** - Reproducible results  
✅ **Auto-download** - 4/6 benchmarks fetch automatically  
✅ **Production-ready** - Error handling, progress bars, logging  
✅ **Well-documented** - Clear examples for custom models  

---

## 📚 Documentation

1. **`README.md`** (this file) - Overview + quick start
2. **`QUICKSTART.md`** - 5-minute tutorial with examples
3. **`HOW_TO_ADD_YOUR_MODEL.py`** - Complete template for custom models
4. **`SCANNET_SETUP.md`** - Detailed guide for ScanNet data setup (ScanQA + SQA3D)
5. **`BUGFIXES.md`** - Known issues and recent bug fixes

---

## 💡 Common Questions

**Q: Will this work with my VLM3R-like model with different architecture?**  
A: **YES!** As long as you can implement `ask_images(images, prompt) -> response`, it works. The framework doesn't care about your model internals.

**Q: What if my model only supports single images?**  
A: Just use `images[0]` and set `supports_multi_image = False`. The framework handles it.

**Q: What about video?**  
A: Either implement native video support, or use the default frame sampling (shown in template).

**Q: Can I use my own checkpoint format?**  
A: Yes! The `model_id` can be a local path. Load your checkpoint however you want in `__init__`.

---

## 🔧 Advanced Usage

### Using Different Model Sizes

```bash
# Use 2B model
python eval_mcq.py --model qwen --model_id "Qwen/Qwen2.5-VL-2B-Instruct" --benchmark cvbench

# Use 72B model (requires multi-GPU)
python eval_mcq.py --model qwen --model_id "Qwen/Qwen2.5-VL-72B-Instruct" --benchmark cvbench
```

### Troubleshooting

**CUDA Out of Memory:**
- Use smaller model or enable 8-bit quantization
- Reduce `max_frames` for video benchmarks

**Dataset Download Issues:**
```bash
export HF_HOME=/path/to/large/disk  # Set cache directory
export HF_ENDPOINT=https://hf-mirror.com  # Use mirror if needed
```

**BLINK Test Split Warning:**
- BLINK's test split has hidden answers for leaderboard submission
- The framework automatically uses `val` split for evaluation
- To override: use `--split test` (but accuracy will show 0%)

**Zero Accuracy / All Wrong:**
- Make sure you're using the correct dataset split (val for BLINK, not test)
- Check that model is actually running inference (look for responses in output)
- See `BUGFIXES.md` for recently resolved issues

#### 3D Scene Setup (ScanQA + SQA3D)

Both **ScanQA** and **SQA3D** use the same **ScanNet v2** scenes. Download ScanNet data **once** and both benchmarks can use it!

> 📘 **Detailed guide:** See [SCANNET_SETUP.md](SCANNET_SETUP.md) for complete instructions on downloading and rendering ScanNet data.

##### Step 1: Prepare ScanNet Data (Shared)

**Download ScanNet scenes** (one-time setup for both benchmarks):

1. **Get ScanNet access:**
   - Apply at [ScanNet website](http://www.scan-net.org/)
   - Requires agreeing to terms of use
   - Download the ScanNet v2 dataset

2. **Render scene visuals** using [ScanQA rendering scripts](https://github.com/ATR-DBI/ScanQA):
   
   **Choose ONE format:**
   
   - **Option A - Videos** (recommended for efficiency):
     ```
     scannet_data/
     └── videos/
         ├── scene0000_00.mp4
         ├── scene0001_00.mp4
         ├── scene0002_00.mp4
         └── ...
     ```
   
   - **Option B - Multi-view images** (better quality):
     ```
     scannet_data/
     └── images/
         ├── scene0000_00/
         │   ├── 0.jpg
         │   ├── 1.jpg
         │   └── ...
         ├── scene0001_00/
         └── ...
     ```

##### Step 2: Download Q&A Data

```bash
# ScanQA Q&A data (automatic)
bash download_scanqa.sh

# SQA3D Q&A data (manual)
# Download from: https://github.com/SilongYong/SQA3D
# Extract to: SQA3D/
```

##### Step 3: Run Evaluations

**ScanQA evaluation:**
```bash
# With videos (Option A)
python eval_scanqa.py --model qwen \
    --data_root data/scanqa \
    --scene_videos_root scannet_data/videos \
    --output results/scanqa

# With images (Option B)
python eval_scanqa.py --model qwen \
    --data_root data/scanqa \
    --scene_images_root scannet_data/images \
    --output results/scanqa

# Compute metrics
cd data/scanqa && python scripts/score.py --folder ../../results/scanqa
```

**SQA3D evaluation:**
```bash
# With videos (Option A)
python eval_sqa3d.py --model qwen \
    --data_root SQA3D \
    --scene_videos_root scannet_data/videos \
    --split val \
    --output results/sqa3d

# With images (Option B)
python eval_sqa3d.py --model qwen \
    --data_root SQA3D \
    --scene_images_root scannet_data/images \
    --split val \
    --output results/sqa3d

# Text-only mode (testing without visuals)
python eval_sqa3d.py --model qwen \
    --data_root SQA3D \
    --split val \
    --max_samples 10
```

**Summary:** 
- ✅ Download ScanNet **once** → `scannet_data/videos/` or `scannet_data/images/`
- ✅ Both ScanQA and SQA3D use the same `--scene_videos_root` or `--scene_images_root`
- ✅ No duplication needed!

## Adding New Models

The framework is designed to support any VLM. Here's how to add a new model:

### 1. Create a Model Wrapper

Create a new file `models/your_model.py`:

```python
from models.base_vlm import BaseVLM
from PIL import Image
from typing import List

class YourModel(BaseVLM):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        # Load your model
        self.model = load_your_model(model_id)
    
    def ask_images(self, images: List[Image.Image], prompt: str, max_new_tokens: int = 64) -> str:
        # Implement inference for images
        return self.model.generate(images, prompt, max_new_tokens)
    
    def ask_video(self, video_path: str, prompt: str, fps: float = 2.0, 
                  max_frames: int = 32, max_new_tokens: int = 64) -> str:
        # Implement inference for video
        frames = extract_frames(video_path, max_frames)
        return self.ask_images(frames, prompt, max_new_tokens)
```

### 2. Update Evaluation Scripts

In `eval_mcq.py` and `eval_scanqa.py`, add your model:

```python
if args.model == "qwen":
    model = QwenVL(model_id=args.model_id)
elif args.model == "your_model":
    from models.your_model import YourModel
    model = YourModel(model_id=args.model_id)
```

### 3. Run Evaluation

```bash
python eval_mcq.py --model your_model --model_id your/model/id --benchmark cvbench --output results/yourmodel_cvbench.json
```

## Configuration

### Environment Variables

```bash
# Model selection
export MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"

# Output directory
export OUTPUT_DIR="results"

# Limit samples for testing
export MAX_SAMPLES="100"
```

### Command-Line Arguments

All scripts support extensive configuration:

```bash
python eval_mcq.py --help
python eval_scanqa.py --help
```

Key arguments:
- `--model`: Model type (qwen, ...)
- `--model_id`: Specific model ID (e.g., Hugging Face model)
- `--benchmark`: Benchmark name
- `--split`: Dataset split (train/val/test)
- `--max_samples`: Limit evaluation samples
- `--output`: Output path

## Advanced Usage

### Using Different Model Sizes

```bash
# Use 2B model
python eval_mcq.py --model qwen --model_id "Qwen/Qwen2.5-VL-2B-Instruct" --benchmark cvbench

# Use 72B model (requires multi-GPU)
python eval_mcq.py --model qwen --model_id "Qwen/Qwen2.5-VL-72B-Instruct" --benchmark cvbench
```

### 8-bit Quantization (Lower Memory)

Modify `models/qwen_vl.py`:

```python
model = QwenVL(model_id="Qwen/Qwen2.5-VL-7B-Instruct", load_in_8bit=True)
```

### Batch Processing with vLLM

For faster throughput, use vLLM:

```bash
# Install vLLM
pip install "vllm>=0.6.3"

# Serve model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code --tensor-parallel-size 1
```

Then modify the evaluation scripts to call the vLLM server endpoint.

## Metrics and Scoring

### MCQ Benchmarks

- **Metric**: Accuracy (% correct)
- **Computation**: Exact match of predicted choice letter to ground truth
- **Output**: JSON with per-sample results + CSV summary

### ScanQA

- **Metrics**: BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr
- **Computation**: Uses official ScanQA scoring script
- **Output**: JSON predictions + metric scores

### Table Formatting

Generate publication-ready tables:

```python
from tables.format_tables import format_scanqa_table

results = {
    "Qwen2.5-VL-7B": {
        "BLEU-1": 25.3,
        "BLEU-4": 12.1,
        "METEOR": 18.4,
        "ROUGE-L": 32.1,
        "CIDEr": 45.6
    }
}

print(format_scanqa_table(results, tablefmt="github"))
```

Output:
```
| Methods       | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---------------|--------|--------|--------|---------|-------|
| Qwen2.5-VL-7B | 25.3   | 12.1   | 18.4   | 32.1    | 45.6  |
```

## Troubleshooting

### CUDA Out of Memory

1. Use smaller model: `Qwen2.5-VL-2B-Instruct`
2. Enable 8-bit quantization: `load_in_8bit=True`
3. Reduce batch size or max frames for video

### Dataset Download Issues

If Hugging Face datasets fail to download:

```bash
# Set cache directory
export HF_HOME=/path/to/large/disk

# Use mirror (if in China)
export HF_ENDPOINT=https://hf-mirror.com
```

### ScanQA Scoring Issues

Ensure you have the official ScanQA repo:

```bash
cd data/scanqa
git pull  # Update to latest version
pip install -r requirements.txt
```

---

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 24GB+ VRAM (for 7B models, less for smaller)
- 50GB disk space

Full setup is automated via `setup_env.sh`.

---

## 🎓 Citation

If you use this framework, please cite the original benchmark papers:

```bibtex
# CV-Bench
@article{tong2024cambrian,
  title={Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs},
  author={Tong, Shengbang and others},
  journal={arXiv preprint arXiv:2406.16860},
  year={2024}
}

# 3DSRBench
@article{huang20243dsrbench,
  title={3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark},
  author={Huang, Xiaojia and others},
  year={2024}
}

# MMSI-Bench
@article{xu2025mmsi,
  title={MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence},
  author={Xu, Runsen and others},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}

# BLINK
@article{fu2024blink,
  title={BLINK: Multimodal Large Language Models Can See but Not Perceive},
  author={Fu, Xingyu and others},
  journal={arXiv preprint arXiv:2404.12390},
  year={2024}
}

# ScanQA
@inproceedings{azuma2022scanqa,
  title={ScanQA: 3D Question Answering for Spatial Scene Understanding},
  author={Azuma, Daichi and others},
  booktitle={CVPR},
  year={2022}
}
```

---

## 📄 License

MIT License - See individual benchmarks for their specific licenses.

---

**Ready to evaluate your model? Start with `HOW_TO_ADD_YOUR_MODEL.py`! 🚀**

## License

This framework code is released under MIT License. Individual benchmarks and models may have their own licenses - please check their respective repositories.

## Contributing

Contributions are welcome! To add a new benchmark or model:

1. Fork the repository
2. Create a feature branch
3. Add your implementation following the existing patterns
4. Submit a pull request

## Acknowledgments

- Qwen team for Qwen2.5-VL
- All benchmark creators for open-sourcing their datasets
- Hugging Face for dataset hosting
- PyTorch and transformers communities

## Contact

For issues or questions, please open an issue on GitHub or contact the maintainers.

---

**Happy Evaluating! 🚀**
