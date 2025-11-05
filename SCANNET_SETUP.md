# ScanNet Data Setup Guide

This guide explains how to set up ScanNet data **once** for both **ScanQA** and **SQA3D** benchmarks.

## Overview

Both ScanQA and SQA3D use the same **ScanNet v2** 3D scene dataset. You only need to download and render ScanNet scenes **once** in a shared location.

## Directory Structure

```
MLLM_eval/
├── scannet_data/           ← Shared ScanNet data (one-time setup)
│   ├── videos/             ← Option A: Video format
│   │   ├── scene0000_00.mp4
│   │   ├── scene0001_00.mp4
│   │   └── ...
│   └── images/             ← Option B: Multi-view images
│       ├── scene0000_00/
│       │   ├── 0.jpg
│       │   ├── 1.jpg
│       │   └── ...
│       └── ...
│
├── data/scanqa/            ← ScanQA Q&A data only
│   └── ScanQA_v1.0/
│
└── SQA3D/                  ← SQA3D Q&A data only
    └── sqa_task/
```

## Step-by-Step Setup

### 1. Get ScanNet Access

1. Go to [ScanNet website](http://www.scan-net.org/)
2. Fill out the **Terms of Use Agreement Form**
3. Wait for approval email (usually 1-2 days)
4. Download the **ScanNet v2** dataset using their provided script

### 2. Download ScanNet Scenes

After getting access:

```bash
# Download ScanNet v2 dataset (requires credentials from step 1)
python download-scannet.py -o scannet_raw --type .sens

# This will download .sens files (sensor data) for all scenes
```

### 3. Render Scene Visuals

You need to convert raw ScanNet .sens files to videos or images.

#### Option A: Render Videos (Recommended)

Videos are more efficient for storage and loading:

```bash
# Clone ScanQA repo for rendering scripts
git clone https://github.com/ATR-DBI/ScanQA.git
cd ScanQA

# Install dependencies
pip install -r requirements.txt

# Render videos from .sens files
python scripts/render_scannet_videos.py \
    --scannet_dir ../scannet_raw \
    --output_dir ../scannet_data/videos \
    --fps 30 \
    --resolution 640x480

# This creates: scannet_data/videos/scene{id}.mp4
```

#### Option B: Render Multi-view Images (Higher Quality)

Images provide better quality but take more disk space:

```bash
# Render images from .sens files
python scripts/render_scannet_images.py \
    --scannet_dir ../scannet_raw \
    --output_dir ../scannet_data/images \
    --num_views 10 \
    --resolution 640x480

# This creates: scannet_data/images/scene{id}/0.jpg, 1.jpg, ...
```

### 4. Verify Setup

Check that you have the correct structure:

```bash
# For videos
ls scannet_data/videos/ | head -5
# Should show: scene0000_00.mp4, scene0001_00.mp4, ...

# For images
ls scannet_data/images/ | head -5
# Should show: scene0000_00/, scene0001_00/, ...

ls scannet_data/images/scene0000_00/
# Should show: 0.jpg, 1.jpg, 2.jpg, ...
```

## Using with Benchmarks

### ScanQA Evaluation

```bash
# With videos
python eval_scanqa.py --model qwen \
    --data_root data/scanqa \
    --scene_videos_root scannet_data/videos \
    --output results/scanqa

# With images
python eval_scanqa.py --model qwen \
    --data_root data/scanqa \
    --scene_images_root scannet_data/images \
    --output results/scanqa
```

### SQA3D Evaluation

```bash
# With videos
python eval_sqa3d.py --model qwen \
    --data_root SQA3D \
    --scene_videos_root scannet_data/videos \
    --split val \
    --output results/sqa3d

# With images
python eval_sqa3d.py --model qwen \
    --data_root SQA3D \
    --scene_images_root scannet_data/images \
    --split val \
    --output results/sqa3d
```

## Storage Requirements

- **Raw ScanNet .sens files**: ~1.5TB for full dataset
- **Rendered videos** (30fps, 640x480): ~200GB for all scenes
- **Rendered images** (10 views per scene, 640x480): ~300GB for all scenes

**Tip:** You can delete the raw .sens files after rendering to save space.

## Troubleshooting

### Missing Scenes

Some scenes may fail to render. Check the rendering logs:

```bash
# List rendered scenes
ls scannet_data/videos/ | wc -l  # Should be ~1500+ scenes

# Compare with required scenes for ScanQA
grep -o 'scene[0-9]*_[0-9]*' data/scanqa/ScanQA_v1.0/ScanQA_v1.0_val.json | sort | uniq > scanqa_scenes.txt

# Compare with required scenes for SQA3D
grep -o 'scene[0-9]*_[0-9]*' SQA3D/sqa_task/balanced/v1_balanced_questions_val_scannetv2.json | sort | uniq > sqa3d_scenes.txt

# Find missing scenes
comm -23 scanqa_scenes.txt <(ls scannet_data/videos/ | sed 's/.mp4//' | sort)
```

### Rendering Errors

If rendering fails:

1. **Check ScanNet credentials**: Make sure you have valid access
2. **Check disk space**: Rendering requires significant temporary space
3. **Update dependencies**: Ensure OpenCV, ffmpeg are installed
4. **Use alternative tools**: Try [ScanNet's official scripts](https://github.com/ScanNet/ScanNet)

### Memory Issues During Evaluation

If you get OOM errors when loading videos:

```bash
# Reduce frame sampling
python eval_scanqa.py --model qwen \
    --scene_videos_root scannet_data/videos \
    --max_frames 16  # Default is 32

# Or use images instead (loads on-demand)
python eval_scanqa.py --model qwen \
    --scene_images_root scannet_data/images
```

## Alternative: Pre-rendered Datasets

Some researchers share pre-rendered ScanNet visuals. Check:

- ScanQA repository issues/discussions
- SQA3D repository issues/discussions
- Contact benchmark authors

**Note:** Pre-rendered data must follow the same directory structure as above.

## Summary

✅ **One-time setup**: Download ScanNet once, render to `scannet_data/`  
✅ **Both benchmarks**: ScanQA and SQA3D use the same data  
✅ **Choose format**: Videos (efficient) or images (high quality)  
✅ **No duplication**: Single shared directory for all scenes  

**Questions?** See the main [README.md](README.md) or open an issue on GitHub.
