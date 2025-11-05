#!/bin/bash
# Setup script for VLM evaluation environment

set -e  # Exit on error

echo "======================================"
echo "VLM Evaluation Environment Setup"
echo "======================================"
echo ""

# Detect if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="vlbench"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Activating..."
        eval "$(conda shell.bash hook)"
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

# Create conda environment
echo "Creating conda environment: ${ENV_NAME}"
conda create -y -n ${ENV_NAME} python=3.10

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch (CUDA 12.1 - adjust if needed)
echo ""
echo "Installing PyTorch with CUDA support..."
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install -U transformers accelerate datasets pillow opencv-python decord av tqdm einops sentencepiece bitsandbytes

# Install evaluation metrics
echo ""
echo "Installing evaluation metrics..."
pip install nltk rouge-score sacrebleu pandas tabulate requests

# Install pycocoevalcap for ScanQA metrics
echo ""
echo "Installing pycocoevalcap..."
pip install git+https://github.com/salaniz/pycocoevalcap

# Download NLTK data (needed for some metrics)
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Optional: install vLLM for faster inference
read -p "Do you want to install vLLM for faster batched inference? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing vLLM..."
    pip install "vllm>=0.6.3"
fi

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
