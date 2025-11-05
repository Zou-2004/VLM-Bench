"""CV-Bench (Cambrian Vision-Centric Benchmark) dataset loader.

Dataset: https://huggingface.co/datasets/nyu-visionx/CV-Bench
Paper: Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs
"""

from typing import List, Dict, Any
from PIL import Image
from io import BytesIO

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class CVBenchDataset:
    """CV-Bench dataset loader."""
    
    def __init__(self, split: str = "test", cache_dir: str = None):
        """Initialize CV-Bench dataset.
        
        Args:
            split: Dataset split (typically "test")
            cache_dir: Directory to cache downloaded data
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Please install: pip install datasets")
        
        print(f"Loading CV-Bench ({split} split)...")
        self.dataset = load_dataset(
            "nyu-visionx/CV-Bench",
            cache_dir=cache_dir
        )
        
        # CV-Bench contains both 2D and 3D subsets
        if split in self.dataset:
            self.data = self.dataset[split]
        else:
            # Combine all splits if specific split not found
            self.data = self.dataset
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.
        
        Returns:
            Dict with keys: images, question, choices, answer, metadata
        """
        record = self.data[idx]
        
        # Extract image
        image_data = record["image"]
        if isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(BytesIO(image_data["bytes"])).convert("RGB")
        else:
            image = image_data.convert("RGB")
        
        # Extract question and choices
        question = record.get("prompt", record.get("question", ""))
        choices = record.get("choices", record.get("options", []))
        answer = record.get("answer", "")
        
        # Metadata (source indicates 2D vs 3D)
        metadata = {
            "source": record.get("source", "unknown"),
            "idx": idx
        }
        
        return {
            "images": [image],
            "question": question,
            "choices": choices,
            "answer": answer,
            "metadata": metadata
        }
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
