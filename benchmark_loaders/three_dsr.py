"""3DSRBench (3D Spatial Reasoning Benchmark) dataset loader.

Dataset: https://huggingface.co/datasets/ccvl/3DSRBench
Paper: 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark
"""

from typing import List, Dict, Any
from PIL import Image
from io import BytesIO
import requests

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class ThreeDSRDataset:
    """3DSRBench dataset loader."""
    
    def __init__(self, split: str = "test", cache_dir: str = None):
        """Initialize 3DSRBench dataset.
        
        Args:
            split: Dataset split
            cache_dir: Directory to cache downloaded data
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Please install: pip install datasets")
        
        print(f"Loading 3DSRBench ({split} split)...")
        self.dataset = load_dataset(
            "ccvl/3DSRBench",
            cache_dir=cache_dir
        )
        
        if split in self.dataset:
            self.data = self.dataset[split]
        else:
            # Use first available split
            available_splits = list(self.dataset.keys())
            if available_splits:
                print(f"Split '{split}' not found. Using '{available_splits[0]}'")
                self.data = self.dataset[available_splits[0]]
            else:
                raise ValueError("No data splits found")
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.
        
        Returns:
            Dict with keys: images, question, choices, answer, metadata
        """
        record = self.data[idx]
        
        # Extract image (may be URL or bytes)
        image = self._load_image(record)
        
        # Extract question
        question = record.get("question", record.get("prompt", ""))
        
        # Extract choices - 3DSRBench has A, B, C, D as separate keys
        choices = []
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            if letter in record:
                choice_text = record[letter]
                # Filter out "None" choices
                if choice_text and str(choice_text).strip().lower() != "none":
                    choices.append(choice_text)
                else:
                    break  # Stop when we hit None
        
        # If no choices found with letter keys, try alternatives
        if not choices:
            choices = record.get("choices", record.get("options", []))
        
        answer = record.get("answer", record.get("label", ""))
        
        metadata = {
            "idx": idx,
            "source": record.get("image_source", "unknown"),
            "category": record.get("category", "unknown")
        }
        
        return {
            "images": [image],
            "question": question,
            "choices": choices,
            "answer": answer,
            "metadata": metadata
        }
    
    def _load_image(self, record: Dict) -> Image.Image:
        """Load image from record (handles URLs and bytes)."""
        # Try different possible fields
        for field in ["image", "image_url", "img"]:
            if field in record:
                img_data = record[field]
                
                # Case 1: Dict with bytes
                if isinstance(img_data, dict) and "bytes" in img_data:
                    return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                
                # Case 2: PIL Image
                elif hasattr(img_data, "convert"):
                    return img_data.convert("RGB")
                
                # Case 3: URL string
                elif isinstance(img_data, str) and img_data.startswith("http"):
                    try:
                        response = requests.get(img_data, timeout=10)
                        return Image.open(BytesIO(response.content)).convert("RGB")
                    except Exception as e:
                        print(f"Warning: Failed to load image from URL: {e}")
        
        # Fallback: create a blank image
        print(f"Warning: Could not find image in record")
        return Image.new("RGB", (224, 224), color="gray")
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
