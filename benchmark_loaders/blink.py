"""BLINK Benchmark dataset loader.

Dataset: https://huggingface.co/datasets/BLINK-Benchmark/BLINK
Website: https://zeyofu.github.io/blink/
"""

from typing import List, Dict, Any, Optional
from PIL import Image
from io import BytesIO

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class BLINKDataset:
    """BLINK Benchmark dataset loader (perception-centric visual understanding)."""
    
    # Available subtasks in BLINK
    SUBTASKS = [
        "Art_Style",
        "Counting",
        "Forensic_Detection",
        "Functional_Correspondence",
        "IQ_Test",
        "Jigsaw",
        "Multi-view_Reasoning",
        "Object_Localization",
        "Relative_Depth",
        "Relative_Reflectance",
        "Semantic_Correspondence",
        "Spatial_Relation",
        "Visual_Correspondence",
        "Visual_Similarity",
    ]
    
    def __init__(
        self, 
        subtask: Optional[str] = None, 
        split: str = "val",
        cache_dir: str = None
    ):
        """Initialize BLINK dataset.
        
        Args:
            subtask: Specific subtask to load (None = all subtasks)
            split: Dataset split (typically "val")
            cache_dir: Directory to cache downloaded data
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Please install: pip install datasets")
        
        self.subtask = subtask
        
        if subtask:
            print(f"Loading BLINK subtask: {subtask} ({split} split)...")
            self.dataset = load_dataset(
                "BLINK-Benchmark/BLINK",
                subtask,
                cache_dir=cache_dir,
                split=split
            )
            self.data = self.dataset
        else:
            # Load all subtasks and concatenate
            print(f"Loading all BLINK subtasks ({split} split)...")
            all_data = []
            for task in self.SUBTASKS:
                try:
                    print(f"  Loading {task}...")
                    task_data = load_dataset(
                        "BLINK-Benchmark/BLINK",
                        task,
                        cache_dir=cache_dir,
                        split=split
                    )
                    # Add subtask label to each record
                    all_data.extend([{**record, "_subtask": task} for record in task_data])
                except Exception as e:
                    print(f"  Warning: Failed to load {task}: {e}")
            
            # Create a simple list-based dataset
            self.data = all_data
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.
        
        Returns:
            Dict with keys: images (may be multiple), question, choices, answer, metadata
        """
        record = self.data[idx]
        
        # Extract image(s) - BLINK can have single or multiple images
        images = self._load_images(record)
        
        # Extract question and choices
        question = record.get("question", record.get("prompt", ""))
        choices = record.get("choices", record.get("options", []))
        answer = record.get("answer", record.get("label", ""))
        
        metadata = {
            "idx": idx,
            "subtask": self.subtask or record.get("_subtask", record.get("task", "unknown")),
            "num_images": len(images)
        }
        
        return {
            "images": images,
            "question": question,
            "choices": choices,
            "answer": answer,
            "metadata": metadata
        }
    
    def _load_images(self, record: Dict) -> List[Image.Image]:
        """Load image(s) from record."""
        images = []
        
        # BLINK has image_1, image_2, image_3, image_4 fields
        for i in range(1, 10):  # Check up to image_9
            img_field = f"image_{i}"
            if img_field in record:
                img_data = record[img_field]
                
                # Skip None/null images
                if img_data is None:
                    continue
                
                # Handle PIL Image directly
                if hasattr(img_data, "convert"):
                    images.append(img_data.convert("RGB"))
                # Handle dict with bytes
                elif isinstance(img_data, dict) and "bytes" in img_data:
                    images.append(
                        Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                    )
        
        # Try "images" field (multiple images)
        if not images and "images" in record:
            img_list = record["images"]
            
            if isinstance(img_list, list):
                for img_data in img_list:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        images.append(
                            Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                        )
                    elif hasattr(img_data, "convert"):
                        images.append(img_data.convert("RGB"))
        
        # Try single "image" field
        if not images and "image" in record:
            img_data = record["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                images.append(
                    Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                )
            elif hasattr(img_data, "convert"):
                images.append(img_data.convert("RGB"))
        
        # Fallback
        if not images:
            print(f"Warning: No images found in record")
            images = [Image.new("RGB", (224, 224), color="gray")]
        
        return images
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
