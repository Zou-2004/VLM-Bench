"""MMSI-Bench (Multi-Modal Spatial Intelligence Benchmark) dataset loader.

Dataset: https://huggingface.co/datasets/RunsenXu/MMSI-Bench
GitHub: https://github.com/OpenRobotLab/MMSI-Bench
"""

from typing import List, Dict, Any
from PIL import Image
from io import BytesIO

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class MMSIDataset:
    """MMSI-Bench dataset loader (multi-image spatial reasoning)."""
    
    def __init__(self, split: str = "test", cache_dir: str = None):
        """Initialize MMSI-Bench dataset.
        
        Args:
            split: Dataset split
            cache_dir: Directory to cache downloaded data
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Please install: pip install datasets")
        
        print(f"Loading MMSI-Bench ({split} split)...")
        self.dataset = load_dataset(
            "RunsenXu/MMSI-Bench",
            cache_dir=cache_dir
        )
        
        if split in self.dataset:
            self.data = self.dataset[split]
        else:
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
            Dict with keys: images (multiple), question, choices, answer, metadata
        """
        record = self.data[idx]
        
        # Extract multiple images (key feature of MMSI-Bench)
        images = self._load_images(record)
        
        # Extract question and choices
        raw_question = record.get("question", record.get("prompt", ""))
        choices = record.get("choices", record.get("options", []))
        
        # MMSI-Bench has choices embedded in the question text
        # Format: "Question\nOptions: A: choice1, B: choice2, C: choice3, D: choice4"
        if not choices and "Options:" in raw_question:
            question, choices = self._parse_question_with_options(raw_question)
        else:
            question = raw_question
        
        answer = record.get("answer", record.get("label", ""))
        
        metadata = {
            "idx": idx,
            "num_images": len(images),
            "category": record.get("category", "unknown")
        }
        
        return {
            "images": images,
            "question": question,
            "choices": choices,
            "answer": answer,
            "metadata": metadata
        }
    
    def _parse_question_with_options(self, text: str) -> tuple:
        """Parse question text that has embedded options.
        
        Args:
            text: Question text with embedded options like "Options: A: x, B: y, C: z"
            
        Returns:
            (question, choices_list)
        """
        import re
        
        # Split on "Options:"
        parts = text.split("Options:", 1)
        if len(parts) != 2:
            return text, []
        
        question = parts[0].strip()
        options_text = parts[1].strip()
        
        # Parse individual choices by finding letter markers
        # MMSI format: "A: Left while moving backward, B: Forward to the left, C: Forward to the right, D: Right while moving backward"
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        choices = []
        
        for i, letter in enumerate(letters):
            # Find this letter marker
            start_marker = f"{letter}:"
            if start_marker not in options_text:
                break
            
            start_idx = options_text.find(start_marker) + len(start_marker)
            
            # Find where this choice ends (next letter marker or end of string)
            # Look for the next letter followed by colon
            next_marker_idx = len(options_text)
            for next_letter in letters[i+1:]:
                next_marker = f" {next_letter}:"
                if next_marker in options_text[start_idx:]:
                    next_marker_idx = start_idx + options_text[start_idx:].find(next_marker)
                    break
            
            # Extract choice text
            choice_text = options_text[start_idx:next_marker_idx].strip()
            # Remove trailing comma if present
            choice_text = choice_text.rstrip(',').strip()
            
            if choice_text:
                choices.append(choice_text)
        
        return question, choices
    
    def _load_images(self, record: Dict) -> List[Image.Image]:
        """Load multiple images from record."""
        images = []
        
        # Try to find images field
        if "images" in record:
            img_list = record["images"]
            
            # Handle list of image data
            if isinstance(img_list, list):
                for img_data in img_list:
                    # Case 1: Dict with bytes
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        images.append(
                            Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                        )
                    # Case 2: PIL Image
                    elif hasattr(img_data, "convert"):
                        images.append(img_data.convert("RGB"))
        
        # Fallback to single image if "images" not found
        elif "image" in record:
            img_data = record["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                images.append(
                    Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                )
            elif hasattr(img_data, "convert"):
                images.append(img_data.convert("RGB"))
        
        # If no images found, return blank
        if not images:
            print(f"Warning: No images found in record")
            images = [Image.new("RGB", (224, 224), color="gray")]
        
        return images
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
