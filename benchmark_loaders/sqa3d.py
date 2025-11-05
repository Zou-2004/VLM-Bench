"""SQA3D dataset loader.

Dataset: https://sqa3d.github.io/
Paper: SQA3D: Situated Question Answering in 3D Scenes (ICLR 2023)
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import glob


class SQA3DDataset:
    """SQA3D dataset loader (Situated 3D Question Answering)."""
    
    def __init__(
        self, 
        split: str = "val",
        data_root: str = "SQA3D",
        scene_images_root: Optional[str] = None,
        scene_videos_root: Optional[str] = None
    ):
        """Initialize SQA3D dataset.
        
        Args:
            split: Dataset split ("train", "val", "test")
            data_root: Path to SQA3D data directory
            scene_images_root: Path to scene images (for multi-view image input)
            scene_videos_root: Path to scene videos (for video input)
        """
        self.split = split
        self.data_root = Path(data_root)
        self.scene_images_root = Path(scene_images_root) if scene_images_root else None
        self.scene_videos_root = Path(scene_videos_root) if scene_videos_root else None
        
        # Load questions
        questions_path = self.data_root / "sqa_task" / "balanced" / f"v1_balanced_questions_{split}_scannetv2.json"
        
        if not questions_path.exists():
            raise FileNotFoundError(
                f"SQA3D questions not found at {questions_path}. "
                f"Please check the data path."
            )
        
        print(f"Loading SQA3D {split} split from {questions_path}...")
        with open(questions_path, "r") as f:
            questions_data = json.load(f)
        
        self.questions = questions_data["questions"]
        
        # Load annotations (contains answers)
        annotations_path = self.data_root / "sqa_task" / "balanced" / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
        
        self.annotations = {}
        if annotations_path.exists():
            print(f"Loading annotations from {annotations_path}...")
            with open(annotations_path, "r") as f:
                annotations_data = json.load(f)
            
            # Create mapping from question_id to annotation
            for ann in annotations_data["annotations"]:
                self.annotations[ann["question_id"]] = ann
        
        print(f"Loaded {len(self.questions)} samples")
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.
        
        Returns:
            Dict with keys: question, answer(s), scene_id, situation, images/video_path, metadata
        """
        question_record = self.questions[idx]
        
        # Extract question and situation
        question = question_record.get("question", "")
        situation = question_record.get("situation", "")
        alternative_situations = question_record.get("alternative_situation", [])
        scene_id = question_record.get("scene_id", "")
        question_id = question_record.get("question_id", idx)
        
        # Get annotation (with answer)
        annotation = self.annotations.get(question_id, {})
        
        # Extract answer(s)
        answers = []
        if "answers" in annotation:
            for ans in annotation["answers"]:
                answers.append(ans["answer"])
        
        # Try to find scene images or video
        images = None
        video_path = None
        
        if self.scene_images_root:
            images = self._load_scene_images(scene_id)
        
        if self.scene_videos_root:
            video_path = self._get_scene_video(scene_id)
        
        # Get position and rotation from annotation (spatial information)
        position = annotation.get("position", {})
        rotation = annotation.get("rotation", {})
        
        metadata = {
            "idx": idx,
            "scene_id": scene_id,
            "question_id": question_id,
            "situation": situation,
            "alternative_situations": alternative_situations,
            "position": position,
            "rotation": rotation,
            "answer_type": annotation.get("answer_type", "unknown"),
            "question_type": annotation.get("question_type", "unknown")
        }
        
        return {
            "question": question,
            "situation": situation,
            "answers": answers,  # List of acceptable answers
            "scene_id": scene_id,
            "images": images,
            "video_path": video_path,
            "metadata": metadata
        }
    
    def _load_scene_images(self, scene_id: str) -> Optional[List[Image.Image]]:
        """Load multi-view images for a scene."""
        if not self.scene_images_root:
            return None
        
        # Try common directory structures
        scene_dir = self.scene_images_root / scene_id
        
        if not scene_dir.exists():
            return None
        
        # Load all images from scene directory
        image_paths = sorted(glob.glob(str(scene_dir / "*.jpg"))) + \
                      sorted(glob.glob(str(scene_dir / "*.png")))
        
        if not image_paths:
            return None
        
        images = []
        for img_path in image_paths[:32]:  # Limit to 32 views
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
        
        return images if images else None
    
    def _get_scene_video(self, scene_id: str) -> Optional[str]:
        """Get path to scene video."""
        if not self.scene_videos_root:
            return None
        
        # Try common video formats
        for ext in [".mp4", ".avi", ".mkv"]:
            video_path = self.scene_videos_root / f"{scene_id}{ext}"
            if video_path.exists():
                return str(video_path)
        
        return None
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
    
    @staticmethod
    def format_predictions_for_scoring(predictions: List[Dict], output_path: str):
        """Format predictions for SQA3D evaluation.
        
        Args:
            predictions: List of dicts with keys: question_id, answer
            output_path: Path to save formatted predictions
        """
        formatted = {}
        for pred in predictions:
            formatted[str(pred["question_id"])] = pred["answer"]
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        
        print(f"Saved predictions to {output_path}")
