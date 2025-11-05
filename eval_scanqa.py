"""Evaluate VLM on ScanQA benchmark (open-ended 3D scene QA).

Usage:
    python eval_scanqa.py --model qwen --data_root data/scanqa --output results/scanqa_qwen
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

from models.base_vlm import BaseVLM
from models.qwen_vl import QwenVL
from benchmark_loaders.scanqa import ScanQADataset

# Optional model imports
try:
    from models.llava_model import LLaVAModel
except ImportError:
    LLaVAModel = None

try:
    from models.internvl2_model import InternVL2Model
except ImportError:
    InternVL2Model = None

try:
    from models.minicpm_v_model import MiniCPMVModel
except ImportError:
    MiniCPMVModel = None


PROMPT_TEMPLATE = (
    "You are answering questions about an indoor 3D scene. "
    "Provide a concise answer in one or two words. "
    "If you are uncertain, provide your best guess.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def evaluate_scanqa(
    model: BaseVLM,
    dataset: ScanQADataset,
    max_samples: int = None,
    verbose: bool = True,
    use_video: bool = False
) -> Dict[str, Any]:
    """Evaluate model on ScanQA benchmark.
    
    Args:
        model: VLM model
        dataset: ScanQA dataset instance
        max_samples: Maximum number of samples to evaluate (None = all)
        verbose: Whether to show progress bar
        use_video: Whether to use video input (vs multi-image)
        
    Returns:
        Dictionary with predictions
    """
    predictions = []
    skipped = 0
    
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    iterator = tqdm(range(num_samples), desc="Evaluating ScanQA") if verbose else range(num_samples)
    
    for idx in iterator:
        sample = dataset[idx]
        
        question = sample["question"]
        answers = sample["answers"]  # Ground truth (for reference, not used in inference)
        scene_id = sample["scene_id"]
        images = sample["images"]
        video_path = sample["video_path"]
        metadata = sample["metadata"]
        question_id = metadata["question_id"]
        
        # Format prompt
        prompt = PROMPT_TEMPLATE.format(question=question)
        
        # Get model prediction
        try:
            if use_video and video_path:
                # Use video input
                response = model.ask_video(
                    video_path, 
                    prompt, 
                    max_new_tokens=20
                )
            elif images:
                # Use multi-image input
                response = model.ask_images(
                    images, 
                    prompt, 
                    max_new_tokens=20
                )
            else:
                # No visual input available
                print(f"\nWarning: No images or video for scene {scene_id}")
                response = "unknown"
                skipped += 1
        except Exception as e:
            print(f"\nError on sample {idx} (scene {scene_id}): {e}")
            response = "unknown"
            skipped += 1
        
        # Clean up response (extract just the answer)
        response = response.strip()
        # Remove common prefixes
        for prefix in ["Answer:", "A:", "The answer is", "It is"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Store prediction
        prediction = {
            "question_id": question_id,
            "answer": response,
            "scene_id": scene_id,
            "question": question,
            "ground_truth": answers  # For reference
        }
        predictions.append(prediction)
    
    summary = {
        "total": num_samples,
        "skipped": skipped,
        "predictions": predictions
    }
    
    return summary


def generate_scanqa_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed statistics from ScanQA evaluation results.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Statistics dictionary with breakdown by categories
    """
    stats = {
        "overall": {
            "total": results["total"],
            "skipped": results["skipped"],
            "answered": results["total"] - results["skipped"]
        }
    }
    
    # Group by scene (scene_id)
    scene_stats = {}
    for pred in results["predictions"]:
        scene_id = pred["scene_id"]
        if scene_id not in scene_stats:
            scene_stats[scene_id] = {"count": 0, "questions": []}
        scene_stats[scene_id]["count"] += 1
        scene_stats[scene_id]["questions"].append({
            "question_id": pred["question_id"],
            "question": pred["question"],
            "answer": pred["answer"]
        })
    
    stats["by_scene"] = {
        scene_id: {"count": info["count"]}
        for scene_id, info in scene_stats.items()
    }
    
    # Response length statistics
    response_lengths = [len(p["answer"].split()) for p in results["predictions"]]
    if response_lengths:
        stats["response_stats"] = {
            "avg_length": sum(response_lengths) / len(response_lengths),
            "min_length": min(response_lengths),
            "max_length": max(response_lengths),
            "total_responses": len(response_lengths)
        }
    
    return stats


def print_scanqa_statistics(stats: Dict[str, Any]):
    """Print ScanQA statistics in a readable format.
    
    Args:
        stats: Statistics dictionary
    """
    print(f"\n{'='*70}")
    print(f"DETAILED STATISTICS - SCANQA")
    print(f"{'='*70}\n")
    
    # Overall stats
    print(f"OVERALL:")
    print(f"  Total samples: {stats['overall']['total']}")
    print(f"  Answered: {stats['overall']['answered']}")
    print(f"  Skipped: {stats['overall']['skipped']}")
    
    # Response statistics
    if "response_stats" in stats:
        rs = stats["response_stats"]
        print(f"\nRESPONSE STATISTICS:")
        print(f"  Average length (words): {rs['avg_length']:.2f}")
        print(f"  Min length: {rs['min_length']}")
        print(f"  Max length: {rs['max_length']}")
    
    # Scene distribution
    if "by_scene" in stats:
        print(f"\nBY SCENE (top 10):")
        sorted_scenes = sorted(
            stats["by_scene"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:10]
        for scene_id, info in sorted_scenes:
            print(f"  {scene_id:20s} {info['count']:4d} questions")
        
        total_scenes = len(stats["by_scene"])
        print(f"\n  Total unique scenes: {total_scenes}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on ScanQA")
    
    # Model arguments
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen",
        choices=["qwen", "llava", "internvl2", "minicpm-v"],
        help="Model to use"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model ID (e.g., Hugging Face model ID). If not provided, uses default for the model."
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/scanqa",
        help="Path to ScanQA data directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split"
    )
    parser.add_argument(
        "--scene_images_root",
        type=str,
        default=None,
        help="Path to scene multi-view images"
    )
    parser.add_argument(
        "--scene_videos_root",
        type=str,
        default=None,
        help="Path to scene videos"
    )
    parser.add_argument(
        "--use_video",
        action="store_true",
        help="Use video input instead of multi-image"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scanqa",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Set default model_id if not provided
    if args.model_id is None:
        model_defaults = {
            "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
            "llava": "llava-hf/llava-1.5-7b-hf",
            "internvl2": "OpenGVLab/InternVL2-8B",
            "minicpm-v": "openbmb/MiniCPM-V-2_6",
        }
        args.model_id = model_defaults.get(args.model, "Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model} ({args.model_id})")
    print(f"{'='*60}\n")
    
    if args.model == "qwen":
        model = QwenVL(model_id=args.model_id)
    elif args.model == "llava":
        if LLaVAModel is None:
            raise ImportError("LLaVA model not available. Install required dependencies.")
        model = LLaVAModel(model_id=args.model_id)
    elif args.model == "internvl2":
        if InternVL2Model is None:
            raise ImportError("InternVL2 model not available. Install required dependencies.")
        model = InternVL2Model(model_id=args.model_id)
    elif args.model == "minicpm-v":
        if MiniCPMVModel is None:
            raise ImportError("MiniCPM-V model not available. Install required dependencies.")
        model = MiniCPMVModel(model_id=args.model_id)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading ScanQA ({args.split} split)")
    print(f"{'='*60}\n")
    
    dataset = ScanQADataset(
        split=args.split,
        data_root=args.data_root,
        scene_images_root=args.scene_images_root,
        scene_videos_root=args.scene_videos_root
    )
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Starting evaluation")
    print(f"Input type: {'video' if args.use_video else 'multi-image'}")
    print(f"{'='*60}\n")
    
    results = evaluate_scanqa(
        model=model,
        dataset=dataset,
        max_samples=args.max_samples,
        verbose=True,
        use_video=args.use_video
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total samples: {results['total']}")
    print(f"Skipped: {results['skipped']}")
    print(f"{'='*60}\n")
    
    # Generate and print detailed statistics
    stats = generate_scanqa_statistics(results)
    print_scanqa_statistics(stats)
    
    # Save predictions in ScanQA format
    output_path = output_dir / f"pred.{args.split}.json"
    
    # Format for ScanQA scoring script
    formatted_predictions = [
        {"question_id": str(p["question_id"]), "answer": p["answer"]}
        for p in results["predictions"]
    ]
    
    with open(output_path, "w") as f:
        json.dump(formatted_predictions, f, indent=2)
    
    print(f"Predictions saved to: {output_path}")
    
    # Also save full results with ground truth for reference
    full_output_path = output_dir / f"results.{args.split}.json"
    with open(full_output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Full results saved to: {full_output_path}")
    
    # Save statistics
    stats_path = output_dir / f"stats.{args.split}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_path}")
    
    # Print scoring instructions
    print(f"\n{'='*60}")
    print(f"To compute metrics, run:")
    print(f"{'='*60}")
    print(f"cd data/scanqa")
    print(f"python scripts/score.py --folder ../../{output_dir.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
