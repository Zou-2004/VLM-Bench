"""Evaluation script for SQA3D benchmark.

SQA3D is an open-ended 3D scene question answering benchmark that tests
situated question answering capabilities.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List

from benchmark_loaders.sqa3d import SQA3DDataset
from metrics import Metrics


def format_sqa3d_prompt(situation: str, question: str) -> str:
    """Format SQA3D prompt with situation and question.
    
    Args:
        situation: Situation description
        question: The question text
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Situation: {situation}\n\nQuestion: {question}\n\nAnswer:"
    return prompt


def evaluate_sqa3d(
    model,
    dataset: SQA3DDataset,
    max_samples: int = None,
    output_dir: str = "results/sqa3d",
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate model on SQA3D benchmark.
    
    Args:
        model: VLM model instance
        dataset: SQA3D dataset
        max_samples: Maximum number of samples to evaluate (None = all)
        output_dir: Directory to save results
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with predictions and metadata
    """
    predictions = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    # Determine input type
    sample = dataset[0]
    if sample["video_path"] is not None:
        input_type = "video"
    elif sample["images"] is not None:
        input_type = "multi-image"
    else:
        input_type = "text-only"
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation")
    print(f"Input type: {input_type}")
    print(f"{'='*60}\n")
    
    iterator = tqdm(range(num_samples), desc="Evaluating SQA3D") if verbose else range(num_samples)
    
    skipped = 0
    for idx in iterator:
        sample = dataset[idx]
        
        # Extract fields
        situation = sample["situation"]
        question = sample["question"]
        answers = sample["answers"]
        scene_id = sample["scene_id"]
        images = sample["images"]
        video_path = sample["video_path"]
        metadata = sample["metadata"]
        
        # Format prompt
        prompt = format_sqa3d_prompt(situation, question)
        
        # Get model prediction
        try:
            if video_path:
                response = model.ask_video(video_path, prompt)
            elif images:
                response = model.ask_images(images, prompt, max_new_tokens=64)
            else:
                # Text-only (no scene visuals)
                print(f"\nWarning: No images or video for scene {scene_id}")
                response = "Unable to answer without scene visual information."
                skipped += 1
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            response = ""
            skipped += 1
        
        # Store prediction
        prediction = {
            "question_id": metadata["question_id"],
            "scene_id": scene_id,
            "question": question,
            "situation": situation,
            "ground_truth_answers": answers,
            "prediction": response.strip(),
            "metadata": metadata
        }
        predictions.append(prediction)
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    
    pred_file = os.path.join(output_dir, f"pred.{dataset.split}.json")
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples}")
    print(f"Skipped: {skipped}")
    print(f"{'='*60}\n")
    
    # Generate statistics
    stats = generate_sqa3d_statistics(predictions)
    
    # Print statistics
    print_sqa3d_statistics(stats)
    
    # Save statistics
    stats_file = os.path.join(output_dir, f"stats.{dataset.split}.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Format for official scoring if needed
    SQA3DDataset.format_predictions_for_scoring(
        [{"question_id": p["question_id"], "answer": p["prediction"]} for p in predictions],
        os.path.join(output_dir, f"predictions_{dataset.split}.json")
    )
    
    results = {
        "predictions": predictions,
        "statistics": stats,
        "num_samples": num_samples,
        "skipped": skipped
    }
    
    results_file = os.path.join(output_dir, f"results.{dataset.split}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Predictions saved to: {pred_file}")
    print(f"Full results saved to: {results_file}")
    print(f"Statistics saved to: {stats_file}")
    
    return results


def generate_sqa3d_statistics(predictions: List[Dict]) -> Dict[str, Any]:
    """Generate statistics from SQA3D predictions.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Statistics dictionary
    """
    stats = {
        "overall": {
            "total": len(predictions),
            "answered": sum(1 for p in predictions if p["prediction"]),
            "skipped": sum(1 for p in predictions if not p["prediction"])
        },
        "response_stats": {
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0
        },
        "by_scene": {},
        "by_answer_type": {},
        "by_question_type": {}
    }
    
    # Response length statistics
    lengths = [len(p["prediction"].split()) for p in predictions if p["prediction"]]
    if lengths:
        stats["response_stats"]["avg_length"] = sum(lengths) / len(lengths)
        stats["response_stats"]["min_length"] = min(lengths)
        stats["response_stats"]["max_length"] = max(lengths)
    
    # By scene
    for pred in predictions:
        scene = pred["scene_id"]
        if scene not in stats["by_scene"]:
            stats["by_scene"][scene] = 0
        stats["by_scene"][scene] += 1
    
    # By answer type
    for pred in predictions:
        answer_type = pred["metadata"].get("answer_type", "unknown")
        if answer_type not in stats["by_answer_type"]:
            stats["by_answer_type"][answer_type] = 0
        stats["by_answer_type"][answer_type] += 1
    
    # By question type
    for pred in predictions:
        question_type = pred["metadata"].get("question_type", "unknown")
        if question_type not in stats["by_question_type"]:
            stats["by_question_type"][question_type] = 0
        stats["by_question_type"][question_type] += 1
    
    return stats


def print_sqa3d_statistics(stats: Dict[str, Any]):
    """Print formatted statistics.
    
    Args:
        stats: Statistics dictionary
    """
    print("\n" + "="*70)
    print("DETAILED STATISTICS - SQA3D")
    print("="*70 + "\n")
    
    # Overall
    print("OVERALL:")
    print(f"  Total samples: {stats['overall']['total']}")
    print(f"  Answered: {stats['overall']['answered']}")
    print(f"  Skipped: {stats['overall']['skipped']}")
    print()
    
    # Response statistics
    print("RESPONSE STATISTICS:")
    print(f"  Average length (words): {stats['response_stats']['avg_length']:.2f}")
    print(f"  Min length: {stats['response_stats']['min_length']}")
    print(f"  Max length: {stats['response_stats']['max_length']}")
    print()
    
    # By scene (top 10)
    print("BY SCENE (top 10):")
    scenes_sorted = sorted(stats["by_scene"].items(), key=lambda x: x[1], reverse=True)
    for scene, count in scenes_sorted[:10]:
        print(f"  {scene:<25} {count} questions")
    print(f"\n  Total unique scenes: {len(stats['by_scene'])}")
    print()
    
    # By answer type
    if stats["by_answer_type"]:
        print("BY ANSWER TYPE:")
        for atype, count in sorted(stats["by_answer_type"].items()):
            print(f"  {atype:<25} {count} questions")
        print()
    
    # By question type
    if stats["by_question_type"] and stats["by_question_type"].get("N/A", 0) != stats["overall"]["total"]:
        print("BY QUESTION TYPE:")
        for qtype, count in sorted(stats["by_question_type"].items()):
            print(f"  {qtype:<25} {count} questions")
        print()
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on SQA3D benchmark")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen"],
        help="Model to evaluate"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model ID or path"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="SQA3D",
        help="Path to SQA3D data directory"
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
        help="Path to scene images directory (optional)"
    )
    parser.add_argument(
        "--scene_videos_root",
        type=str,
        default=None,
        help="Path to scene videos directory (optional)"
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
        default="results/sqa3d",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Set default model_id if not provided
    if args.model_id is None:
        model_defaults = {
            "qwen": "Qwen/Qwen2.5-VL-7B-Instruct"
        }
        args.model_id = model_defaults.get(args.model, "Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model} ({args.model_id})")
    print(f"{'='*60}\n")
    
    if args.model == "qwen":
        from models.qwen_vl import QwenVL
        model = QwenVL(model_id=args.model_id)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading SQA3D ({args.split} split)")
    print(f"{'='*60}\n")
    
    dataset = SQA3DDataset(
        split=args.split,
        data_root=args.data_root,
        scene_images_root=args.scene_images_root,
        scene_videos_root=args.scene_videos_root
    )
    
    # Run evaluation
    results = evaluate_sqa3d(
        model=model,
        dataset=dataset,
        max_samples=args.max_samples,
        output_dir=args.output,
        verbose=True
    )


if __name__ == "__main__":
    main()
