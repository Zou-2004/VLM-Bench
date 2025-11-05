"""Evaluate VLM on multiple-choice benchmarks (CV-Bench, 3DSRBench, MMSI, BLINK).

Usage:
    python eval_mcq.py --model qwen --benchmark cvbench --output results/cvbench_qwen.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import pandas as pd

from models.base_vlm import BaseVLM
from models.qwen_vl import QwenVL
from metrics import Metrics, format_mcq_prompt, answer_to_letter

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


# Dataset loaders
DATASET_LOADERS = {
    "cvbench": ("benchmark_loaders.cvbench", "CVBenchDataset"),
    "3dsr": ("benchmark_loaders.three_dsr", "ThreeDSRDataset"),
    "mmsi": ("benchmark_loaders.mmsi", "MMSIDataset"),
    "blink": ("benchmark_loaders.blink", "BLINKDataset"),
}


def load_dataset(benchmark: str, **kwargs):
    """Dynamically load dataset."""
    if benchmark not in DATASET_LOADERS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    module_name, class_name = DATASET_LOADERS[benchmark]
    module = __import__(module_name, fromlist=[class_name])
    dataset_class = getattr(module, class_name)
    
    return dataset_class(**kwargs)


def evaluate_mcq(
    model: BaseVLM,
    dataset,
    max_samples: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate model on multiple-choice benchmark.
    
    Args:
        model: VLM model
        dataset: Dataset instance
        max_samples: Maximum number of samples to evaluate (None = all)
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with results
    """
    results = []
    correct = 0
    total = 0
    
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    iterator = tqdm(range(num_samples), desc="Evaluating") if verbose else range(num_samples)
    
    for idx in iterator:
        sample = dataset[idx]
        
        # Extract fields
        images = sample["images"]
        question = sample["question"]
        choices = sample["choices"]
        answer = sample["answer"]
        metadata = sample.get("metadata", {})
        
        # Convert answer to letter format
        gt_letter = answer_to_letter(answer, choices)
        
        # Format prompt
        prompt = format_mcq_prompt(question, choices)
        
        # Get model prediction
        try:
            response = model.ask_images(images, prompt, max_new_tokens=10)
            pred_letter = Metrics.extract_choice_letter(response, "ABCDEFGH"[:len(choices)])
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            response = ""
            pred_letter = ""
        
        # Check correctness
        is_correct = (pred_letter == gt_letter)
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        result = {
            "idx": idx,
            "question": question,
            "choices": choices,
            "ground_truth": gt_letter,
            "prediction": pred_letter,
            "raw_response": response,
            "correct": is_correct,
            "metadata": metadata
        }
        results.append(result)
        
        # Update progress bar with current accuracy
        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({"accuracy": f"{correct/total:.3f}"})
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0.0
    
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }
    
    return summary


def generate_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed statistics from evaluation results.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Statistics dictionary with breakdown by categories
    """
    stats = {
        "overall": {
            "total": results["total"],
            "correct": results["correct"],
            "accuracy": results["accuracy"]
        }
    }
    
    # Group by metadata categories
    if results["results"]:
        # Check what metadata fields are available
        sample_metadata = results["results"][0].get("metadata", {})
        
        # Statistics by subtask (for BLINK)
        if "subtask" in sample_metadata:
            subtask_stats = {}
            for result in results["results"]:
                subtask = result["metadata"].get("subtask", "unknown")
                if subtask not in subtask_stats:
                    subtask_stats[subtask] = {"total": 0, "correct": 0}
                subtask_stats[subtask]["total"] += 1
                if result["correct"]:
                    subtask_stats[subtask]["correct"] += 1
            
            # Calculate accuracies
            for subtask, counts in subtask_stats.items():
                counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            
            stats["by_subtask"] = subtask_stats
        
        # Statistics by source (for CV-Bench, 3DSR)
        if "source" in sample_metadata:
            source_stats = {}
            for result in results["results"]:
                source = result["metadata"].get("source", "unknown")
                if source not in source_stats:
                    source_stats[source] = {"total": 0, "correct": 0}
                source_stats[source]["total"] += 1
                if result["correct"]:
                    source_stats[source]["correct"] += 1
            
            # Calculate accuracies
            for source, counts in source_stats.items():
                counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            
            stats["by_source"] = source_stats
        
        # Statistics by category (for 3DSR)
        if "category" in sample_metadata:
            category_stats = {}
            for result in results["results"]:
                category = result["metadata"].get("category", "unknown")
                if category not in category_stats:
                    category_stats[category] = {"total": 0, "correct": 0}
                category_stats[category]["total"] += 1
                if result["correct"]:
                    category_stats[category]["correct"] += 1
            
            # Calculate accuracies
            for category, counts in category_stats.items():
                counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            
            stats["by_category"] = category_stats
    
    return stats


def print_statistics(stats: Dict[str, Any], benchmark: str):
    """Print statistics in a readable format.
    
    Args:
        stats: Statistics dictionary
        benchmark: Benchmark name
    """
    print(f"\n{'='*70}")
    print(f"DETAILED STATISTICS - {benchmark.upper()}")
    print(f"{'='*70}\n")
    
    # Overall stats
    print(f"OVERALL:")
    print(f"  Total samples: {stats['overall']['total']}")
    print(f"  Correct: {stats['overall']['correct']}")
    print(f"  Accuracy: {stats['overall']['accuracy']:.4f} ({stats['overall']['accuracy']*100:.2f}%)")
    
    # By subtask
    if "by_subtask" in stats:
        print(f"\nBY SUBTASK:")
        sorted_subtasks = sorted(stats["by_subtask"].items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for subtask, counts in sorted_subtasks:
            print(f"  {subtask:30s} {counts['correct']:4d}/{counts['total']:4d}  {counts['accuracy']:.4f} ({counts['accuracy']*100:.2f}%)")
    
    # By source
    if "by_source" in stats:
        print(f"\nBY SOURCE:")
        sorted_sources = sorted(stats["by_source"].items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for source, counts in sorted_sources:
            print(f"  {source:30s} {counts['correct']:4d}/{counts['total']:4d}  {counts['accuracy']:.4f} ({counts['accuracy']*100:.2f}%)")
    
    # By category
    if "by_category" in stats:
        print(f"\nBY CATEGORY:")
        sorted_categories = sorted(stats["by_category"].items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for category, counts in sorted_categories:
            print(f"  {category:30s} {counts['correct']:4d}/{counts['total']:4d}  {counts['accuracy']:.4f} ({counts['accuracy']*100:.2f}%)")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on MCQ benchmarks")
    
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
    
    # Benchmark arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["cvbench", "3dsr", "mmsi", "blink"],
        help="Benchmark to evaluate on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split"
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default=None,
        help="Subtask (for BLINK)"
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
        default=None,
        help="Output path for results JSON"
    )
    
    args = parser.parse_args()
    
    # Auto-adjust split for BLINK (test split has hidden answers)
    if args.benchmark == "blink" and args.split == "test":
        print("\n" + "="*60)
        print("WARNING: BLINK test split has hidden answers!")
        print("Switching to 'val' split for evaluation.")
        print("To override, use: --split test")
        print("="*60 + "\n")
        args.split = "val"
    
    # Set default model_id if not provided
    if args.model_id is None:
        model_defaults = {
            "qwen": "Qwen/Qwen2.5-VL-7B-Instruct"
            # "llava": "llava-hf/llava-1.5-7b-hf",
            # "internvl2": "OpenGVLab/InternVL2-8B",
            # "minicpm-v": "openbmb/MiniCPM-V-2_6",
        }
        args.model_id = model_defaults.get(args.model, "Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model} ({args.model_id})")
    print(f"{'='*60}\n")
    
    if args.model == "qwen":
        model = QwenVL(model_id=args.model_id)
    # elif args.model == "llava":
    #     if LLaVAModel is None:
    #         raise ImportError("LLaVA model not available. Install required dependencies.")
    #     model = LLaVAModel(model_id=args.model_id)
    # elif args.model == "internvl2":
    #     if InternVL2Model is None:
    #         raise ImportError("InternVL2 model not available. Install required dependencies.")
    #     model = InternVL2Model(model_id=args.model_id)
    # elif args.model == "minicpm-v":
    #     if MiniCPMVModel is None:
    #         raise ImportError("MiniCPM-V model not available. Install required dependencies.")
    #     model = MiniCPMVModel(model_id=args.model_id)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading benchmark: {args.benchmark}")
    print(f"{'='*60}\n")
    
    dataset_kwargs = {"split": args.split}
    if args.benchmark == "blink" and args.subtask:
        dataset_kwargs["subtask"] = args.subtask
    
    dataset = load_dataset(args.benchmark, **dataset_kwargs)
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Starting evaluation")
    print(f"{'='*60}\n")
    
    results = evaluate_mcq(
        model=model,
        dataset=dataset,
        max_samples=args.max_samples,
        verbose=True
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Benchmark: {args.benchmark}")
    if args.subtask:
        print(f"Subtask: {args.subtask}")
    print(f"Total samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # Generate and print detailed statistics
    stats = generate_statistics(results)
    print_statistics(stats, args.benchmark)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        
        # Save statistics as separate file
        stats_path = output_path.parent / (output_path.stem + "_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")
        
        # Also save summary CSV
        csv_path = output_path.with_suffix(".csv")
        df = pd.DataFrame(results["results"])
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
