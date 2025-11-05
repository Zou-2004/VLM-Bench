"""Format evaluation results into publication-ready tables."""

from typing import Dict, List, Any
from tabulate import tabulate
import json
from pathlib import Path


def format_scanqa_table(
    results_dict: Dict[str, Dict[str, float]],
    model_names: List[str] = None,
    tablefmt: str = "github"
) -> str:
    """Format ScanQA results into a table like the paper.
    
    Args:
        results_dict: Dict mapping model names to their scores
                      Each score dict should have: BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr
        model_names: Optional list of model names to include (in order)
        tablefmt: Table format for tabulate (github, latex, etc.)
        
    Returns:
        Formatted table string
    """
    if model_names is None:
        model_names = list(results_dict.keys())
    
    headers = ["Methods", "BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    
    rows = []
    for name in model_names:
        if name not in results_dict:
            continue
        
        scores = results_dict[name]
        
        row = [
            name,
            f"{scores.get('BLEU-1', 0.0):.1f}",
            f"{scores.get('BLEU-4', 0.0):.1f}",
            f"{scores.get('METEOR', 0.0):.1f}",
            f"{scores.get('ROUGE-L', 0.0):.1f}",
            f"{scores.get('CIDEr', 0.0):.1f}"
        ]
        rows.append(row)
    
    return tabulate(rows, headers=headers, tablefmt=tablefmt)


def format_mcq_table(
    results_dict: Dict[str, float],
    model_names: List[str] = None,
    benchmark_name: str = "Benchmark",
    tablefmt: str = "github"
) -> str:
    """Format MCQ benchmark results into a table.
    
    Args:
        results_dict: Dict mapping model names to accuracy scores
        model_names: Optional list of model names to include (in order)
        benchmark_name: Name of the benchmark for the header
        tablefmt: Table format for tabulate
        
    Returns:
        Formatted table string
    """
    if model_names is None:
        model_names = list(results_dict.keys())
    
    headers = ["Model", f"{benchmark_name} Accuracy (%)"]
    
    rows = []
    for name in model_names:
        if name not in results_dict:
            continue
        
        accuracy = results_dict[name]
        row = [name, f"{accuracy * 100:.2f}"]
        rows.append(row)
    
    return tabulate(rows, headers=headers, tablefmt=tablefmt)


def format_multi_benchmark_table(
    results_dict: Dict[str, Dict[str, float]],
    benchmarks: List[str],
    model_names: List[str] = None,
    tablefmt: str = "github"
) -> str:
    """Format results across multiple benchmarks into one table.
    
    Args:
        results_dict: Nested dict: {model_name: {benchmark: accuracy}}
        benchmarks: List of benchmark names (columns)
        model_names: Optional list of model names (rows)
        tablefmt: Table format for tabulate
        
    Returns:
        Formatted table string
    """
    if model_names is None:
        model_names = list(results_dict.keys())
    
    headers = ["Model"] + benchmarks
    
    rows = []
    for model in model_names:
        if model not in results_dict:
            continue
        
        row = [model]
        for benchmark in benchmarks:
            score = results_dict[model].get(benchmark, 0.0)
            row.append(f"{score * 100:.2f}")
        
        rows.append(row)
    
    return tabulate(rows, headers=headers, tablefmt=tablefmt)


def load_scanqa_scores(score_file: Path) -> Dict[str, float]:
    """Load scores from ScanQA scoring script output.
    
    Args:
        score_file: Path to scores JSON file
        
    Returns:
        Dictionary of metric scores
    """
    with open(score_file, "r") as f:
        scores = json.load(f)
    
    return scores


def load_mcq_results(result_file: Path) -> Dict[str, Any]:
    """Load MCQ evaluation results.
    
    Args:
        result_file: Path to results JSON file
        
    Returns:
        Results dictionary
    """
    with open(result_file, "r") as f:
        results = json.load(f)
    
    return results


def print_scanqa_table_from_file(
    score_file: str,
    model_name: str = "Qwen2.5-VL-7B",
    tablefmt: str = "github"
):
    """Load ScanQA scores from file and print formatted table.
    
    Args:
        score_file: Path to scores JSON file
        model_name: Name to display for the model
        tablefmt: Table format
    """
    scores = load_scanqa_scores(Path(score_file))
    
    results_dict = {model_name: scores}
    
    table = format_scanqa_table(results_dict, tablefmt=tablefmt)
    print(table)


def main():
    """Example usage of table formatting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Format evaluation results into tables")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scanqa", "mcq", "multi"],
        required=True,
        help="Table mode"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file(s) (JSON)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-VL-7B",
        help="Model name for display"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="github",
        choices=["github", "latex", "grid", "simple"],
        help="Table format"
    )
    
    args = parser.parse_args()
    
    if args.mode == "scanqa":
        # Example: python tables/format_tables.py --mode scanqa --input results/scanqa/scores.json
        print_scanqa_table_from_file(
            args.input,
            model_name=args.model_name,
            tablefmt=args.format
        )
    
    elif args.mode == "mcq":
        # Example: python tables/format_tables.py --mode mcq --input results/cvbench.json
        results = load_mcq_results(Path(args.input))
        
        results_dict = {args.model_name: results["accuracy"]}
        
        benchmark_name = Path(args.input).stem
        table = format_mcq_table(
            results_dict,
            benchmark_name=benchmark_name,
            tablefmt=args.format
        )
        print(table)
    
    elif args.mode == "multi":
        # Would need to load multiple files
        print("Multi-benchmark mode: provide multiple result files")


if __name__ == "__main__":
    main()
