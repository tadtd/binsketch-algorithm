"""Experiment 1: Accuracy of Similarity Estimation.

This experiment evaluates how accurately different sketching algorithms
can estimate pairwise similarities compared to ground truth values.
"""
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np

from src import BinSketch, BinaryCompressionSchema, MinHash, SimHash
from src.gpu_utils import GPUConfig
from save_ground_truth import (
    calculate_experiment1_ground_truth,
    load_experiment1_ground_truth,
    save_experiment1_ground_truth
)

# Import from local utils (works both as module and standalone)

from experiment.utils import (
    load_data, filter_pairs_by_threshold, run_algorithm_experiment, save_plot
)


# Algorithm mapping
ALGO_MAP = {
    'BinSketch': BinSketch,
    'BCS': BinaryCompressionSchema,
    'MinHash': MinHash,
    'SimHash': SimHash
}

# Default compression lengths
DEFAULT_COMPRESSION_LENGTHS = [100, 500, 1000, 2000, 3000, 4000, 5000]


def run_experiment1(
    data_path: str,
    algorithms: List[str],
    thresholds: List[float],
    similarity_score: str,
    eval_metric: str,
    compression_lengths: List[int],
    output_dir: str,
    seed: int = 42,
    ground_truth_path: Optional[str] = None
) -> None:
    """
    Run Experiment 1: Accuracy of Similarity Estimation.
    
    This experiment compares estimated similarities from compressed representations
    against ground truth similarities from the original data.
    
    Args:
        data_path: Path to data file
        algorithms: List of algorithm names to test
        thresholds: List of similarity thresholds
        similarity_score: Similarity metric name ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        eval_metric: Evaluation metric name ('mse', 'minus_log_mse')
        compression_lengths: List of compression lengths to test
        output_dir: Output directory for results
        seed: Random seed
        ground_truth_path: Optional path to ground truth JSON file
    """
    X_dense, X_csr = load_data(data_path)
    
    dataset_name = Path(data_path).stem.replace('_binary', '')
    
    # Load or calculate ground truth
    if ground_truth_path is not None:
        ground_truth_file = Path(ground_truth_path)
    else:
        ground_truth_file = Path(output_dir) / f"ground_truth_{dataset_name}_{similarity_score}.json"
    
    if ground_truth_file.exists():
        gt_data = load_experiment1_ground_truth(str(ground_truth_file))
        ground_truth = gt_data['ground_truth']
        pair_indices = gt_data['pairs']
        print(f"Loaded {len(ground_truth)} ground truth pairs")
    else:
        print(f"Calculating ground truth (this may take a while)...")
        ground_truth, pair_indices = calculate_experiment1_ground_truth(X_dense, similarity_score)
        print(f"Saving ground truth to {ground_truth_file}...")
        save_experiment1_ground_truth(data_path, similarity_score, ground_truth, pair_indices, str(ground_truth_file))
        print(f"Saved {len(ground_truth)} ground truth pairs")
    
    # Run experiments for each threshold
    for threshold in thresholds:
        print(f"\n{'='*40}")
        print(f"Processing Threshold {threshold}")
        print(f"{'='*40}")
        
        gt_filtered, pairs_filtered = filter_pairs_by_threshold(
            ground_truth, pair_indices, threshold
        )
        
        if gt_filtered is None:
            print(f"  No pairs found > {threshold}")
            continue
        
        print(f"  Valid Pairs: {len(gt_filtered)}")
        
        results = {}
        for algo_name in algorithms:
            if algo_name not in ALGO_MAP:
                print(f"  Unknown algorithm: {algo_name}, skipping...")
                continue
            
            print(f"  > Running {algo_name}...")
            model = ALGO_MAP[algo_name](seed=seed)
            
            try:
                scores = run_algorithm_experiment(
                    model, X_csr, pairs_filtered, gt_filtered,
                    compression_lengths, algo_name, similarity_score, eval_metric
                )
                results[algo_name] = scores
            except Exception as e:
                print(f"  Error: Failed to run {algo_name}: {e}")
                continue
        
        if results:
            filepath = save_plot(
                compression_lengths, results, similarity_score, eval_metric, threshold, output_dir, dataset_name
            )
            print(f"  Saved: {filepath}")
        else:
            print(f"  No results to plot for threshold {threshold}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment 1: Accuracy of Similarity Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to the .npy file containing the binary matrix'
    )
    
    parser.add_argument(
        '--algo', nargs='+', default=['BinSketch'],
        choices=list(ALGO_MAP.keys()),
        help='Algorithms to run'
    )
    
    parser.add_argument(
        '--threshold', nargs='+', type=float, default=[0.9],
        help='Similarity thresholds to evaluate'
    )
    
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration (requires CuPy and CUDA)'
    )
    
    parser.add_argument(
        '--eval_metric', type=str, default='minus_log_mse',
        choices=['mse', 'minus_log_mse'],
        help='Evaluation metric for comparing estimates to ground truth'
    )
    
    parser.add_argument(
        '--similarity_score', type=str, default='cosine_similarity',
        choices=['cosine_similarity', 'jaccard_similarity', 'inner_product'],
        help='Similarity metric for ground truth calculation and estimation'
    )
    
    parser.add_argument(
        '--compression_lengths', nargs='+', type=int,
        default=None,
        help='Compression lengths to test (default: 100 500 1000 2000 3000 4000 5000)'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help='Directory to save output plots'
    )
    
    parser.add_argument(
        '--ground_truth_path', type=str, default=None,
        help='Path to load/save ground truth JSON file'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.use_gpu:
        success = GPUConfig.enable_gpu()
        if not success:
            print("Warning: GPU not available, continuing with CPU")
    else:
        print("Running on CPU (use --use_gpu to enable GPU acceleration)")
    
    invalid_algos = [a for a in args.algo if a not in ALGO_MAP]
    if invalid_algos:
        print(f"Error: Unknown algorithms: {invalid_algos}")
        print(f"Available algorithms: {list(ALGO_MAP.keys())}")
        return
    
    compression_lengths = (
        args.compression_lengths
        if args.compression_lengths is not None
        else DEFAULT_COMPRESSION_LENGTHS
    )
    
    run_experiment1(
        data_path=args.data_path,
        algorithms=args.algo,
        thresholds=args.threshold,
        similarity_score=args.similarity_score,
        eval_metric=args.eval_metric,
        compression_lengths=compression_lengths,
        output_dir=args.output_dir,
        seed=args.seed,
        ground_truth_path=args.ground_truth_path
    )


if __name__ == '__main__':
    main()
