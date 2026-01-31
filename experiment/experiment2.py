"""Experiment 2: Retrieval Performance Evaluation.

This experiment evaluates the retrieval performance of sketching algorithms
by splitting data into training and query sets, compressing the training data,
and measuring precision, recall, F1, and accuracy.

All utility functions are imported from experiment.utils module.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from typing import List, Dict, Optional
import numpy as np

from src import BinSketch, BinaryCompressionSchema, MinHash, SimHash
from src.gpu_utils import GPUConfig


from experiment.utils import (
    load_data,
    split_dataset,
    compute_pairwise_similarities,
    compute_estimated_similarity_matrix,
    compute_retrieval_metrics,
    find_max_similarity,
    save_experiment2_ground_truth,
    load_experiment2_ground_truth,
    plot_experiment2_results
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


def run_experiment2(
    data_path: str,
    algorithms: List[str],
    thresholds: List[float],
    similarity_score: str,
    retrieval_metric: str,
    compression_lengths: List[int],
    train_ratio: float = 0.9,
    seed: int = 42,
    output_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None
) -> Dict:
    """
    Run Experiment 2: Retrieval Performance Evaluation.
    
    This experiment splits data into training and query sets, compresses the
    training data, and evaluates retrieval performance using precision, recall,
    F1, or accuracy metrics.
    
    OPTIMIZED: Computes sketches and similarity matrices once per (algorithm, k),
    then iterates through thresholds to avoid redundant computations.
    
    Args:
        data_path: Path to data file
        algorithms: List of algorithm names
        thresholds: List of similarity thresholds
        similarity_score: Similarity metric name
        retrieval_metric: Retrieval metric to evaluate ('precision', 'recall', 'f1', 'accuracy')
        compression_lengths: List of compression lengths
        train_ratio: Training set ratio
        seed: Random seed
        output_path: Optional path to save results
        ground_truth_path: Optional path to load/save ground truth JSON file
        
    Returns:
        Dictionary of results
    """
    from experiment.utils import compute_estimated_similarity_matrix
    
    # Enable GPU if requested
    if GPUConfig.is_enabled():
        print("GPU acceleration enabled")
    
    # Load data
    X_dense, _ = load_data(data_path)
    
    # Split dataset
    X_train, X_query, train_indices, query_indices = split_dataset(
        X_dense, train_ratio=train_ratio, seed=seed
    )
    
    # Store results - reorganized structure
    results = {
        'data_path': data_path,
        'similarity_score': similarity_score,
        'retrieval_metric': retrieval_metric,
        'train_ratio': train_ratio,
        'n_train': len(train_indices),
        'n_query': len(query_indices),
        'seed': seed,
        'experiments': []  # Will be populated per threshold
    }
    
    # =========================================================================
    # Step 1: Load or compute ground truth similarity matrix ONCE
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Loading/Computing Ground Truth Similarities")
    print(f"{'='*60}")
    
    gt_data = load_experiment2_ground_truth(data_path, similarity_score, ground_truth_path)
    
    if gt_data is not None:
        # Verify indices match
        if (np.array_equal(gt_data['train_indices'], train_indices) and 
            np.array_equal(gt_data['query_indices'], query_indices)):
            gt_similarities = gt_data['similarities']
            print(f"Using cached ground truth similarity matrix")
        else:
            print(f"Cached ground truth indices mismatch, recomputing...")
            gt_data = None
    
    if gt_data is None:
        gt_similarities = compute_pairwise_similarities(X_query, X_train, similarity_score)
        # Save for future use
        save_experiment2_ground_truth(
            train_indices, query_indices,
            similarity_score, gt_similarities, data_path,
            output_path=ground_truth_path
        )
    
    # =========================================================================
    # Step 2: Precompute ground truth neighbors for all thresholds
    # =========================================================================
    print(f"\nPrecomputing ground truth neighbors for {len(thresholds)} thresholds...")
    ground_truth_per_threshold = {}
    for threshold in thresholds:
        ground_truth = []
        for i in range(len(X_query)):
            neighbors = np.where(gt_similarities[i] >= threshold)[0].tolist()
            ground_truth.append(neighbors)
        ground_truth_per_threshold[threshold] = ground_truth
        avg_neighbors = np.mean([len(n) for n in ground_truth])
        print(f"  Threshold {threshold}: avg {avg_neighbors:.2f} neighbors per query")
    
    # =========================================================================
    # Step 3: Initialize results structure per threshold
    # =========================================================================
    threshold_results_map = {}
    for threshold in thresholds:
        ground_truth = ground_truth_per_threshold[threshold]
        threshold_results_map[threshold] = {
            'threshold': threshold,
            'avg_neighbors': np.mean([len(n) for n in ground_truth]),
            'total_pairs': sum([len(n) for n in ground_truth]),
            'algorithms': {}
        }
    
    # =========================================================================
    # Step 4: For each algorithm and compression length, compute similarity 
    #         matrix ONCE, then evaluate all thresholds
    # =========================================================================
    for algo_name in algorithms:
        print(f"\n{'='*60}")
        print(f"Algorithm: {algo_name}")
        print(f"{'='*60}")
        
        # Initialize results storage for this algorithm across all thresholds
        for threshold in thresholds:
            threshold_results_map[threshold]['algorithms'][algo_name] = {}
        
        for k in compression_lengths:
            print(f"\n  Compression length k={k}")
            
            # Create model and compute sketches + similarity matrix ONCE
            AlgoClass = ALGO_MAP[algo_name]
            model = AlgoClass(seed=seed)
            
            # Compute estimated similarity matrix (sketches computed internally)
            est_similarity_matrix = compute_estimated_similarity_matrix(
                model, X_train, X_query, algo_name, similarity_score, k=k
            )
            
            # Now iterate through all thresholds using the pre-computed matrix
            for threshold in thresholds:
                ground_truth = ground_truth_per_threshold[threshold]
                
                # Extract neighbors above threshold from estimated similarity matrix
                retrieved = []
                for i in range(len(X_query)):
                    neighbors = np.where(est_similarity_matrix[i] >= threshold)[0].tolist()
                    retrieved.append(neighbors)
                
                # Compute metrics
                metrics = compute_retrieval_metrics(ground_truth, retrieved, retrieval_metric)
                
                print(f"    Threshold {threshold}: {retrieval_metric.capitalize()} = {metrics[retrieval_metric]:.4f}")
                
                # Store results
                threshold_results_map[threshold]['algorithms'][algo_name][k] = metrics
    
    # =========================================================================
    # Step 5: Organize results in the expected format (per threshold)
    # =========================================================================
    for threshold in thresholds:
        results['experiments'].append(threshold_results_map[threshold])
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    # Generate plot
    plot_experiment2_results(results, output_dir="results/experiment2")
    
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment 2: Retrieval Performance Evaluation',
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
        '--threshold', nargs='+', type=float, required=True,
        help='Similarity thresholds to evaluate'
    )
    
    parser.add_argument(
        '--similarity_score', type=str, default='cosine_similarity',
        choices=['cosine_similarity', 'jaccard_similarity', 'inner_product'],
        help='Similarity metric'
    )
    
    parser.add_argument(
        '--retrieval_metric', type=str, default='f1',
        choices=['precision', 'recall', 'f1', 'accuracy'],
        help='Retrieval metric to evaluate'
    )
    
    parser.add_argument(
        '--compression_lengths', nargs='+', type=int,
        default=None,
        help='Compression lengths to test'
    )
    
    parser.add_argument(
        '--train_ratio', type=float, default=0.9,
        help='Ratio of training data'
    )
    
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save results JSON file'
    )
    
    parser.add_argument(
        '--ground_truth_path', type=str, default=None,
        help='Path to load/save ground truth JSON file'
    )
    
    parser.add_argument(
        '--find_max_inner_product', action='store_true',
        help='Find and print maximum inner product value in dataset'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Configure GPU
    if args.use_gpu:
        GPUConfig.enable_gpu()
    
    # Find max inner product if requested
    if args.find_max_inner_product:
        X_dense, _ = load_data(args.data_path)
        max_val = find_max_similarity(X_dense, 'inner_product', sample_size=2000)
        print(f"\nSuggested thresholds for inner product:")
        for ratio in [0.95, 0.9, 0.8, 0.6, 0.5, 0.3, 0.2]:
            print(f"  {ratio:.2f} * max = {ratio * max_val:.2f}")
        return
    
    # Set default compression lengths
    compression_lengths = args.compression_lengths or DEFAULT_COMPRESSION_LENGTHS
    
    # Generate output path if not provided
    if args.output_path is None:
        dataset_name = Path(args.data_path).stem.replace('_binary', '')
        args.output_path = f"retrieval_results_{dataset_name}_{args.similarity_score}_{args.retrieval_metric}.json"
    
    # Run experiment
    run_experiment2(
        data_path=args.data_path,
        algorithms=args.algo,
        thresholds=args.threshold,
        similarity_score=args.similarity_score,
        retrieval_metric=args.retrieval_metric,
        compression_lengths=compression_lengths,
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_path=args.output_path,
        ground_truth_path=args.ground_truth_path
    )


if __name__ == '__main__':
    main()
