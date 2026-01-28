"""Experiment 2: Retrieval Performance Evaluation.

This experiment evaluates the retrieval performance of sketching algorithms
by splitting data into training and query sets, compressing the training data,
and measuring precision, recall, F1, and accuracy.

All utility functions are imported from experiment.utils module.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from src import BinSketch, BinaryCompressionSchema, MinHash, SimHash
from src.gpu_utils import GPUConfig


from experiment.utils import (
    load_data,
    split_dataset,
    compute_pairwise_similarities,
    compress_and_retrieve,
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
    # Enable GPU if requested
    if GPUConfig.is_enabled():
        print("GPU acceleration enabled")
    
    # Load data
    X_dense, _ = load_data(data_path)
    
    # Split dataset
    X_train, X_query, train_indices, query_indices = split_dataset(
        X_dense, train_ratio=train_ratio, seed=seed
    )
    
    # Store results
    results = {
        'data_path': data_path,
        'similarity_score': similarity_score,
        'retrieval_metric': retrieval_metric,
        'train_ratio': train_ratio,
        'n_train': len(train_indices),
        'n_query': len(query_indices),
        'seed': seed,
        'experiments': []
    }
    
    # Run experiments for each threshold
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Threshold: {threshold}")
        print(f"{'='*60}")
        
        # Try to load ground truth from cache
        gt_data = load_experiment2_ground_truth(data_path, similarity_score, ground_truth_path)
        
        if gt_data is not None:
            # Verify indices match
            if (np.array_equal(gt_data['train_indices'], train_indices) and 
                np.array_equal(gt_data['query_indices'], query_indices)):
                similarities = gt_data['similarities']
                print(f"Using cached ground truth similarity matrix")
            else:
                print(f"Cached ground truth indices mismatch, recomputing...")
                gt_data = None
        
        # Compute ground truth if not cached
        if gt_data is None:
            similarities = compute_pairwise_similarities(X_query, X_train, similarity_score)
            # Save for future use
            save_experiment2_ground_truth(
                train_indices, query_indices,
                similarity_score, similarities, data_path,
                output_path=ground_truth_path
            )
        
        # Find neighbors above threshold from similarity matrix
        ground_truth = []
        for i in range(len(X_query)):
            neighbors = np.where(similarities[i] >= threshold)[0].tolist()
            ground_truth.append(neighbors)
        
        threshold_results = {
            'threshold': threshold,
            'avg_neighbors': np.mean([len(n) for n in ground_truth]),
            'algorithms': {}
        }
        
        # Test each algorithm
        for algo_name in algorithms:
            print(f"\n{algo_name}:")
            algo_results = {}
            
            for k in compression_lengths:
                print(f"  Compression length k={k}")
                
                AlgoClass = ALGO_MAP[algo_name]
                model = AlgoClass(k=k, seed=seed)
                
                # Compress and retrieve using utility function
                retrieved = compress_and_retrieve(
                    model, X_train, X_query, algo_name, threshold, similarity_score
                )
                
                # Compute metrics
                metrics = compute_retrieval_metrics(ground_truth, retrieved, retrieval_metric)
                
                print(f"    {retrieval_metric.capitalize()}: {metrics[retrieval_metric]:.4f} ± {metrics[f'{retrieval_metric}_std']:.4f}")
                
                algo_results[k] = metrics
            
            threshold_results['algorithms'][algo_name] = algo_results
        
        results['experiments'].append(threshold_results)
    
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
        GPUConfig.enable()
    
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
