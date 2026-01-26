"""Main experiment runner for BinSketch algorithm evaluation."""
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.sparse import csr_matrix

from src import (
    BinSketch, BinaryCompressionSchema, MinHash, SimHash,
    cosine_similarity, jaccard_similarity, mse, minus_log_mse
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='BinSketch Experiment Runner',
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
        '--eval_metric', type=str, default='minus_log_mse',
        choices=['mse', 'minus_log_mse'],
        help='Evaluation metric for comparing estimates to ground truth'
    )
    
    parser.add_argument(
        '--similarity_score', type=str, default='cosine_similarity',
        choices=['cosine_similarity', 'jaccard_similarity'],
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
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def load_data(data_path: str) -> Tuple[np.ndarray, csr_matrix]:
    """
    Load binary matrix from file.
    
    Args:
        data_path: Path to .npy file containing binary matrix
        
    Returns:
        Tuple of (dense_matrix, sparse_matrix)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading matrix from {data_path}...")
    X_dense = np.load(data_path)
    X_csr = csr_matrix(X_dense)
    print(f"  Matrix Shape: {X_dense.shape}")
    
    return X_dense, X_csr


def calculate_ground_truth(
    X_dense: np.ndarray,
    similarity_score: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ground truth similarities for all pairs.
    
    Args:
        X_dense: Dense binary matrix (n_samples, n_features)
        similarity_score: Similarity metric ('cosine_similarity' or 'jaccard_similarity')
        
    Returns:
        Tuple of (ground_truth_values, pair_indices)
    """
    n_samples = X_dense.shape[0]
    pairs = list(combinations(range(n_samples), 2))
    
    print(f"Calculating Ground Truth ({similarity_score}) for {len(pairs)} pairs...")
    
    ground_truth = []
    for i, j in pairs:
        if similarity_score == 'cosine_similarity':
            val = cosine_similarity(X_dense[i], X_dense[j])
        else:
            val = jaccard_similarity(X_dense[i], X_dense[j])
        ground_truth.append(val)
    
    return np.array(ground_truth), np.array(pairs)


def get_estimator(model, algo_name: str, similarity_score: str) -> Callable:
    """
    Get the appropriate similarity estimator function for a model.
    Always uses the model's own estimation methods when available.
    
    Args:
        model: Model instance
        algo_name: Name of the algorithm
        similarity_score: Similarity metric name ('cosine_similarity' or 'jaccard_similarity')
        
    Returns:
        Estimator function bound to the model instance
        
    Raises:
        AttributeError: If the model doesn't have the required estimation method
    """
    # Always prioritize model's own estimation methods for sketched data
    if similarity_score == 'cosine_similarity':
        if hasattr(model, 'estimate_cosine_similarity'):
            return model.estimate_cosine_similarity
        else:
            raise AttributeError(
                f"Model {algo_name} does not have 'estimate_cosine_similarity' method. "
                f"Cannot estimate cosine similarity from sketches."
            )
    elif similarity_score == 'jaccard_similarity':
        if hasattr(model, 'estimate_jaccard_similarity'):
            return model.estimate_jaccard_similarity
        else:
            raise AttributeError(
                f"Model {algo_name} does not have 'estimate_jaccard_similarity' method. "
                f"Cannot estimate jaccard similarity from sketches."
            )
    else:
        raise ValueError(f"Unknown similarity_score: {similarity_score}")


def estimate_similarities(
    model,
    sketch_data: np.ndarray,
    pairs: np.ndarray,
    algo_name: str,
    similarity_score: str
) -> np.ndarray:
    """
    Estimate similarities for pairs using sketches.
    
    Args:
        model: Model instance
        sketch_data: Compressed sketch data (n_samples, k)
        pairs: Array of pair indices (n_pairs, 2)
        algo_name: Name of the algorithm
        similarity_score: Similarity metric name
        
    Returns:
        Array of estimated similarity values
    """
    estimator = get_estimator(model, algo_name, similarity_score)
    estimates = []
    
    for i, j in pairs:
        u, v = sketch_data[i], sketch_data[j]
        est = estimator(u, v)
        estimates.append(est)
    
    return np.array(estimates)


def get_evaluation_metric(eval_metric: str) -> Callable:
    """
    Get the evaluation metric function.
    
    Args:
        eval_metric: Evaluation metric name ('mse' or 'minus_log_mse')
        
    Returns:
        Evaluation metric function
    """
    if eval_metric == 'mse':
        return mse
    elif eval_metric == 'minus_log_mse':
        return minus_log_mse
    else:
        raise ValueError(f"Unknown evaluation metric: {eval_metric}")


def run_algorithm_experiment(
    model,
    X_csr: csr_matrix,
    pairs_filtered: np.ndarray,
    gt_filtered: np.ndarray,
    compression_lengths: List[int],
    algo_name: str,
    similarity_score: str,
    eval_metric: str
) -> List[float]:
    """
    Run experiment for a single algorithm across compression lengths.
    
    Args:
        model: Model instance
        X_csr: Sparse input matrix
        pairs_filtered: Filtered pair indices
        gt_filtered: Filtered ground truth values
        compression_lengths: List of compression lengths to test
        algo_name: Name of the algorithm
        similarity_score: Similarity metric name
        eval_metric: Evaluation metric name
        
    Returns:
        List of evaluation scores for each compression length
    """
    scores = []
    eval_func = get_evaluation_metric(eval_metric)
    
    for N in compression_lengths:
        # Compress data
        try:
            X_sketch = model.mapping(X_csr, k=N)
        except Exception as e:
            raise RuntimeError(f"Model {algo_name} failed to compress with k={N}: {e}")
        
        # Estimate similarities
        est_vals = estimate_similarities(
            model, X_sketch, pairs_filtered, algo_name, similarity_score
        )
        
        # Calculate evaluation metric
        score = eval_func(est_vals, gt_filtered)
        
        # Handle infinite values (when MSE is 0) for minus_log_mse
        if eval_metric == 'minus_log_mse' and np.isinf(score):
            score = -np.log(1e-10)
        
        scores.append(score)
    
    return scores


def filter_pairs_by_threshold(
    ground_truth: np.ndarray,
    pair_indices: np.ndarray,
    threshold: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filter pairs by similarity threshold.
    
    Args:
        ground_truth: Ground truth similarity values
        pair_indices: Pair indices
        threshold: Similarity threshold
        
    Returns:
        Tuple of (filtered_ground_truth, filtered_pair_indices) or (None, None) if no pairs
    """
    valid_mask = ground_truth > threshold
    if np.sum(valid_mask) == 0:
        return None, None
    
    return ground_truth[valid_mask], pair_indices[valid_mask]


def save_plot(
    compression_lengths: List[int],
    results: Dict[str, List[float]],
    similarity_score: str,
    eval_metric: str,
    threshold: float,
    output_dir: str
) -> str:
    """
    Create and save accuracy plot.
    
    Args:
        compression_lengths: List of compression lengths
        results: Dictionary mapping algorithm names to scores
        similarity_score: Similarity metric name
        eval_metric: Evaluation metric name
        threshold: Threshold value
        output_dir: Output directory
        
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(10, 6))
    
    for algo_name, scores in results.items():
        plt.plot(compression_lengths, scores, marker='o', label=algo_name)
    
    # Create title and labels based on evaluation metric
    if eval_metric == 'minus_log_mse':
        ylabel = "-log(MSE)"
        title_suffix = "Accuracy"
    else:
        ylabel = "MSE"
        title_suffix = "Error"
    
    plt.title(f"{similarity_score.replace('_', ' ').title()} {title_suffix} (Threshold={threshold})")
    plt.xlabel("Compression Length (N)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    threshold_str = str(threshold).replace('.', '_')
    filename = f"result_{similarity_score}_{eval_metric}_t{threshold_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def run_experiment(
    data_path: str,
    algorithms: List[str],
    thresholds: List[float],
    similarity_score: str,
    eval_metric: str,
    compression_lengths: List[int],
    output_dir: str,
    seed: int = 42
) -> None:
    """
    Run the complete experiment pipeline.
    
    Args:
        data_path: Path to data file
        algorithms: List of algorithm names to test
        thresholds: List of similarity thresholds
        similarity_score: Similarity metric name
        eval_metric: Evaluation metric name
        compression_lengths: List of compression lengths to test
        output_dir: Output directory for results
        seed: Random seed
    """
    # Load data
    X_dense, X_csr = load_data(data_path)
    
    # Calculate ground truth
    ground_truth, pair_indices = calculate_ground_truth(X_dense, similarity_score)
    
    # Run experiments for each threshold
    for threshold in thresholds:
        print(f"\n{'='*40}")
        print(f"Processing Threshold {threshold}")
        print(f"{'='*40}")
        
        # Filter pairs by threshold
        gt_filtered, pairs_filtered = filter_pairs_by_threshold(
            ground_truth, pair_indices, threshold
        )
        
        if gt_filtered is None:
            print(f"  [Skipping] No pairs found > {threshold}")
            continue
        
        print(f"  Valid Pairs: {len(gt_filtered)}")
        
        # Run each algorithm
        results = {}
        for algo_name in algorithms:
            if algo_name not in ALGO_MAP:
                print(f"  [Warning] Unknown algorithm: {algo_name}, skipping...")
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
                print(f"  [Error] Failed to run {algo_name}: {e}")
                continue
        
        # Save plot
        if results:
            filepath = save_plot(
                compression_lengths, results, similarity_score, eval_metric, threshold, output_dir
            )
            print(f"  Saved: {filepath}")
        else:
            print(f"  [Warning] No results to plot for threshold {threshold}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate algorithms
    invalid_algos = [a for a in args.algo if a not in ALGO_MAP]
    if invalid_algos:
        print(f"Error: Unknown algorithms: {invalid_algos}")
        print(f"Available algorithms: {list(ALGO_MAP.keys())}")
        return
    
    # Set compression lengths
    compression_lengths = (
        args.compression_lengths
        if args.compression_lengths is not None
        else DEFAULT_COMPRESSION_LENGTHS
    )
    
    # Run experiment
    run_experiment(
        data_path=args.data_path,
        algorithms=args.algo,
        thresholds=args.threshold,
        similarity_score=args.similarity_score,
        eval_metric=args.eval_metric,
        compression_lengths=compression_lengths,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
