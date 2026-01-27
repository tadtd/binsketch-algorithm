"""Utility functions for experiments."""
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .metric import mse, minus_log_mse


def save_compression_matrix(matrix: np.ndarray, output_path: str):
    """Save compression matrix to file."""
    np.save(output_path, matrix)


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


def get_estimator(model, algo_name: str, similarity_score: str) -> Callable:
    """
    Get the appropriate similarity estimator function for a model.
    
    Args:
        model: Model instance
        algo_name: Name of the algorithm
        similarity_score: Similarity metric name
        
    Returns:
        Estimator function bound to the model instance
    """
    if similarity_score == 'cosine_similarity':
        if hasattr(model, 'estimate_cosine_similarity'):
            return model.estimate_cosine_similarity
        else:
            raise AttributeError(
                f"Model {algo_name} does not have 'estimate_cosine_similarity' method."
            )
    elif similarity_score == 'jaccard_similarity':
        if hasattr(model, 'estimate_jaccard_similarity'):
            return model.estimate_jaccard_similarity
        else:
            raise AttributeError(
                f"Model {algo_name} does not have 'estimate_jaccard_similarity' method."
            )
    elif similarity_score == 'inner_product':
        if hasattr(model, 'estimate_inner_product'):
            return model.estimate_inner_product
        else:
            raise AttributeError(
                f"Model {algo_name} does not have 'estimate_inner_product' method."
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
    
    for i, j in tqdm(pairs, desc=f"Estimating {algo_name}", unit="pair", leave=False):
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
        try:
            X_sketch = model.mapping(X_csr, k=N)
        except Exception as e:
            raise RuntimeError(f"Model {algo_name} failed to compress with k={N}: {e}")
        
        est_vals = estimate_similarities(
            model, X_sketch, pairs_filtered, algo_name, similarity_score
        )
        
        score = eval_func(est_vals, gt_filtered)
        
        if eval_metric == 'minus_log_mse' and np.isinf(score):
            score = -np.log(1e-10)
        
        scores.append(score)
    
    return scores


def save_plot(
    compression_lengths: List[int],
    results: Dict[str, List[float]],
    similarity_score: str,
    eval_metric: str,
    threshold: float,
    output_dir: str,
    dataset_name: str = None
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
        dataset_name: Name of the dataset (optional)
        
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(10, 6))
    
    for algo_name, scores in results.items():
        plt.plot(compression_lengths, scores, marker='o', label=algo_name)
    
    if eval_metric == 'minus_log_mse':
        ylabel = "-log(MSE)"
        title_suffix = "Accuracy"
    else:
        ylabel = "MSE"
        title_suffix = "Error"
    
    dataset_prefix = f"{dataset_name.upper()} - " if dataset_name else ""
    plt.title(f"{dataset_prefix}{similarity_score.replace('_', ' ').title()} {title_suffix} (Threshold={threshold})")
    plt.xlabel("Compression Length (N)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    threshold_str = str(threshold).replace('.', '_')
    filename = f"result_{dataset_name}_{similarity_score}_{eval_metric}_t{threshold_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath
