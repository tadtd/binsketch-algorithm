"""Utility functions for experiments.

This module contains all shared utilities for both Experiment 1 and Experiment 2.
All experiment-related functions have been moved here from src.utils and experiment2.py.
"""
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.metric import mse, minus_log_mse, precision_score, recall_score, f1_score, accuracy_score
from src.similarity_scores import cosine_similarity, jaccard_similarity, inner_product
from src.similarity_scores import batch_inner_product, batch_cosine_similarity, batch_jaccard_similarity
from src.gpu_utils import GPUConfig, to_gpu, to_cpu, get_array_module


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
    Get the evaluation metric function for Experiment 1.
    
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
    Create and save accuracy plot for Experiment 1.
    
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
    import os
    
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


def plot_experiment2_results(
    results: Dict,
    output_dir: str = "results/experiment2"
) -> str:
    """
    Create and save multi-threshold subplot visualization for Experiment 2.
    Creates a grid of subplots, one for each threshold value.
    
    Args:
        results: Results dictionary from run_experiment2
        output_dir: Output directory for plots
        
    Returns:
        Path to saved plot file
    """
    import os
    import math
    
    experiments = results['experiments']
    n_thresholds = len(experiments)
    
    if n_thresholds == 0:
        print("No results to plot")
        return None
    
    # Determine grid layout
    n_cols = min(4, n_thresholds)
    n_rows = math.ceil(n_thresholds / n_cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # Flatten axes for easy iteration
    if n_thresholds == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_thresholds > 1 else [axes]
    
    # Extract dataset name
    dataset_name = Path(results['data_path']).stem.replace('_binary', '')
    similarity_score = results['similarity_score']
    retrieval_metric = results['retrieval_metric']
    
    # Plot each threshold
    for idx, exp_data in enumerate(experiments):
        ax = axes[idx]
        threshold = exp_data['threshold']
        
        # Extract data for each algorithm
        for algo_name, algo_data in exp_data['algorithms'].items():
            compression_lengths = sorted(algo_data.keys())
            metric_values = [algo_data[k][retrieval_metric] for k in compression_lengths]
            
            ax.plot(compression_lengths, metric_values, marker='o', label=algo_name, linewidth=2)
        
        # Set labels and title
        ax.set_title(f"threshold = {threshold}", fontsize=11)
        ax.set_xlabel("Compression Length", fontsize=10)
        ax.set_ylabel(retrieval_metric.capitalize(), fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    # Hide unused subplots
    for idx in range(n_thresholds, len(axes)):
        axes[idx].set_visible(False)
    
    # Overall title
    fig.suptitle(
        f"Experiments on {dataset_name.upper()} to calculate {retrieval_metric.capitalize()} using {similarity_score.replace('_', ' ').title()}",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{dataset_name}_{similarity_score}_{retrieval_metric}_all_thresholds.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {filepath}")
    
    return filepath


def plot_experiment1_results(
    all_results: Dict,
    compression_lengths: List[int],
    similarity_score: str,
    eval_metric: str,
    dataset_name: str,
    output_dir: str = "results/experiment1"
) -> str:
    """
    Create and save multi-threshold subplot visualization for Experiment 1.
    Creates a grid of subplots, one for each threshold value.
    
    Args:
        all_results: Dictionary mapping thresholds to {algo_name: scores}
        compression_lengths: List of compression lengths
        similarity_score: Similarity metric name
        eval_metric: Evaluation metric name
        dataset_name: Name of the dataset
        output_dir: Output directory for plots
        
    Returns:
        Path to saved plot file
    """
    import os
    import math
    
    thresholds = sorted(all_results.keys(), reverse=True)
    n_thresholds = len(thresholds)
    
    if n_thresholds == 0:
        print("No results to plot")
        return None
    
    # Determine grid layout
    n_cols = min(4, n_thresholds)
    n_rows = math.ceil(n_thresholds / n_cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # Flatten axes for easy iteration
    if n_thresholds == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_thresholds > 1 else [axes]
    
    # Determine y-axis label
    if eval_metric == 'minus_log_mse':
        ylabel = "-log(MSE)"
    else:
        ylabel = "MSE"
    
    # Plot each threshold
    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]
        results = all_results[threshold]
        
        # Plot each algorithm
        for algo_name, scores in results.items():
            ax.plot(compression_lengths, scores, marker='o', label=algo_name, linewidth=2)
        
        # Set labels and title
        ax.set_title(f"threshold = {threshold}", fontsize=11)
        ax.set_xlabel("Compression Length", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_thresholds, len(axes)):
        axes[idx].set_visible(False)
    
    # Overall title
    fig.suptitle(
        f"Experiments on {dataset_name.upper()} to calculate {ylabel} using {similarity_score.replace('_', ' ').title()}",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{dataset_name}_{similarity_score}_{eval_metric}_all_thresholds.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nCombined plot saved to {filepath}")
    
    return filepath


# ============================================================================
# Experiment 2 Utilities
# ============================================================================

def split_dataset(X: np.ndarray, train_ratio: float = 0.9, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and query sets.
    
    Args:
        X: Input data matrix
        train_ratio: Ratio for training set
        seed: Random seed
        
    Returns:
        Tuple of (X_train, X_query, train_indices, query_indices)
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    query_indices = indices[n_train:]
    
    X_train = X[train_indices]
    X_query = X[query_indices]
    
    print(f"Dataset split: {len(train_indices)} training, {len(query_indices)} query samples")
    
    return X_train, X_query, train_indices, query_indices


def compute_pairwise_similarities(X1: np.ndarray, X2: np.ndarray, similarity_score: str) -> np.ndarray:
    """
    Compute pairwise similarity matrix between two sets of vectors.
    Uses GPU-accelerated batch operations when available.
    
    Args:
        X1: First set of vectors (n1, d)
        X2: Second set of vectors (n2, d)
        similarity_score: Type of similarity metric
        
    Returns:
        Similarity matrix (n1, n2)
    """
    use_gpu = GPUConfig.is_enabled()
    
    if use_gpu:
        # GPU-accelerated batch computation
        print(f"Computing {similarity_score} with GPU acceleration...")
        X1_gpu = to_gpu(X1.astype(np.float32))
        X2_gpu = to_gpu(X2.astype(np.float32))
        xp = get_array_module(X1_gpu)
        
        # Compute in batches to manage memory
        batch_size = 100
        n1 = X1.shape[0]
        similarities = xp.zeros((n1, X2.shape[0]), dtype=xp.float32)
        
        for i in tqdm(range(0, n1, batch_size), desc=f"Computing {similarity_score}", unit="batch"):
            batch_end = min(i + batch_size, n1)
            X1_batch = X1_gpu[i:batch_end]
            
            if similarity_score == 'inner_product':
                similarities[i:batch_end] = batch_inner_product(X1_batch, X2_gpu)
            elif similarity_score == 'cosine_similarity':
                similarities[i:batch_end] = batch_cosine_similarity(X1_batch, X2_gpu)
            elif similarity_score == 'jaccard_similarity':
                similarities[i:batch_end] = batch_jaccard_similarity(X1_batch, X2_gpu)
            else:
                raise ValueError(f"Unknown similarity_score: {similarity_score}")
        
        return to_cpu(similarities)
    else:
        # CPU fallback
        print(f"Computing {similarity_score} (CPU)...")
        similarity_func = {
            'inner_product': inner_product,
            'cosine_similarity': cosine_similarity,
            'jaccard_similarity': jaccard_similarity
        }.get(similarity_score)
        
        if similarity_func is None:
            raise ValueError(f"Unknown similarity_score: {similarity_score}")
        
        n1, n2 = X1.shape[0], X2.shape[0]
        similarities = np.zeros((n1, n2))
        
        for i in tqdm(range(n1), desc=f"Computing {similarity_score}"):
            for j in range(n2):
                similarities[i, j] = similarity_func(X1[i], X2[j])
        
        return similarities


def find_ground_truth_neighbors(
    X_query: np.ndarray,
    X_train: np.ndarray,
    threshold: float,
    similarity_score: str
) -> List[List[int]]:
    """
    Find all neighbors above threshold for each query point.
    
    Args:
        X_query: Query vectors
        X_train: Training vectors
        threshold: Similarity threshold
        similarity_score: Type of similarity metric
        
    Returns:
        List of neighbor lists for each query point
    """
    print(f"Computing ground truth neighbors (threshold={threshold})...")
    similarities = compute_pairwise_similarities(X_query, X_train, similarity_score)
    
    neighbors = []
    for i in range(len(X_query)):
        neighbor_indices = np.where(similarities[i] >= threshold)[0]
        neighbors.append(neighbor_indices.tolist())
    
    print(f"Average neighbors per query: {np.mean([len(n) for n in neighbors]):.2f}")
    
    return neighbors


def compress_and_retrieve(
    model,
    X_train: np.ndarray,
    X_query: np.ndarray,
    algo_name: str,
    threshold: float,
    similarity_score: str,
    k: int
) -> List[List[int]]:
    """
    Compress training and query data, then retrieve neighbors for query points.
    Uses mapping() to compress data (no fit() needed).
    
    Args:
        model: Model instance
        X_train: Training vectors
        X_query: Query vectors
        algo_name: Algorithm name
        threshold: Similarity threshold
        similarity_score: Type of similarity metric
        k: Compression length
        
    Returns:
        List of retrieved neighbor lists for each query point
    """
    from scipy.sparse import csr_matrix
    
    # Convert to sparse if needed
    X_train_csr = csr_matrix(X_train) if not isinstance(X_train, csr_matrix) else X_train
    X_query_csr = csr_matrix(X_query) if not isinstance(X_query, csr_matrix) else X_query
    
    # Compress both training and query data using mapping
    print(f"  Compressing data...")
    sketch_train = model.mapping(X_train_csr, k=k)
    sketch_query = model.mapping(X_query_csr, k=k)
    
    # Get the estimator function
    estimator = get_estimator(model, algo_name, similarity_score)
    
    # GPU acceleration: transfer sketches to GPU if enabled
    use_gpu = GPUConfig.is_enabled()
    if use_gpu:
        sketch_train = to_gpu(sketch_train)
        sketch_query = to_gpu(sketch_query)
        print(f"  Computing similarities (GPU)...")
    else:
        print(f"  Computing similarities...")
    
    # Compute similarity matrix using estimator
    n_query = len(sketch_query)
    n_train = len(sketch_train)
    
    if use_gpu:
        # GPU
        xp = get_array_module(sketch_train)
        similarity_matrix = xp.zeros((n_query, n_train), dtype=xp.float32)
        
        for i in tqdm(range(n_query), desc="Queries", leave=False):
            for j in range(n_train):
                similarity_matrix[i, j] = estimator(sketch_query[i], sketch_train[j])
        
        similarity_matrix = to_cpu(similarity_matrix)
    else:
        # CPU
        similarity_matrix = np.zeros((n_query, n_train))
        
        for i in tqdm(range(n_query), desc="Queries", leave=False):
            for j in range(n_train):
                similarity_matrix[i, j] = estimator(sketch_query[i], sketch_train[j])
    
    # Extract neighbors above threshold
    print(f"  Retrieving neighbors...")
    retrieved = []
    for i in range(len(X_query)):
        neighbors = np.where(similarity_matrix[i] >= threshold)[0].tolist()
        retrieved.append(neighbors)
    
    return retrieved


def compute_retrieval_metrics(
    ground_truth: List[List[int]], 
    retrieved: List[List[int]], 
    metric: str = 'precision'
) -> Dict[str, float]:
    """
    Compute retrieval metrics.
    
    Args:
        ground_truth: List of ground truth neighbor lists
        retrieved: List of retrieved neighbor lists
        metric: Metric to compute ('precision', 'recall', 'f1', 'accuracy')
        
    Returns:
        Dictionary with mean and std of the metric
    """
    scores = []
    
    for gt, ret in zip(ground_truth, retrieved):
        gt_set = set(gt)
        ret_set = set(ret)
        
        # Count true positives, false positives, false negatives
        tp = len(gt_set & ret_set)
        fp = len(ret_set - gt_set)
        fn = len(gt_set - ret_set)
        
        # Compute metric
        if metric == 'precision':
            score = precision_score(tp, fp) if len(ret_set) > 0 else (0.0 if len(gt_set) > 0 else 1.0)
        elif metric == 'recall':
            score = recall_score(tp, fn) if len(gt_set) > 0 else 1.0
        elif metric == 'f1':
            score = f1_score(tp, fp, fn)
        elif metric == 'accuracy':
            score = accuracy_score(tp, fp, fn)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    return {
        f'{metric}': np.mean(scores),
        f'{metric}_std': np.std(scores)
    }


def find_max_similarity(X: np.ndarray, similarity_score: str, sample_size: int = 1000) -> float:
    """
    Find maximum similarity value in dataset (sampled for efficiency).
    Uses similarity functions from src module.
    
    Args:
        X: Data matrix
        similarity_score: Type of similarity metric
        sample_size: Number of random pairs to sample
        
    Returns:
        Maximum similarity value
    """
    n = X.shape[0]
    
    # Sample random pairs
    np.random.seed(42)
    n_samples = min(sample_size, n * (n - 1) // 2)
    
    # Map similarity score to function from src
    similarity_func = {
        'inner_product': inner_product,
        'cosine_similarity': cosine_similarity,
        'jaccard_similarity': jaccard_similarity
    }.get(similarity_score)
    
    if similarity_func is None:
        raise ValueError(f"Unknown similarity_score: {similarity_score}")
    
    max_sim = 0
    for _ in tqdm(range(n_samples), desc="Finding max similarity"):
        i, j = np.random.choice(n, 2, replace=False)
        sim = similarity_func(X[i], X[j])
        max_sim = max(max_sim, sim)
    
    print(f"Maximum {similarity_score}: {max_sim}")
    return max_sim


# ============================================================================
# Ground Truth Utilities
# ============================================================================

def save_experiment2_ground_truth(
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    similarity_score: str,
    similarities: np.ndarray,
    data_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Save Experiment 2 ground truth similarity matrix to JSON file.
    
    Args:
        train_indices: Training indices
        query_indices: Query indices
        similarity_score: Similarity metric name
        similarities: Similarity matrix (n_query, n_train)
        data_path: Original data path
        output_path: Output JSON file path (default: auto-generated)
        
    Returns:
        Path to saved JSON file
    """
    import json
    import os
    
    dataset_name = Path(data_path).stem.replace('_binary', '')
    
    # Generate filepath if not provided
    if output_path is None:
        filename = f"ground_truth_exp2_{dataset_name}_{similarity_score}.json"
    else:
        # Replace template placeholders if present
        filename = output_path.replace('{DATASET}', dataset_name).replace('{SIMILARITY_SCORE}', similarity_score).replace('{SIMILARITY}', similarity_score)
    
    # Always place in experiment/ground_truth/ directory
    filepath = f"experiment/ground_truth/{filename}"
    
    # Create parent directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    data = {
        'data_path': data_path,
        'similarity_score': similarity_score,
        'n_train': len(train_indices),
        'n_query': len(query_indices),
        'train_indices': train_indices.tolist(),
        'query_indices': query_indices.tolist(),
        'similarities': similarities.tolist()
    }
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Ground truth saved to {filepath}")
    return filepath


def load_experiment2_ground_truth(
    data_path: str,
    similarity_score: str,
    ground_truth_path: Optional[str] = None
) -> Optional[Dict]:
    """
    Load Experiment 2 ground truth similarity matrix from JSON file if it exists.
    
    Args:
        data_path: Original data path
        similarity_score: Similarity metric name
        ground_truth_path: Path to ground truth JSON file (default: auto-generated)
        
    Returns:
        Dictionary with ground truth data or None if file doesn't exist
    """
    import json
    import os
    
    dataset_name = Path(data_path).stem.replace('_binary', '')
    
    # Generate filepath if not provided
    if ground_truth_path is None:
        filename = f"ground_truth_exp2_{dataset_name}_{similarity_score}.json"
    else:
        # Replace template placeholders if present
        filename = ground_truth_path.replace('{DATASET}', dataset_name).replace('{SIMILARITY_SCORE}', similarity_score).replace('{SIMILARITY}', similarity_score)
    
    # Always place in experiment/ground_truth/ directory
    filepath = f"experiment/ground_truth/{filename}"
    
    if not os.path.exists(filepath):
        return None
    
    # Load from JSON
    print(f"Loading ground truth from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    
    # Convert lists back to numpy arrays where needed
    data['train_indices'] = np.array(data['train_indices'])
    data['query_indices'] = np.array(data['query_indices'])
    data['similarities'] = np.array(data['similarities'])
    
    print(f"  Loaded similarity matrix ({data['n_query']}, {data['n_train']})")
    
    return data
