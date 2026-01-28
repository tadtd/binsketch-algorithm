"""Utility to compute and save ground truth similarities to JSON file.

Supports two modes:
1. Experiment 1: All pairs ground truth
2. Experiment 2: Train/query split with neighbors above threshold
"""
import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
from itertools import combinations
from tqdm import tqdm

from src import batch_inner_product, batch_cosine_similarity, batch_jaccard_similarity
from src.gpu_utils import to_gpu, to_cpu, get_array_module


def _get_similarity_function(similarity_score: str):
    """
    Get similarity function from src.similarity_scores module.
    
    Args:
        similarity_score: Similarity metric name
        
    Returns:
        Similarity function
    """
    similarity_func = {
        'inner_product': inner_product,
        'cosine_similarity': cosine_similarity,
        'jaccard_similarity': jaccard_similarity
    }.get(similarity_score)
    
    if similarity_func is None:
        raise ValueError(f"Unknown similarity score: {similarity_score}")
    
    return similarity_func


def calculate_experiment1_ground_truth(
    X_dense: np.ndarray,
    similarity_score: str,
    use_gpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ground truth similarities for all pairs (Experiment 1).
    Uses vectorized GPU operations for maximum performance.
    
    Args:
        X_dense: Dense binary matrix (n_samples, n_features)
        similarity_score: Similarity metric ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        use_gpu: Whether to use GPU for computations
        
    Returns:
        Tuple of (ground_truth_values, pair_indices)
    """
    # Transfer to GPU if requested
    if use_gpu:
        print("Transferring data to GPU...")
        X_dense = to_gpu(X_dense)
        print("✓ Data transferred to GPU")
    
    xp = get_array_module(X_dense)
    n_samples = X_dense.shape[0]
    pairs = list(combinations(range(n_samples), 2))
    n_pairs = len(pairs)
    
    device = "GPU" if use_gpu else "CPU"
    print(f"Calculating Ground Truth ({similarity_score}) for {n_pairs} pairs on {device}...")
    
    # Compute full similarity matrix using vectorized batch operations
    print("Computing similarity matrix with vectorized operations...")
    if similarity_score == 'inner_product':
        sim_matrix = batch_inner_product(X_dense, X_dense)
    elif similarity_score == 'cosine_similarity':
        sim_matrix = batch_cosine_similarity(X_dense, X_dense)
    elif similarity_score == 'jaccard_similarity':
        sim_matrix = batch_jaccard_similarity(X_dense, X_dense)
    else:
        raise ValueError(f"Unknown similarity score: {similarity_score}")
    
    # Extract upper triangle (all pairs)
    print("Extracting pair similarities...")
    pairs_array = xp.array(pairs)
    ground_truth = sim_matrix[pairs_array[:, 0], pairs_array[:, 1]]
    
    # Transfer to CPU
    ground_truth = to_cpu(ground_truth)
    
    return ground_truth, np.array(pairs)


def save_experiment1_ground_truth(
    data_path: str,
    similarity_score: str,
    ground_truth: np.ndarray,
    pairs: np.ndarray,
    output_path: str = None
) -> str:
    """
    Save Experiment 1 ground truth data to JSON file.
    
    Args:
        data_path: Path to original data file
        similarity_score: Similarity metric used
        ground_truth: Ground truth values array
        pairs: Pairs array
        output_path: Output JSON file path (default: auto-generated)
        
    Returns:
        Path to saved JSON file
    """
    # Load data for metadata
    X_dense = np.load(data_path)
    n_samples, n_features = X_dense.shape
    n_pairs = len(pairs)
    
    # Generate output path if not provided
    if output_path is None:
        dataset_name = Path(data_path).stem
        output_path = f"ground_truth_{dataset_name}_{similarity_score}.json"
    
    # Prepare data for JSON
    data_to_save = {
        "data_path": data_path,
        "similarity_score": similarity_score,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_pairs": int(n_pairs),
        "pairs": pairs.tolist(),
        "ground_truth": ground_truth.tolist()
    }
    
    # Save to JSON
    print(f"\nSaving ground truth to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"✓ Saved {n_pairs} ground truth values")
    print(f"  Min similarity: {ground_truth.min():.6f}")
    print(f"  Max similarity: {ground_truth.max():.6f}")
    print(f"  Mean similarity: {ground_truth.mean():.6f}")
    
    return output_path


def calculate_and_save_experiment1_ground_truth(
    data_path: str,
    similarity_score: str,
    output_path: str = None,
    use_gpu: bool = False
) -> str:
    """
    Calculate Experiment 1 ground truth similarities and save to JSON.
    
    Args:
        data_path: Path to .npy file containing binary matrix
        similarity_score: Similarity metric ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        output_path: Output JSON file path (default: auto-generated)
        use_gpu: Whether to use GPU for computations
        
    Returns:
        Path to saved JSON file
    """
    # Load data
    print(f"Loading matrix from {data_path}...")
    X_dense = np.load(data_path)
    print(f"  Matrix Shape: {X_dense.shape}")
    
    # Calculate ground truth
    ground_truth, pairs = calculate_experiment1_ground_truth(X_dense, similarity_score, use_gpu)
    
    # Save to file
    return save_experiment1_ground_truth(data_path, similarity_score, ground_truth, pairs, output_path)


def load_experiment1_ground_truth(json_path: str):
    """
    Load Experiment 1 ground truth from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        dict with keys: data_path, similarity_score, n_samples, n_features, n_pairs, pairs, ground_truth
    """
    print(f"Loading ground truth from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert back to numpy arrays
    data['pairs'] = np.array(data['pairs'])
    data['ground_truth'] = np.array(data['ground_truth'])
    
    print(f"✓ Loaded {data['n_pairs']} ground truth values")
    print(f"  Dataset: {data['data_path']}")
    print(f"  Similarity: {data['similarity_score']}")
    print(f"  Shape: ({data['n_samples']}, {data['n_features']})")
    
    return data


def calculate_experiment2_ground_truth(
    X_train: np.ndarray,
    X_query: np.ndarray,
    similarity_score: str,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Calculate ground truth similarities for Experiment 2 (train/query split).
    Computes similarity matrix between query and training vectors.
    Uses vectorized GPU operations for maximum performance.
    
    Args:
        X_train: Training vectors (n_train, n_features)
        X_query: Query vectors (n_query, n_features)
        similarity_score: Similarity metric
        use_gpu: Whether to use GPU for computations
        
    Returns:
        Similarity matrix (n_query, n_train)
    """
    # Transfer to GPU if requested
    if use_gpu:
        print("Transferring data to GPU...")
        X_train = to_gpu(X_train)
        X_query = to_gpu(X_query)
        print("✓ Data transferred to GPU")
    
    device = "GPU" if use_gpu else "CPU"
    print(f"Calculating Experiment 2 Ground Truth ({similarity_score}) on {device}...")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Query samples: {X_query.shape[0]}")
    
    xp = get_array_module(X_train)
    
    # Compute full similarity matrix using vectorized operations
    print("Computing similarity matrix with GPU acceleration...")
    if similarity_score == 'inner_product':
        # Inner product: Q @ T.T
        similarities = xp.dot(X_query, X_train.T)
    elif similarity_score == 'cosine_similarity':
        # Cosine: (Q @ T.T) / (||Q|| * ||T||.T)
        query_norms = xp.sqrt(xp.sum(X_query ** 2, axis=1, keepdims=True))
        train_norms = xp.sqrt(xp.sum(X_train ** 2, axis=1, keepdims=True))
        query_norms = xp.maximum(query_norms, 1e-10)
    # Compute full similarity matrix using vectorized batch operations
    print("Computing similarity matrix with GPU acceleration...")
    if similarity_score == 'inner_product':
        similarities = batch_inner_product(X_query, X_train)
    elif similarity_score == 'cosine_similarity':
        similarities = batch_cosine_similarity(X_query, X_train)
    elif similarity_score == 'jaccard_similarity':
        similarities = batch_jaccard_similarity(X_query, X_train)

def save_experiment2_ground_truth(
    data_path: str,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    similarity_score: str,
    similarities: np.ndarray,
    output_path: str = None
) -> str:
    """
    Save Experiment 2 ground truth to JSON file.
    
    Args:
        data_path: Path to original data file
        train_indices: Training set indices
        query_indices: Query set indices
        similarity_score: Similarity metric
        similarities: Similarity matrix (n_query, n_train)
        output_path: Output JSON file path (default: auto-generated)
        
    Returns:
        Path to saved JSON file
    """
    # Load data for metadata
    X_dense = np.load(data_path)
    n_samples, n_features = X_dense.shape
    
    # Generate output path if not provided
    if output_path is None:
        dataset_name = Path(data_path).stem.replace('_binary', '')
        output_path = f"experiment/ground_truth/exp2_gt_{dataset_name}_{similarity_score}.json"
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON
    data_to_save = {
        "experiment": 2,
        "data_path": data_path,
        "similarity_score": similarity_score,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_train": len(train_indices),
        "n_query": len(query_indices),
        "train_indices": train_indices.tolist(),
        "query_indices": query_indices.tolist(),
        "similarities": similarities.tolist()
    }
    
    # Save to JSON
    print(f"\nSaving ground truth to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"✓ Saved similarity matrix ({similarities.shape[0]}, {similarities.shape[1]})")
    
    return output_path


def calculate_and_save_experiment2_ground_truth(
    data_path: str,
    train_ratio: float,
    similarity_score: str,
    seed: int = 42,
    output_path: str = None,
    use_gpu: bool = False
) -> str:
    """
    Calculate and save Experiment 2 ground truth.
    
    Args:
        data_path: Path to .npy file containing binary matrix
        train_ratio: Ratio for training set
        similarity_score: Similarity metric
        seed: Random seed
        output_path: Output JSON file path (default: auto-generated)
        use_gpu: Whether to use GPU for computations
        
    Returns:
        Path to saved JSON file
    """
    # Load data
    print(f"Loading matrix from {data_path}...")
    X_dense = np.load(data_path)
    print(f"  Matrix Shape: {X_dense.shape}")
    
    # Split dataset
    np.random.seed(seed)
    n_samples = X_dense.shape[0]
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    query_indices = indices[n_train:]
    
    X_train = X_dense[train_indices]
    X_query = X_dense[query_indices]
    
    print(f"Dataset split: {len(train_indices)} training, {len(query_indices)} query")
    
    # Calculate ground truth similarities
    similarities = calculate_experiment2_ground_truth(
        X_train, X_query, similarity_score, use_gpu
    )
    
    # Save to file
    return save_experiment2_ground_truth(
        data_path, train_indices, query_indices,
        similarity_score, similarities, output_path
    )


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and save ground truth similarities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--experiment', type=int, choices=[1, 2], default=1,
        help='Experiment type: 1 (all pairs) or 2 (train/query split)'
    )
    
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to the .npy file containing the binary matrix'
    )
    
    parser.add_argument(
        '--similarity_score', type=str, required=True,
        choices=['cosine_similarity', 'jaccard_similarity', 'inner_product'],
        help='Similarity metric for ground truth calculation'
    )
    
    parser.add_argument(
        '--train_ratio', type=float, default=0.9,
        help='Training set ratio (for experiment 2)'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (for experiment 2)'
    )
    
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--load', type=str, default=None,
        help='Load and display ground truth from existing JSON file'
    )
    
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Use GPU for computations if available'
    )
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing ground truth
        load_experiment1_ground_truth(args.load)
    elif args.experiment == 1:
        # Calculate and save Experiment 1 ground truth
        calculate_and_save_experiment1_ground_truth(
            args.data_path,
            args.similarity_score,
            args.output,
            args.use_gpu
        )
    elif args.experiment == 2:
        # Calculate and save Experiment 2 ground truth
        calculate_and_save_experiment2_ground_truth(
            args.data_path,
            args.train_ratio,
            args.similarity_score,
            args.seed,
            args.output,
            args.use_gpu
        )


if __name__ == '__main__':
    main()
