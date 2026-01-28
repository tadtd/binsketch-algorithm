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

from src import cosine_similarity, jaccard_similarity, inner_product


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
    similarity_score: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ground truth similarities for all pairs (Experiment 1).
    Uses similarity functions from src.similarity_scores module.
    
    Args:
        X_dense: Dense binary matrix (n_samples, n_features)
        similarity_score: Similarity metric ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        
    Returns:
        Tuple of (ground_truth_values, pair_indices)
    """
    n_samples = X_dense.shape[0]
    pairs = list(combinations(range(n_samples), 2))
    n_pairs = len(pairs)
    
    print(f"Calculating Ground Truth ({similarity_score}) for {n_pairs} pairs...")
    
    # Get similarity function
    similarity_func = _get_similarity_function(similarity_score)
    
    # Compute similarities for all pairs
    ground_truth = []
    for i, j in tqdm(pairs, desc="Computing ground truth", unit="pair"):
        val = similarity_func(X_dense[i], X_dense[j])
        ground_truth.append(val)
    
    return np.array(ground_truth), np.array(pairs)


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
    output_path: str = None
) -> str:
    """
    Calculate Experiment 1 ground truth similarities and save to JSON.
    
    Args:
        data_path: Path to .npy file containing binary matrix
        similarity_score: Similarity metric ('cosine_similarity', 'jaccard_similarity', 'inner_product')
        output_path: Output JSON file path (default: auto-generated)
        
    Returns:
        Path to saved JSON file
    """
    # Load data
    print(f"Loading matrix from {data_path}...")
    X_dense = np.load(data_path)
    print(f"  Matrix Shape: {X_dense.shape}")
    
    # Calculate ground truth
    ground_truth, pairs = calculate_experiment1_ground_truth(X_dense, similarity_score)
    
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
    similarity_score: str
) -> np.ndarray:
    """
    Calculate ground truth similarities for Experiment 2 (train/query split).
    Computes similarity matrix between query and training vectors.
    Uses similarity functions from src.similarity_scores module.
    
    Args:
        X_train: Training vectors (n_train, n_features)
        X_query: Query vectors (n_query, n_features)
        similarity_score: Similarity metric
        
    Returns:
        Similarity matrix (n_query, n_train)
    """
    print(f"Calculating Experiment 2 Ground Truth ({similarity_score})...")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Query samples: {X_query.shape[0]}")
    
    # Get similarity function
    similarity_func = _get_similarity_function(similarity_score)
    
    # Compute similarity matrix
    n_query = len(X_query)
    n_train = len(X_train)
    similarities = np.zeros((n_query, n_train))
    
    for i in tqdm(range(n_query), desc="Computing similarities"):
        for j in range(n_train):
            similarities[i, j] = similarity_func(X_query[i], X_train[j])
    
    print(f"  Min similarity: {similarities.min():.6f}")
    print(f"  Max similarity: {similarities.max():.6f}")
    print(f"  Mean similarity: {similarities.mean():.6f}")
    
    return similarities


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
    output_path: str = None
) -> str:
    """
    Calculate and save Experiment 2 ground truth.
    
    Args:
        data_path: Path to .npy file containing binary matrix
        train_ratio: Ratio for training set
        similarity_score: Similarity metric
        seed: Random seed
        output_path: Output JSON file path (default: auto-generated)
        
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
        X_train, X_query, similarity_score
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
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing ground truth
        load_experiment1_ground_truth(args.load)
    elif args.experiment == 1:
        # Calculate and save Experiment 1 ground truth
        calculate_and_save_experiment1_ground_truth(
            args.data_path,
            args.similarity_score,
            args.output
        )
    elif args.experiment == 2:
        # Calculate and save Experiment 2 ground truth
        calculate_and_save_experiment2_ground_truth(
            args.data_path,
            args.train_ratio,
            args.similarity_score,
            args.seed,
            args.output
        )


if __name__ == '__main__':
    main()
